
import shutil
import sys
import os
import json
import numpy as np
import paths as paths
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import paths
import data_manager
from paths import path_constants as pc
from particleGPT.dictionary import Dictionary

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = os.cpu_count() # 256--old hardcoded value
IO_BUFFER = 16 * 1024 * 1024  # 16 MB
COPY_BUFFER = 256 * 1024 * 1024  # 256 MB
    
def analyze_dataset(dataset_filepath, delimiter = ';'):
    num_events = 0
    num_particles_max = 0
    with open(dataset_filepath, 'r', buffering=1024*1024) as file:
        for line in tqdm(file, desc="Analyzing dataset", unit=" events", unit_scale=True):
            line = line.rstrip('\n')
            if not line:
                continue
            num_events += 1
            col_count = 1 + line.count(delimiter)
            if col_count > num_particles_max:
                num_particles_max = col_count

    return num_events, num_particles_max

def make_byte_ranges(filepath, n_ranges):
    """
    Create n_ranges equal-sized bytes. Size is derived from the file @ filepath.
    """
     
    if n_ranges <= 0:
        raise ValueError("n_ranges must be positive!")
     
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return []
     
    n_ranges = min(n_ranges, file_size)
    chunk_size = file_size // n_ranges

    ranges = []
    for i in range(n_ranges):
        start = i * chunk_size
        end = file_size if i == n_ranges - 1 else (i + 1) * chunk_size
        ranges.append((i, start, end))
    return ranges

class EUsageType(Enum):
    ENONE    = 0,
    EEncoder = 1,
    EDecoder = 2
    
@dataclass
class DecodeEventResult():
    event: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    event_idx: int = -1
    success: bool = False
    failure_reason: str | None = None


class BaseTokenizer():
    
    def __init__(
        self,
        dictionary: Dictionary, 
        temp_dir: Path,
        sequence_length: int | None = None,
        dtype: np.dtype = np.uint16,
        max_buffer_len: int = 1024 ** 2,
    ):
        self.usage_type = EUsageType.ENONE
        self.dictionary = dictionary
        
        self.total_sequences_written = 0
        self.total_events_written = 0
        self.total_tokens_written = 0
        
        self.temp_dir = temp_dir
        self.sequence_length = sequence_length
        self.dtype = dtype
        self.max_buffer_len = max_buffer_len
        
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.max_buffer_len < 0:
            raise ValueError("max_buffer_len must be positive!")
            
    def prepare_temp(self):
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*.bin"):
                f.unlink()
            for f in self.temp_dir.glob("*.tmp"):
                f.unlink()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def build_caches(self):
        """For performance, we cache whatever we can here"""
        pass
    
    # =====================
    # Encoding
    # =====================
    
    def _encode_byte_range_worker(self, worker_idx, input_data_filepath: Path, start_byte, end_byte):
        """
        Encodes all complete events present in the byte range (start_byte, end_byte).
        Event @ start_byte is ignored if it is a partial event.
        Event @ end_byte is NOT ignored even if it is a partial event.
        The above setup ensures all events are encoded.
        """
        temp_output_data_filepath = self.temp_dir / f"token_stream_batch_{worker_idx}.tmp"
        temp_output_lens_filepath = self.temp_dir / f"lens_stream_batch_{worker_idx}.tmp"
        output_data_filepath = self.temp_dir / f"token_stream_batch_{worker_idx}.bin"
        output_lens_filepath = self.temp_dir / f"lens_stream_batch_{worker_idx}.bin"

        num_events_written = 0
        num_tokens_written = 0
        token_buffer: list[int] = []
        event_lens_buffer: list[int] = []
        with input_data_filepath.open("rb", buffering=IO_BUFFER) as in_file, \
             temp_output_data_filepath.open("wb", buffering=IO_BUFFER) as out_data_file, \
             temp_output_lens_filepath.open("wb", buffering=IO_BUFFER) as out_lens_file:

            if start_byte == 0:
                in_file.seek(0)
            else:
                # Check whether start_byte is already at the beginning of a line.
                in_file.seek(start_byte - 1)
                prev_byte = in_file.read(1)

                # If previous byte is not newline, we landed in the middle of an event,
                # so discard the partial line.
                if prev_byte != b"\n":
                    in_file.readline()

            while in_file.tell() < end_byte:
                line = in_file.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    event = np.fromstring(line.replace(b";", b" "), sep=" ", dtype=np.float64)
                except ValueError as exc:
                    raise RuntimeError(f"Worker {worker_idx} failed to parse line near byte {in_file.tell()}: {line[:500]}") from exc

                tokenized_event = self.encode_event(event)
                if tokenized_event is None or len(tokenized_event) == 0:
                    continue
                
                token_buffer.extend(tokenized_event)
                event_lens_buffer.append(len(tokenized_event))
                
                num_events_written += 1
                num_tokens_written += len(tokenized_event)
                
                if len(token_buffer) >= self.max_buffer_len:
                    self.flush_binary_buffer(token_buffer, out_data_file, self.dtype)
                    token_buffer.clear()
                if len(event_lens_buffer) >= self.max_buffer_len:
                    self.flush_binary_buffer(event_lens_buffer, out_lens_file, self.dtype)
                    event_lens_buffer.clear()
                    
            # Flush final buffer
            if token_buffer:
                self.flush_binary_buffer(token_buffer, out_data_file, self.dtype)
                token_buffer.clear()
            if event_lens_buffer:
                self.flush_binary_buffer(event_lens_buffer, out_lens_file, self.dtype)
                event_lens_buffer.clear()
                
        temp_output_data_filepath.replace(output_data_filepath)
        temp_output_lens_filepath.replace(output_lens_filepath)
        return worker_idx, num_events_written, num_tokens_written
    
    def encode_dataset(self, input_data_filepath: Path) -> None:
        """
        Handles encoding events using byte-range parallelism.

        Each worker seeks to a different region of the file, discards one
        possible partial first line, and then processes complete event lines via
        `encode_byte_range_worker`
        """
        
        self.prepare_temp()        
        self.usage_type = EUsageType.EEncoder
        
        # ===== Run default checks =====
        # Ensure provided dtype can handle our token values (0, vocab_size)
        dtype_info = np.iinfo(self.dtype)
        if self.dictionary.vocab_size - 1 > dtype_info.max:
            raise ValueError(
                f"Provided dtype={self.dtype} cannot handle the vocab size of {self.dictionary.vocab_size}.\n"
                f"dtype range=[{dtype_info.min}, {dtype_info.max}]"
            )
            
        # ===== Encode dataset =====
        
        self.input_data_filepath = input_data_filepath            
        ranges = make_byte_ranges(self.input_data_filepath, N_WORKERS)
        
        print(f"Launching {N_WORKERS} workers")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [
                executor.submit(self._encode_byte_range_worker, worker_idx, self.input_data_filepath, start_byte, end_byte)
                for worker_idx, start_byte, end_byte in ranges
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
                worker_idx, num_events_written, num_tokens_written = f.result()
                self.total_events_written += num_events_written
                self.total_tokens_written += num_tokens_written

        print(f"Tokenized {self.total_events_written:,} valid events, composed of {self.total_tokens_written:,} tokens.")
        print(f"Concatenating data.")
        
        # ===== Concatenate encoded dataset =====
        
        worker_indices = [worker_idx for worker_idx, _, _ in ranges]
        concat_tokens, concat_events = self.concatenate_encoded_batches(worker_indices)
        if concat_tokens != self.total_tokens_written:
            raise RuntimeError(
                f"Token count mismatch: workers reported {self.total_tokens_written}, "
                f"but concatenated file contains {concat_tokens}."
            )
        if concat_events != self.total_events_written:
            raise RuntimeError(
                f"Event count mismatch: workers reported {self.total_events_written}, "
                f"but concatenated file contains {concat_events}."
            )

    def encode_event(self, event: list[float]) -> list[int] | None:
        raise NotImplementedError("Subclasses must implement this method.")

    def verify_data(self):
        concat_data_filepath = self.temp_dir / 'concatenated_data.bin'
        concat_lens_filepath = self.temp_dir / 'concatenated_lens.bin'
        
        data = np.memmap(concat_data_filepath, mode='r', dtype=self.dtype)
        data_len = len(data)
        del data
        
        if data_len % self.sequence_length != 0:
            raise RuntimeError(
                f"Final token count {data_len} is not divisible by sequence_length {self.sequence_length}. "
                f"Some tokens will be dropped when reshaping into sequences."
            )
            
        self.total_sequences_written = int(data_len // self.sequence_length)
        # This tokenizer should have num_events == num_sequences since its one event per sequence
        if self.total_sequences_written != self.total_events_written:
            raise RuntimeError(
                f"Inconsistent event and sequence counts: {self.total_events_written} events, {self.total_sequences_written} sequences. "
                "EventPerSequenceParticleFeatureTokenizer should have num_events == num_sequences since its one event per sequence."
            )

    # =====================
    # Decoding
    # =====================
    
    def decode_dataset_from_file(self, sampled_data_filepath: Path):
        """
        For the time being, I am not worried about memory usage. We do not decode enough events for now.
        @TODO: implement a memory-efficient decoding method later on.
        """
        
        if not sampled_data_filepath.exists():
            raise FileNotFoundError(f"Sampled data file does not exist: {sampled_data_filepath}")
        self.input_data_filepath = sampled_data_filepath
        
        # ===== Decode dataset =====
        
        # @TODO: make this load binary data using the datatype stored by sample once I make sampler output binary data
        match sampled_data_filepath.suffix:
            case ".csv":
                data = np.loadtxt(sampled_data_filepath, delimiter=' ', dtype=np.int32)
            case ".bin":
                data = np.memmap(sampled_data_filepath, mode='r', dtype=np.int32)
            case __:
                raise NotImplementedError("This filetype is not support for datasets!")
        
        self.decode_dataset(data)
        
    def decode_dataset(self, data: np.ndarray):
        """
        For the time being, I am not worried about memory usage. We do not decode enough events for now.
        Decode data provided as an np.ndarray. Does not require reading files.
        @TODO: implement a memory-efficient decoding method later on.
        """
        
        self.usage_type = EUsageType.EDecoder
        self.build_caches()
        
        # In principle, this will handle packed streams as well as even per row.
        # More precisely, this converts the input to a packed stream and decodes it
        data_flat = data.ravel()
        data_flat = data_flat[data_flat != self.dictionary.padding_token]
        event_start_idxs = np.flatnonzero(data_flat == self.dictionary.event_start_token)
        
        decoded_events: list[list] = []
        failed_events: list[DecodeEventResult] = []
        event_bounds = np.r_[event_start_idxs, len(data_flat)]
        for idx in range(len(event_start_idxs)):
            event = data_flat[event_bounds[idx]:event_bounds[idx + 1]]
            decode_result = self.decode_event(event)
            decode_result.event_idx = idx
            if not decode_result.success:
                failed_events.append(decode_result)
                continue
            decoded_events.append(decode_result.event)
            self.total_events_written += 1
            self.total_tokens_written += len(decode_result.event)
            
        self.decoded_dataset = decoded_events
        self.failed_events = failed_events
    
    def decode_event(self, tokens: list[int]) -> DecodeEventResult:
        raise NotImplementedError("Subclasses must implement this method.")

    # =====================
    # Common
    # =====================

    def flush_binary_buffer(self, token_buffer: list[int], out_file, dtype):
        """
        Write buffered token IDs to a binary file. Assumes out_file is opened when this function is called.
        """
        if not token_buffer:
            return None
        np.asarray(token_buffer, dtype=dtype).tofile(out_file)
    
    def concatenate_encoded_batches(self, worker_indices: list[int]) -> int:
        """
        Concatenate worker .bin streams in worker-index order.
        Both the token stream and lens stream will be concatenated in the same order, ensuring they remain aligned.

        This preserves global event/token order because worker_idx corresponds to
        increasing byte ranges in the original input file.
        
        The output is a flattened sequence of encoded events. To read properly, use the
        associated metadata file.
        """
        token_stream_files = [
            self.temp_dir / f"token_stream_batch_{worker_idx}.bin"
            for worker_idx in sorted(worker_indices)
        ]
        lens_stream_files = [
            self.temp_dir / f"lens_stream_batch_{worker_idx}.bin"
            for worker_idx in sorted(worker_indices)
        ]
        
        missing_tokens_files = [p for p in token_stream_files if not p.exists()]
        if missing_tokens_files:
            raise RuntimeError(
                "Missing expected worker token stream output files:\n"
                + "\n".join(str(p) for p in missing_tokens_files)
            )
        missing_lens_files = [p for p in lens_stream_files if not p.exists()]
        if missing_lens_files:
            raise RuntimeError(
                "Missing expected worker lens stream output files:\n"
                + "\n".join(str(p) for p in missing_lens_files)
            )
        
        itemsize = np.dtype(self.dtype).itemsize
        total_tokens = 0
        total_events = 0

        out_data_file = self.temp_dir / 'concatenated_data.bin'
        with open(out_data_file, "wb", buffering=0) as outfile:
            for file_path in tqdm(token_stream_files, desc="Concatenating binary data streams"):
                file_size = file_path.stat().st_size

                if file_size % itemsize != 0:
                    raise RuntimeError(
                        f"Worker file {file_path} has size {file_size} bytes, "
                        f"which is not divisible by dtype itemsize {itemsize}."
                    )

                total_tokens += file_size // itemsize

                with open(file_path, "rb", buffering=0) as infile:
                    shutil.copyfileobj(infile, outfile, length=COPY_BUFFER)
        
        out_lens_file = self.temp_dir / 'concatenated_lens.bin'
        with open(out_lens_file, "wb", buffering=0) as outfile:
            for file_path in tqdm(lens_stream_files, desc="Concatenating binary lens streams"):
                file_size = file_path.stat().st_size

                if file_size % itemsize != 0:
                    raise RuntimeError(
                        f"Worker file {file_path} has size {file_size} bytes, "
                        f"which is not divisible by dtype itemsize {itemsize}."
                    )

                total_events += file_size // itemsize

                with open(file_path, "rb", buffering=0) as infile:
                    shutil.copyfileobj(infile, outfile, length=COPY_BUFFER)

        return total_tokens, total_events

    def postprocess_data(self):
        """
        Takes final concatenated encoded events and runs through post processing.
        This method should me implemented per tokenizer depending on the postprocessing
        required.
        
        Does some basic validation to ensure the data is probably good-to-go
        
        NOTE: This should calculate self.total_sequences_written.
        """
        concat_data_filepath = self.temp_dir / 'concatenated_data.bin'
        concat_lens_filepath = self.temp_dir / 'concatenated_lens.bin'
        
        data = np.memmap(concat_data_filepath, mode='r', dtype=self.dtype)
        data_len = len(data)
        del data
        
        if data_len % self.sequence_length != 0:
            raise RuntimeError(
                f"Final token count {data_len} is not divisible by sequence_length {self.sequence_length}. "
                f"Some tokens will be dropped when reshaping into sequences."
            )
            
        self.total_sequences_written = int(data_len // self.sequence_length)
        # This tokenizer should have num_events == num_sequences since its one event per sequence
        if self.total_sequences_written != self.total_events_written:
            raise RuntimeError(
                f"Inconsistent event and sequence counts: {self.total_events_written} events, {self.total_sequences_written} sequences. "
                "EventPerSequenceParticleFeatureTokenizer should have num_events == num_sequences since its one event per sequence."
            )
    
    
    def _save_data_encoder(self):
        match self.output_data_filepath.suffix:
            case ".csv":
                raise NotImplementedError(".csv files are not supported!")
            case ".bin":
                # Simply copy the flattened file to the final filepath
                concat_data_filepath = self.temp_dir / 'concatenated_data.bin'
                concat_lens_filepath = self.temp_dir / 'concatenated_lens.bin'

                self.output_lens_filepath = self.output_data_filepath.parent / 'tokenized_lens.bin'
                shutil.copyfile(concat_data_filepath, self.output_data_filepath)
                shutil.copyfile(concat_lens_filepath, self.output_lens_filepath)
            case ".npy":
                raise NotImplementedError(".npy files are not supported!")
            case __:
                raise NotImplementedError("This output format is not supported!")
            
    def _dataset_padder(self, in_dataset):
        # pad the dataset to convert to np.array
        max_particle_count = np.max(len(e) for e in in_dataset)
        max_values_count = max_particle_count * NUM_FEATURES_PER_PARTICLE_RAW
        for i, e in in_dataset:
            pad_shape = (0, max_values_count - len(e))
            in_dataset[i] = np.pad(e, pad_shape, "constant")
        return np.array(in_dataset)
        
    def _save_data_decoder(self):
        data_manager.save_geant4_dataset(self.decoded_dataset, self.output_data_filepath)
            
        # Write invalid events file
        failed_dict = []
        for fe in self.failed_events:
            failed_dict.append({
                'event_idx': fe.event_idx,
                'failure_reason': fe.failure_reason,
                'event_row': " ".join(str(v) for v in fe.event),
            })
        
        if self.invalid_data_filepath is None:
            self.invalid_data_filepath = self.output_data_filepath.parent / 'invalid_token_events.json'
        with self.invalid_data_filepath.open('w') as invalid_out:
            json.dump(failed_dict, invalid_out, indent=4)
    
    def save_data(self, output_data_filepath: Path, invalid_data_filepath: Path | None = None, skip_write_metadata: bool = False):
        self.output_data_filepath = output_data_filepath
        self.invalid_data_filepath = invalid_data_filepath
        
        match self.usage_type:
            case EUsageType.EEncoder:
                self._save_data_encoder()
                output_metadata_filepath = self.output_data_filepath.parent / 'tokenized_data.bin.json'
            case EUsageType.EDecoder:
                self._save_data_decoder()
                output_metadata_filepath = self.output_data_filepath.parent / paths.as_json(pc.samples_decoded_filename)
            case EUsageType.ENONE:
                raise RuntimeError("usage_type was not set! Has the tokenizer been used yet?")
        
        # in case the user wants to save it in a location of their choice they can call the function directly
        if not skip_write_metadata:
            self.write_metadata(output_metadata_filepath)
        
        
    def write_metadata(self, output_metadata_filepath: Path):
        """
        Write sidecar metadata needed to safely reload the raw binary file.
        """
        if self.usage_type == EUsageType.ENONE:
            raise RuntimeError("usage_type was not set! Has the tokenizer been used yet?")
        
        if self.usage_type == EUsageType.EEncoder:
            metadata = {
                "tokenizer_class": type(self).__name__,
                "tokenization_schema": self.dictionary.tokenization_schema,
                "tokenized_dtype": np.dtype(self.dtype).name,
                "byte_order": sys.byteorder,
                "vocab_size": self.dictionary.vocab_size,
                "sequence_length": self.sequence_length,
                "total_sequences": int(self.total_sequences_written),
                "total_events": int(self.total_events_written),
                "total_tokens": int(self.total_tokens_written),
                "num_full_sequences": int(self.total_tokens_written // self.sequence_length),
                "remainder_tokens": int(self.total_tokens_written % self.sequence_length),
                "input_data_filepath": str(paths.project_relative_path(self.input_data_filepath)),
                "output_data_filepath": str(paths.project_relative_path(self.output_data_filepath)),
                "tokenized_lens_file": str(paths.project_relative_path(self.output_lens_filepath)),
                "dictionary_filepath":  str(paths.project_relative_path(self.dictionary.dictionary_filename))
            }
        elif self.usage_type == EUsageType.EDecoder:
            input_data_source = (
                paths.project_relative_path(self.input_data_filepath)
                if self.input_data_filepath is not None and self.input_data_filepath.exists()
                else "none"
            )
            output_data_loc = (
                paths.project_relative_path(self.output_data_filepath)
                if self.output_data_filepath is not None and self.output_data_filepath.exists()
                else "none"
            )
            invalid_events_loc = (
                paths.project_relative_path(self.invalid_data_filepath)
                if self.invalid_data_filepath is not None and self.invalid_data_filepath.exists()
                else "none"
            )
            metadata = {
                "input_data_filepath": input_data_source,
                "output_data_filepath": output_data_loc,
                "invalid_events_filepath": invalid_events_loc,
                "tokenization_schema": self.dictionary.tokenization_schema,
                # "float_precision": self.float_precision, @TODO: implement this
                "total_events_written": self.total_events_written,
                "total_tokens_written": self.total_tokens_written,
                # "total_empty_samples": self.total_empty_samples,
                "total_invalid_events": len(self.failed_events),
            }
        
        with output_metadata_filepath.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"Wrote metadata: {output_metadata_filepath}")
