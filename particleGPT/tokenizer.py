"""
Run from the main project directory with:

python -m particleGPT.tokenizer dictionary.json
"""

import shutil
import sys
import os
import json
import math
import csv
import numpy as np
import paths as paths
from numba import njit, float64
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

from particleGPT.dictionary import Dictionary
import particleGPT.configurator as conf

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = os.cpu_count() # 256--old hardcoded value
IO_BUFFER = 16 * 1024 * 1024  # 16 MB
COPY_BUFFER = 256 * 1024 * 1024  # 256 MB

@njit("int64(float64, float64[:])", cache=True, nogil=True)
def custom_searchsorted(value, bins):
    if value < bins[0]:
        return 0
    elif value >= bins[-1]:
        return len(bins) - 1
    return np.searchsorted(bins, value, side='right')

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

@dataclass
class Paths:
    # Input raw data
    input_data_filepath: Path = None
    # Output tokenized data
    tokenized_data_filepath: Path = None
    # Directory to store temp tokenized files before concatenation
    temp_data_dir: Path = None
    dictionary_filepath: Path = None

class BaseTokenizer():
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: Paths,
        dtype=np.uint16,
        sequence_length: int = 1024,
        flush_tokens: int = 1024 ** 2,
        flush_lens: int = 1024 ** 2,
        clean_temp_dir: bool = True
    ):
        self.dictionary = dictionary
        self.in_paths = in_paths
        self.dtype = np.dtype(dtype)
        self.flush_tokens = int(flush_tokens)
        self.flush_lens = int(flush_lens)
        self.sequence_length = int(sequence_length)
        self.clean_temp_dir = clean_temp_dir
        
        self.total_sequences_written = 0
        self.total_events_written = 0
        self.total_tokens_written = 0
        
        if self.flush_tokens <= 0:
            raise ValueError("flush_tokens must be positive.")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"dtype must be an integer dtype, got {self.dtype}")
        
        # Ensure provided dtype can handle our token values (0, vocab_size)
        dtype_info = np.iinfo(self.dtype)
        if self.dictionary.vocab_size - 1 > dtype_info.max:
            raise ValueError(
                f"Provided dtype={self.dtype} cannot handle the vocab size of {self.dictionary.vocab_size}.\n"
                f"dtype range=[{dtype_info.min}, {dtype_info.max}]"
            )
    
    def encode_event(self, tokens: list[float]) -> list[int]:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def make_byte_ranges(self, filepath, n_ranges):
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

    def batch_encode_byte_ranges(self):
        """
        Handles encoding events using byte-range parallelism.

        Each worker seeks to a different region of the file, discards one
        possible partial first line, and then processes complete event lines via
        `encode_byte_range_worker`
        """
        if self.clean_temp_dir:
            for f in self.in_paths.temp_data_dir.glob("token_stream_batch_*.bin"):
                f.unlink()
            for f in self.in_paths.temp_data_dir.glob("token_stream_batch_*.tmp"):
                f.unlink()
            for f in self.in_paths.temp_data_dir.glob("concatenated_data.bin"):
                f.unlink()
        
        self.in_paths.temp_data_dir.mkdir(parents=True, exist_ok=True)

        ranges = self.make_byte_ranges(self.in_paths.input_data_filepath, N_WORKERS)
        
        print(f"Launching {N_WORKERS} workers")

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [
                executor.submit(
                    self.encode_byte_range_worker,
                    worker_idx,
                    self.in_paths.input_data_filepath,
                    self.in_paths.temp_data_dir,
                    start_byte,
                    end_byte,
                )
                for worker_idx, start_byte, end_byte in ranges
            ]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
                worker_idx, num_events_written, num_tokens_written = f.result()
                self.total_events_written += num_events_written
                self.total_tokens_written += num_tokens_written

        print(f"Tokenized {self.total_events_written:,} valid events, composed of {self.total_tokens_written:,} tokens.")
        print(f"Concatenating data.")
        
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

    def encode_byte_range_worker(
        self,
        worker_idx,
        input_data_filepath: Path,
        temp_data_dir: Path,
        start_byte,
        end_byte,
    ):
        """
        Encodes all complete events present in the byte range (start_byte, end_byte).
        Event @ start_byte is ignored if it is a partial event.
        Event @ end_type is NOT ignored even if it is a partial event.
        The above setup ensures all events are encoded.
        """
        temp_output_data_filepath = temp_data_dir / f"token_stream_batch_{worker_idx}.tmp"
        temp_output_lens_filepath = temp_data_dir / f"lens_stream_batch_{worker_idx}.tmp"
        output_data_filepath = temp_data_dir / f"token_stream_batch_{worker_idx}.bin"
        output_lens_filepath = temp_data_dir / f"lens_stream_batch_{worker_idx}.bin"

        num_events_written = 0
        num_tokens_written = 0
        token_buffer: list[int] = []
        event_lens_buffer: list[int] = []
        with open(input_data_filepath, "rb", buffering=IO_BUFFER) as in_file, \
             open(temp_output_data_filepath, "wb", buffering=IO_BUFFER) as out_data_file, \
             open(temp_output_lens_filepath, "wb", buffering=IO_BUFFER) as out_lens_file:

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
                    raise RuntimeError(
                        f"Worker {worker_idx} failed to parse line near byte {in_file.tell()}: {line[:500]}"
                    ) from exc

                tokenized_event = self.encode_event(event)
                if tokenized_event is None or len(tokenized_event) == 0:
                    continue
                
                token_buffer.extend(tokenized_event)
                event_lens_buffer.append(len(tokenized_event))
                
                num_events_written += 1
                num_tokens_written += len(tokenized_event)
                
                if len(token_buffer) >= self.flush_tokens:
                    self.flush_binary_tokens(token_buffer, out_data_file)
                    token_buffer.clear()
                if len(event_lens_buffer) >= self.flush_tokens:
                    self.flush_binary_tokens(event_lens_buffer, out_lens_file)
                    event_lens_buffer.clear()
                    
            # Flush final buffer
            if token_buffer:
                self.flush_binary_tokens(token_buffer, out_data_file)
                token_buffer.clear()
            if event_lens_buffer:
                self.flush_binary_tokens(event_lens_buffer, out_lens_file)
                event_lens_buffer.clear()
                
        temp_output_data_filepath.replace(output_data_filepath)
        temp_output_lens_filepath.replace(output_lens_filepath)
        return worker_idx, num_events_written, num_tokens_written

    def flush_binary_tokens(self, token_buffer: list[int], out_file):
        """
        Write buffered token IDs to a binary file. Assumes out_file is opened when this function is called.
        """
        
        if not token_buffer:
            return None

        np.asarray(token_buffer, dtype=self.dtype).tofile(out_file)
    
    def concatenate_encoded_batches(self, worker_indices: list[int]) -> int:
        """
        Concatenate worker .bin streams in worker-index order.
        Both the token stream and lens stream will be concatenated in the same order, ensuring they remain aligned.

        This preserves global event/token order because worker_idx corresponds to
        increasing byte ranges in the original input file.
        
        The output is a flattened sequence of encoded events. To read properly, use the
        associated metadata file.
        """
        # output_data_filepath = self.in_paths.temp_data_dir / f"token_stream_batch_{worker_idx}.bin"
        # output_lens_filepath = self.in_paths.temp_data_dir / f"lens_stream_batch_{worker_idx}.bin"
        
        token_stream_files = [
            self.in_paths.temp_data_dir / f"token_stream_batch_{worker_idx}.bin"
            for worker_idx in sorted(worker_indices)
        ]
        lens_stream_files = [
            self.in_paths.temp_data_dir / f"lens_stream_batch_{worker_idx}.bin"
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
        
        itemsize = self.dtype.itemsize
        total_tokens = 0
        total_events = 0

        out_data_file = self.in_paths.temp_data_dir / 'concatenated_data.bin'
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
        
        out_lens_file = self.in_paths.temp_data_dir / 'concatenated_lens.bin'
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
        
        NOTE: This should calculate self.total_sequences_written.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def write_metadata(self):
        """
        Write sidecar metadata needed to safely reload the raw binary file.
        """
        metadata = {
            "tokenizer_class": type(self).__name__,
            "format": "base_tokenizer",
            "tokenization_schema": self.dictionary.tokenization_schema,
            "dtype": self.dtype.name,
            "byte_order": sys.byteorder,
            "vocab_size": self.dictionary.vocab_size,
            "sequence_length": self.sequence_length,
            "total_sequences": int(self.total_sequences_written),
            "total_events": int(self.total_events_written),
            "total_tokens": int(self.total_tokens_written),
            "num_full_sequences": int(self.total_tokens_written // self.sequence_length),
            "remainder_tokens": int(self.total_tokens_written % self.sequence_length),
            "tokenized_data_file": str(paths.project_relative_path(self.in_paths.tokenized_data_filepath)),
            "dictionary_file": str(paths.project_relative_path(self.in_paths.dictionary_filepath))
        }
        
        metadata_path = self.in_paths.tokenized_data_filepath.with_suffix(self.in_paths.tokenized_data_filepath.suffix + ".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Wrote metadata: {metadata_path}")


class EventPerSequenceParticleFeatureTokenizer(BaseTokenizer):
    """
    This tokenizer does the following:
    1) Each particle has features as defined in dictionary.json.
    2) EVENT_START and EVENT_END tokens are at the start and end of each event.
    3) All events are padded to a uniform sequence length.
    4) Each sequence contains 1 event.
    
    Example (N particles with 4 features per particle.):
    EVENT_START P1_F1 P1_F2 P1_F3 P1_F4 P2_F1 P2_F2 P2_F3 P2_F4 P3_F1 P3_F2 P3_F3 P3_F4 EVENT_END PAD-- PAD-- PAD-- PAD--
    EVENT_START P1_F1 P1_F2 P1_F3 P1_F4 P2_F1 P2_F2 P2_F3 P2_F4 EVENT_END PAD-- PAD-- PAD-- PAD-- PAD-- PAD-- PAD-- PAD--
    EVENT_START P1_F1 P1_F2 P1_F3 P1_F4 P2_F1 P2_F2 P2_F3 P2_F4 P3_F1 P3_F2 P3_F3 P3_F4 EVENT_END PAD-- PAD-- PAD-- PAD--
    """
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: Paths,
        dtype=np.uint16,
        flush_tokens: int = 1024 ** 2,
        clean_temp_dir: bool = True,
    ):
        num_events, num_particles_max = analyze_dataset(in_paths.input_data_filepath)
        print(f"Found {num_events:,} events in input dataset.")
        print(f"Max particles/event: {num_particles_max:,}")
        
        # Handle num_particles_max override
        if dictionary.particle_count_override is not None:
            if dictionary.particle_count_override < num_particles_max:
                raise ValueError(
                    "The particle_count_override in the dictionary must be >= maximum "
                    "number of particles found in the dataset."
                )
            num_particles_max = dictionary.particle_count_override
        
        self.padding_sequence = dictionary.get_padding_sequence()
        if len(self.padding_sequence) == 0:
            raise ValueError("padding_sequence cannot be empty.")
        
        # This tokenizer pads to the max possible particles. +2 to handle EVENT_START and EVENT_END tokens.
        sequence_length = num_particles_max * dictionary.num_tokens_per_particle + 2
        super().__init__(dictionary, in_paths, dtype, sequence_length, flush_tokens, clean_temp_dir)
        
    def tokenize_event(self, event: list[float]) -> list[int]:
        """
        Tokenizes and returns a single event.
        """
        tokenized_event = [self.dictionary.event_start_token]
        assert len(event) % NUM_FEATURES_PER_PARTICLE_RAW == 0, (
            f"Malformed event: got {len(event)} float, which is not divisible by {NUM_FEATURES_PER_PARTICLE_RAW}"
        )
        num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
        
        for particle_idx in range(num_particles):
            pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
            energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
            px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
            py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
            pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
            
            # Finite checks
            if not all(math.isfinite(x) for x in (energy, px, py, pz)):
                return None
            
            r      = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
            pt     = math.sqrt(px ** 2 + py ** 2)
            if pt == 0.0:
                return None
            
            theta  = math.acos(pz / r) if r != 0 else 0.0
            phi    = math.atan2(py, px)
            eta    = math.asinh(pz / pt)
            
            # Eta finite check and constrain
            if not math.isfinite(eta) or abs(eta) > 4:
                return None
            
            particle_index = int(self.dictionary.pdgids_to_index[int(pdgid)])
            for schema in self.dictionary.tokenization_schema:
                if schema == "pdgid":
                    tokenized_event.append(particle_index + self.dictionary.PDGID_OFFSET)
                elif schema == "pt":
                    tokenized_event.append(custom_searchsorted(pt, self.dictionary.pt_bins) + self.dictionary.PT_OFFSET)
                elif schema == "px":
                    tokenized_event.append(custom_searchsorted(px, self.dictionary.px_bins) + self.dictionary.PX_OFFSET)
                elif schema == "py":
                    tokenized_event.append(custom_searchsorted(py, self.dictionary.py_bins) + self.dictionary.PY_OFFSET)
                elif schema == "pz":
                    tokenized_event.append(custom_searchsorted(pz, self.dictionary.pz_bins) + self.dictionary.PZ_OFFSET)
                elif schema == "energy":
                    tokenized_event.append(custom_searchsorted(energy, self.dictionary.e_bins) + self.dictionary.ENERGY_OFFSET)
                elif schema == "eta":
                    tokenized_event.append(custom_searchsorted(eta, self.dictionary.eta_bins) + self.dictionary.ETA_OFFSET)
                elif schema == "theta":
                    tokenized_event.append(custom_searchsorted(theta, self.dictionary.theta_bins) + self.dictionary.THETA_OFFSET)
                elif schema == "phi":
                    tokenized_event.append(custom_searchsorted(phi, self.dictionary.phi_bins) + self.dictionary.PHI_OFFSET)
                elif schema == "particle_start":
                    tokenized_event.append(self.dictionary.particle_start_token)
                elif schema == "particle_end":
                    tokenized_event.append(self.dictionary.particle_end_token)
                else:
                    raise RuntimeError(f"pTokenizer: Unknown tokenization schema: {schema}")
        
        tokenized_event.append(self.dictionary.event_end_token)
        return tokenized_event

    def pad_sequence(self, event: list[float]) -> list[float]:
        """
        event: tokenzied, unpadded event
        num_particles_max: used to determine padding length to ensure uniform sequence lengths
        
        Pads a single event to the maximum number of particles. Padding sequence used is derived
        from the dictionary's tokenization schema.
        """
        
        if event is None or len(event) == 0:
            return None

        len_pad_required = self.sequence_length - len(event)
        if len_pad_required < 0:
            raise RuntimeError(
                f"Tokenized event length {len(event)} exceeds "
                f"sequence_length {self.sequence_length}"
            )
        if len_pad_required % len(self.padding_sequence) != 0:
            raise RuntimeError(
                f"Cannot pad exactly: need {len_pad_required} tokens, "
                f"but padding_sequence has length {len(self.padding_sequence)}."
            )

        num_padding_sequences_required = len_pad_required // len(self.padding_sequence)
        padded_event = event + self.padding_sequence * num_padding_sequences_required
        return padded_event

    def encode_event(self, tokens: list[float]) -> list[int]:
        tokenized_event = self.tokenize_event(tokens)
        padded_event = self.pad_sequence(tokenized_event)
        
        if padded_event is not None and len(padded_event) != self.sequence_length:
            raise RuntimeError(
                f"Padding failed: got {len(padded_event)}, expected {self.sequence_length}."
            )
            
        return padded_event
    
    def postprocess_data(self):
        """
        For this tokenizer, no post processing is required since the encoded events are the data.
        The encoded events already include padding to the correct sequence length.
        Important: a meta-data file needs to be written so the correct sequence length is used
        
        @TODO: Perhaps it will be faster to perform padding here? Like fill a np.zero() matrix with
        our data. It would have to be streamed and likely reconcatenated so I guess thats really no
        different.
        """

        # Simply copy the flattened file to the final filepath
        concat_data_filepath = self.in_paths.temp_data_dir / 'concatenated_data.bin'
        concat_lens_filepath = self.in_paths.temp_data_dir / 'concatenated_lens.bin'
        
        data = np.memmap(concat_data_filepath, mode='r+', dtype=self.dtype)
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
            
        shutil.copyfile(concat_data_filepath, self.in_paths.tokenized_data_filepath)
        shutil.copyfile(concat_lens_filepath, self.in_paths.tokenized_lens_filepath)


class PackedEventStreamParticleFeatureTokenizer(BaseTokenizer):
    """
    The goal is to create natural language style packed tokens.
    
    This tokenizer does the following:
    1) Each particle has features as defined in dictionary.json.
    2) EVENT_START and EVENT_END tokens are at the start and end of each event.
    3) Events are packed into sequences, no padding is applied.
    4) Each sequence contains whatever number of events. Sequences can contain partial events.
    
    Example (N particles with 4 features per particle.):
    EVENT_START P1_F1 P1_F2 P1_F3 P1_F4 P2_F1 P2_F2 P2_F3 P2_F4 P3_F1 P3_F2 P3_F3 P3_F4 EVENT_END EVENT_START P1_F1 P1_F2 P1_F3
    P1_F4 P2_F1 P2_F2 P2_F3 P2_F4 P3_F1 P3_F2 P3_F3 P3_F4 EVENT_END EVENT_START P1_F1 P1_F2 P1_F3 P1_F4 P2_F1 P2_F2 P2_F3 P2_F4
    P3_F1 P3_F2 P3_F3 P3_F4 EVENT_END EVENT_START P1_F1 P1_F2 P1_F3 P1_F4 P2_F1 P2_F2 P2_F3 P2_F4 P3_F1 P3_F2 P3_F3 P3_F4 EVENT_END
    """

    def tokenize_event(self, event: list[float]) -> list[int]:
        """
        Tokenizes and returns a single event.
        """
        tokenized_event = [self.dictionary.event_start_token]
        assert len(event) % NUM_FEATURES_PER_PARTICLE_RAW == 0, (
            f"Malformed event: got {len(event)} float, which is not divisible by {NUM_FEATURES_PER_PARTICLE_RAW}"
        )
        num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
        
        for particle_idx in range(num_particles):
            pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
            energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
            px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
            py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
            pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
            
            # Finite checks
            if not all(math.isfinite(x) for x in (energy, px, py, pz)):
                return None
            
            r      = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
            pt     = math.sqrt(px ** 2 + py ** 2)
            if pt == 0.0:
                return None
            theta  = math.acos(pz / r) if r != 0 else 0.0
            phi    = math.atan2(py, px)
            eta    = math.asinh(pz / pt)
            
            # Eta finite check and constrain
            if not math.isfinite(eta) or abs(eta) > 4:
                return None
            
            particle_index = int(self.dictionary.pdgids_to_index[int(pdgid)])
            for schema in self.dictionary.tokenization_schema:
                if schema == "pdgid":
                    tokenized_event.append(particle_index + self.dictionary.PDGID_OFFSET)
                elif schema == "pt":
                    tokenized_event.append(custom_searchsorted(pt, self.dictionary.pt_bins) + self.dictionary.PT_OFFSET)
                elif schema == "px":
                    tokenized_event.append(custom_searchsorted(px, self.dictionary.px_bins) + self.dictionary.PX_OFFSET)
                elif schema == "py":
                    tokenized_event.append(custom_searchsorted(py, self.dictionary.py_bins) + self.dictionary.PY_OFFSET)
                elif schema == "pz":
                    tokenized_event.append(custom_searchsorted(pz, self.dictionary.pz_bins) + self.dictionary.PZ_OFFSET)
                elif schema == "energy":
                    tokenized_event.append(custom_searchsorted(energy, self.dictionary.e_bins) + self.dictionary.ENERGY_OFFSET)
                elif schema == "eta":
                    tokenized_event.append(custom_searchsorted(eta, self.dictionary.eta_bins) + self.dictionary.ETA_OFFSET)
                elif schema == "theta":
                    tokenized_event.append(custom_searchsorted(theta, self.dictionary.theta_bins) + self.dictionary.THETA_OFFSET)
                elif schema == "phi":
                    tokenized_event.append(custom_searchsorted(phi, self.dictionary.phi_bins) + self.dictionary.PHI_OFFSET)
                elif schema == "particle_start":
                    tokenized_event.append(self.dictionary.particle_start_token)
                elif schema == "particle_end":
                    tokenized_event.append(self.dictionary.particle_end_token)
                else:
                    raise RuntimeError(f"pTokenizer: Unknown tokenization schema: {schema}")
        
        tokenized_event.append(self.dictionary.event_end_token)
        return tokenized_event

    def encode_event(self, tokens: list[float]) -> list[int] | None:
        """
        Tokenize one event, but do NOT pad it.
        """
        return self.tokenize_event(tokens)

    def postprocess_data(self):
        """
        Pad the final sequence such that the data is divisible by sequence_length.
        
        Important: a meta-data file needs to be written so the correct sequence length is used
        """

        concat_data_filepath = self.in_paths.temp_data_dir / 'concatenated_data.bin'
        concat_lens_filepath = self.in_paths.temp_data_dir / 'concatenated_lens.bin'
        data = np.memmap(concat_data_filepath, mode='r', dtype=self.dtype)
        data_len = len(data)
        del data
        
        new_len = data_len
        remainder_tokens = data_len % self.sequence_length
        if remainder_tokens != 0:
            padding_len = self.sequence_length - remainder_tokens
            new_len = data_len + padding_len
            
            warnings.warn(
                f"Warning: total tokens {data_len} is not divisible by sequence_length {self.sequence_length}. "
                f"Padding with {padding_len} additional tokens.",
                RuntimeWarning
            )
            
            data = np.memmap(concat_data_filepath, mode='r+', dtype=self.dtype, shape=(new_len,))
            data[data_len:new_len] = self.dictionary.padding_token
            data.flush()
            del data
        
        self.total_tokens_written = int(new_len)
        self.total_sequences_written = int(new_len // self.sequence_length)
        
        shutil.copyfile(concat_data_filepath, self.in_paths.tokenized_data_filepath)
        shutil.copyfile(concat_lens_filepath, self.in_paths.tokenized_lens_filepath)


class EventPerSequenceWholeParticleTokenizer(BaseTokenizer):
    """
    This tokenizer does the following:
    1) Each unique particle gets a unique token.
        e.g for 20 bins across 3 features we get 20**3=8,000 unique tokens.
    2) EVENT_START and EVENT_END tokens are at the start and end of each event.
    3) All events are padded to a uniform sequence length.
    4) Each sequence contains 1 event.
    
    Example (N particles with 4 features per particle.):
    EVENT_START P_1 P_2 P_3 P_4 EVENT_END PAD-- PAD-- PAD--
    EVENT_START P_1 P_2 P_3 P_4 P_5 P_6 EVENT_END PAD-- PAD--
    EVENT_START P_1 P_2 P_3 EVENT_END PAD-- PAD-- PAD-- PAD--
    """
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: Paths,
        dtype=np.uint16,
        flush_tokens: int = 1024 ** 2,
        clean_temp_dir: bool = True,
    ):
        num_events, num_particles_max = analyze_dataset(in_paths.input_data_filepath)
        print(f"Found {num_events:,} events in input dataset.")
        print(f"Max particles/event: {num_particles_max:,}")
        
        # Handle num_particles_max override
        if dictionary.particle_count_override is not None:
            if dictionary.particle_count_override < num_particles_max:
                raise ValueError(
                    "The particle_count_override in the dictionary must be >= maximum "
                    "number of particles found in the dataset."
                )
            num_particles_max = dictionary.particle_count_override
        
        self.padding_sequence = dictionary.get_padding_sequence()
        # This tokenizer pads to the max possible particles. +2 to handle EVENT_START and EVENT_END tokens.
        sequence_length = num_particles_max * dictionary.num_tokens_per_particle + 2
        super().__init__(dictionary, in_paths, dtype, sequence_length, flush_tokens, clean_temp_dir)
        
    def tokenize_event(self, event: list[float]) -> list[int]:
        """
        Tokenizes and returns a single event.
        
        This version uses the following mixed-radix encoding to ensure a unique
        token per particle:
        
        token = base_offset + eta_bin + (n_eta_bins * pt_bin) + (n_eta_bins * n_pt_bins * phi_bin)
        
        Notably, the PDGID is ignored.
        """
        tokenized_event = [self.dictionary.event_start_token]
        assert len(event) % NUM_FEATURES_PER_PARTICLE_RAW == 0, (
            f"Malformed event: got {len(event)} float, which is not divisible by {NUM_FEATURES_PER_PARTICLE_RAW}"
        )
        num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
        
        for particle_idx in range(num_particles):
            pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
            energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
            px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
            py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
            pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
            
            # Finite checks
            if not all(math.isfinite(x) for x in (energy, px, py, pz)):
                return None
            
            r      = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
            pt     = math.sqrt(px ** 2 + py ** 2)
            if pt == 0.0:
                return None
            theta  = math.acos(pz / r) if r != 0 else 0.0
            phi    = math.atan2(py, px)
            eta    = math.asinh(pz / pt)
            
            # Eta finite check and constrain
            if not math.isfinite(eta) or abs(eta) > 4:
                return None
            
            n_pt_bins = len(self.dictionary.pt_bins)
            n_eta_bins = len(self.dictionary.eta_bins)
            n_phi_bins = len(self.dictionary.phi_bins)
            
            pt_bin = custom_searchsorted(pt, self.dictionary.pt_bins)
            eta_bin = custom_searchsorted(eta, self.dictionary.eta_bins)
            phi_bin = custom_searchsorted(phi, self.dictionary.phi_bins)
            
            token = self.dictionary.ETA_OFFSET + eta_bin + (n_eta_bins * pt_bin) + (n_eta_bins * n_pt_bins * phi_bin)
            tokenized_event.append(token)
        
        tokenized_event.append(self.dictionary.event_end_token)
        return tokenized_event

    def pad_sequence(self, event: list[float]) -> list[float]:
        """
        event: tokenzied, unpadded event
        num_particles_max: used to determine padding length to ensure uniform sequence lengths
        
        Pads a single event to the maximum number of particles. Padding sequence used is derived
        from the dictionary's tokenization schema.
        """
        
        if event is None or len(event) == 0:
            return None

        len_pad_required = self.sequence_length - len(event)
        if len_pad_required < 0:
            raise RuntimeError(
                f"Tokenized event length {len(event)} exceeds "
                f"sequence_length {self.sequence_length}"
            )
        if len_pad_required % len(self.padding_sequence) != 0:
            raise RuntimeError(
                f"Cannot pad exactly: need {len_pad_required} tokens, "
                f"but padding_sequence has length {len(self.padding_sequence)}."
            )

        num_padding_sequences_required = len_pad_required // len(self.padding_sequence)
        padded_event = event + self.padding_sequence * num_padding_sequences_required
        return padded_event

    def encode_event(self, tokens: list[float]) -> list[int]:
        tokenized_event = self.tokenize_event(tokens)
        padded_event = self.pad_sequence(tokenized_event)
        
        if padded_event is not None and len(padded_event) != self.sequence_length:
            raise RuntimeError(
                f"Padding failed: got {len(padded_event)}, expected {self.sequence_length}."
            )
        
        return padded_event
    
    def postprocess_data(self):
        """
        For this tokenizer, no post processing is required since the encoded events are the data.
        The encoded events already include padding to the correct sequence length.
        Important: a meta-data file needs to be written so the correct sequence length is used
        
        @TODO: Perhaps it will be faster to perform padding here? Like fill a np.zero() matrix with
        our data. It would have to be streamed and likely reconcatenated so I guess thats really no
        different.
        """

        # Simply copy the flattened file to the final filepath
        concat_data_filepath = self.in_paths.temp_data_dir / 'concatenated_data.bin'
        concat_lens_filepath = self.in_paths.temp_data_dir / 'concatenated_lens.bin'
        
        # @TODO: set total sequences_written and check it matches total_events_written since its one event per sequence
        
        shutil.copyfile(concat_data_filepath, self.in_paths.tokenized_data_filepath)
        shutil.copyfile(concat_lens_filepath, self.in_paths.tokenized_lens_filepath)

    def write_metadata(self):
        """
        Write sidecar metadata needed to safely reload the raw binary file.
        """
        vocab_size = self.dictionary.ETA_OFFSET + (len(self.dictionary.eta_bins) * len(self.dictionary.pt_bins) * len(self.dictionary.phi_bins))
        metadata = {
            "format": "whole_particle",
            "dtype": self.dtype.name,
            "vocab_size": vocab_size,
            "sequence_length": self.sequence_length,
            "total_sequences": int(self.total_sequences_written),
            "total_events": int(self.total_events_written),
            "total_tokens": int(self.total_tokens_written),
            "num_full_sequences": int(self.total_tokens_written // self.sequence_length),
            "remainder_tokens": int(self.total_tokens_written % self.sequence_length),
            "tokenized_data_filepath": str(self.in_paths.tokenized_data_filepath),
        }
        
        metadata_path = self.in_paths.tokenized_data_filepath.with_suffix(self.in_paths.tokenized_data_filepath.suffix + ".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Wrote metadata: {metadata_path}")


# Main can also do everything in case we only want to tokenize the data but not prepare
# it for training.
if __name__ == "__main__":
    dictionary_filepath = sys.argv[1]
    dictionary = Dictionary(dictionary_filepath)
    
    relevant_paths = Paths(
        input_data_filepath     = paths.PROJECT_DIR / 'data' / 'raw' / dictionary.dataset_name,
        tokenized_data_filepath = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'tokenized_data.bin',
        temp_data_dir           = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'temp',
        dictionary_filepath     = paths.PROJECT_DIR / dictionary_filepath
    )
    humanized_dictionary_filepath = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'humanized_dictionary.txt'
    
    # dictionary.update_dictionary_particle_list(relevant_paths.input_data_filepath, dictionary_filepath)
    dictionary.output_humanized_dictionary(humanized_dictionary_filepath)
    
    tokenizer_class = globals()[dictionary.tokenizer_class_str]
    if tokenizer_class is None:
        raise RuntimeError("tokenizer_class cannot be none!")
    
    tokenizer = tokenizer_class(
        dictionary=dictionary,
        in_paths=relevant_paths,
        dtype=np.uint16,
        clean_temp_dir=True
    )
    tokenizer.batch_encode_byte_ranges()
    tokenizer.postprocess_data()
    tokenizer.write_metadata()