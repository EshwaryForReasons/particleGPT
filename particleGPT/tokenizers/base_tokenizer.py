
import shutil
import sys
import os
import json
import numpy as np
import paths as paths
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

from particleGPT.dictionary import Dictionary
from particleGPT.tokenizers.tokenizer_paths import TokenizerPaths

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

class BaseTokenizer():
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: TokenizerPaths,
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
        if self.flush_lens <= 0:
            raise ValueError("flush_lens must be positive.")
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
    
    def encode_event(self, tokens: list[float]) -> list[int] | None:
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
            for f in self.in_paths.temp_data_dir.glob("lens_stream_batch_*.bin"):
                f.unlink()
            for f in self.in_paths.temp_data_dir.glob("lens_stream_batch_*.tmp"):
                f.unlink()
            for f in self.in_paths.temp_data_dir.glob("concatenated_data.bin"):
                f.unlink()
            for f in self.in_paths.temp_data_dir.glob("concatenated_lens.bin"):
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
                if len(event_lens_buffer) >= self.flush_lens:
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
            "tokenized_lens_file": str(paths.project_relative_path(self.in_paths.tokenized_lens_filepath)),
            "dictionary_file": str(paths.project_relative_path(self.in_paths.dictionary_filepath))
        }
        
        metadata_path = self.in_paths.tokenized_data_filepath.with_suffix(self.in_paths.tokenized_data_filepath.suffix + ".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Wrote metadata: {metadata_path}")
