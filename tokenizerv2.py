import shutil
import sys
import os
import json
import math
import csv
import numpy as np
from numba import njit, float64
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from dictionary import Dictionary
import configurator as conf

script_dir = Path(__file__).resolve().parent

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = 256
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

class BaseTokenizer():
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: Paths,
        dtype=np.uint16,
        sequence_length: int = 1024,
        flush_tokens: int = 1024 ** 2,
        clean_temp_dir: bool = False
    ):
        self.dictionary = dictionary
        self.in_paths = in_paths
        self.dtype = np.dtype(dtype)
        self.flush_tokens = int(flush_tokens)
        self.sequence_length = int(sequence_length)
        self.clean_temp_dir = clean_temp_dir
        
        self.total_events_written = 0
        self.total_tokens_written = 0
        
        if self.flush_tokens <= 0:
            raise ValueError("flush_tokens must be positive.")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(f"dtype must be an integer dtype, got {self.dtype}")
    
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
        
        self.concatenate_encoded_batches()

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
        output_data_filepath = temp_data_dir / f"token_stream_batch_{worker_idx}.bin"

        num_events_written = 0
        num_tokens_written = 0
        token_buffer: list[int] = []
        with open(input_data_filepath, "rb", buffering=IO_BUFFER) as in_file, \
             open(output_data_filepath, "wb", buffering=IO_BUFFER) as out_file:

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

                line = line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    event = list(map(
                        float,
                        (
                            feature_str for particle_str in line.split(";")
                            for feature_str in particle_str.split()
                        )
                    ))
                except ValueError as exc:
                    raise RuntimeError(
                        f"Worker {worker_idx} failed to parse line near byte {in_file.tell()}: {line[:500]}"
                    ) from exc

                tokenized_event = self.encode_event(event)
                if tokenized_event is None or len(tokenized_event) == 0:
                    continue
                
                token_buffer.extend(tokenized_event)
                
                num_events_written += 1
                num_tokens_written += len(tokenized_event)
                
                if len(token_buffer) >= self.flush_tokens:
                    self.flush_binary_tokens(token_buffer, out_file)
                    token_buffer.clear()
                    
            # Flush final buffer
            if token_buffer:
                self.flush_binary_tokens(token_buffer, out_file)
                token_buffer.clear()

        return worker_idx, num_events_written, num_tokens_written

    def flush_binary_tokens(self, token_buffer: list[int], out_file):
        """
        Write buffered token IDs to a binary file. Assumes out_file is opened when this function is called.
        """
        
        if not token_buffer:
            return None
        
        arr = np.array(token_buffer, dtype=np.int64)
        
        # Ensure this datatype can handle our token values
        dtype_info = np.iinfo(self.dtype)
        arr_min = int(arr.min())
        arr_max = int(arr.max())
        if arr_min < dtype_info.min or arr_max > dtype_info.max:
            raise ValueError(
                f"Token ID out of range for dtype={self.dtype}: "
                f"min={arr_min}, max={arr_max}, "
                f"dtype range=[{dtype_info.min}, {dtype_info.max}]"
            )
        
        arr.astype(self.dtype, copy=False).tofile(out_file)
    
    def concatenate_encoded_batches(self) -> int:
        """
        Concatenate worker .bin streams in worker-index order.

        This preserves global event/token order because worker_idx corresponds to
        increasing byte ranges in the original input file.
        
        The output is a flattened sequence of encoded events. To read properly, use the
        associated metadata file.
        """
        token_stream_files = sorted(
            [
                f
                for f in self.in_paths.temp_data_dir.iterdir()
                if f.name.startswith("token_stream_batch_")
                and f.name.endswith(".bin")
            ],
            key=lambda x: int(x.stem.split("_")[-1])
        )

        itemsize = self.dtype.itemsize
        total_tokens = 0

        out_file = self.in_paths.temp_data_dir / 'concatenated_data.bin'
        with open(out_file, "wb", buffering=0) as outfile:
            for file_path in tqdm(token_stream_files, desc="Concatenating binary streams"):
                file_size = file_path.stat().st_size

                if file_size % itemsize != 0:
                    raise RuntimeError(
                        f"Worker file {file_path} has size {file_size} bytes, "
                        f"which is not divisible by dtype itemsize {itemsize}."
                    )

                total_tokens += file_size // itemsize

                with open(file_path, "rb", buffering=0) as infile:
                    shutil.copyfileobj(infile, outfile, length=COPY_BUFFER)

        return total_tokens

    def postprocess_data(self):
        """
        Takes final concatenated encoded events and runs through post processing.
        This method should me implemented per tokenizer depending on the postprocessing
        required.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def write_metadata(self):
        """
        Write sidecar metadata needed to safely reload the raw binary file.
        """
        metadata = {
            "format": "base_tokenizer",
            "dtype": self.dtype.name,
            "vocab_size": dictionary.vocab_size,
            "sequence_length": self.sequence_length,
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


class EventPerSequenceParticleFeatureTokenizer(BaseTokenizer):
    """
    """
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: Paths,
        dtype=np.uint16,
        flush_tokens: int = 1024 ** 2,
        clean_temp_dir: bool = False,
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
        """
        tokenized_event = [dictionary.event_start_token]
        num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
        
        for particle_idx in range(num_particles):
            pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
            energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
            px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
            py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
            pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
            
            r      = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
            pt     = math.sqrt(px ** 2 + py ** 2)
            theta  = math.acos(pz / r) if r != 0 else 0.0
            phi    = math.atan2(py, px)
            eta    = -math.log(math.tan(theta / 2)) if theta != 0 else 0.0
            
            # Eta constrain
            if abs(eta) > 4:
                return []
            
            particle_index = int(dictionary.pdgids_to_index[int(pdgid)])
            for schema in dictionary.tokenization_schema:
                if schema == "pdgid":
                    tokenized_event.append(particle_index + dictionary.PDGID_OFFSET)
                elif schema == "pt":
                    tokenized_event.append(custom_searchsorted(pt, dictionary.pt_bins) + dictionary.PT_OFFSET)
                elif schema == "px":
                    tokenized_event.append(custom_searchsorted(px, dictionary.px_bins) + dictionary.PX_OFFSET)
                elif schema == "py":
                    tokenized_event.append(custom_searchsorted(py, dictionary.py_bins) + dictionary.PY_OFFSET)
                elif schema == "pz":
                    tokenized_event.append(custom_searchsorted(pz, dictionary.pz_bins) + dictionary.PZ_OFFSET)
                elif schema == "energy":
                    tokenized_event.append(custom_searchsorted(energy, dictionary.e_bins) + dictionary.ENERGY_OFFSET)
                elif schema == "eta":
                    tokenized_event.append(custom_searchsorted(eta, dictionary.eta_bins) + dictionary.ETA_OFFSET)
                elif schema == "theta":
                    tokenized_event.append(custom_searchsorted(theta, dictionary.theta_bins) + dictionary.THETA_OFFSET)
                elif schema == "phi":
                    tokenized_event.append(custom_searchsorted(phi, dictionary.phi_bins) + dictionary.PHI_OFFSET)
                elif schema == "particle_start":
                    tokenized_event.append(dictionary.particle_start_token)
                elif schema == "particle_end":
                    tokenized_event.append(dictionary.particle_end_token)
                else:
                    raise RuntimeError(f"pTokenizer: Unknown tokenization schema: {schema}")
        
        tokenized_event.append(dictionary.event_end_token)
        return tokenized_event

    def pad_sequence(self, event: list[float]) -> list[float]:
        """
        event: tokenzied, unpadded event
        num_particles_max: used to determine padding length to ensure uniform sequence lengths
        
        Pads a single event to the maximum number of particles. Padding sequence used is derived
        from the dictionary's tokenization schema.
        """
        
        if len(event) == 0:
            return None

        len_pad_required = self.sequence_length - len(event)
        if len_pad_required < 0:
            raise RuntimeError(
                f"Tokenized event length {len(event)} exceeds "
                f"sequence_length {self.sequence_length}"
            )

        num_padding_sequences_required = len_pad_required // len(self.padding_sequence)
        padded_event = event + self.padding_sequence * num_padding_sequences_required
        return padded_event

    def encode_event(self, tokens: list[float]) -> list[int]:
        tokenized_event = self.tokenize_event(tokens)
        padded_event = self.pad_sequence(tokenized_event)
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
        shutil.copyfile(concat_data_filepath, self.in_paths.tokenized_data_filepath)
    
    
class PackedEventStreamParticleFeatureTokenizer(BaseTokenizer):
    """
    """

    def tokenize_event(self, event: list[float]) -> list[int]:
        """
        Tokenizes and returns a single event.
        """
        tokenized_event = [dictionary.event_start_token]
        num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
        
        for particle_idx in range(num_particles):
            pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
            energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
            px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
            py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
            pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
            
            r      = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
            pt     = math.sqrt(px ** 2 + py ** 2)
            theta  = math.acos(pz / r) if r != 0 else 0.0
            phi    = math.atan2(py, px)
            eta    = -math.log(math.tan(theta / 2)) if theta != 0 else 0.0
            
            # Eta constrain
            if abs(eta) > 4:
                return []
            
            particle_index = int(dictionary.pdgids_to_index[int(pdgid)])
            for schema in dictionary.tokenization_schema:
                if schema == "pdgid":
                    tokenized_event.append(particle_index + dictionary.PDGID_OFFSET)
                elif schema == "pt":
                    tokenized_event.append(custom_searchsorted(pt, dictionary.pt_bins) + dictionary.PT_OFFSET)
                elif schema == "px":
                    tokenized_event.append(custom_searchsorted(px, dictionary.px_bins) + dictionary.PX_OFFSET)
                elif schema == "py":
                    tokenized_event.append(custom_searchsorted(py, dictionary.py_bins) + dictionary.PY_OFFSET)
                elif schema == "pz":
                    tokenized_event.append(custom_searchsorted(pz, dictionary.pz_bins) + dictionary.PZ_OFFSET)
                elif schema == "energy":
                    tokenized_event.append(custom_searchsorted(energy, dictionary.e_bins) + dictionary.ENERGY_OFFSET)
                elif schema == "eta":
                    tokenized_event.append(custom_searchsorted(eta, dictionary.eta_bins) + dictionary.ETA_OFFSET)
                elif schema == "theta":
                    tokenized_event.append(custom_searchsorted(theta, dictionary.theta_bins) + dictionary.THETA_OFFSET)
                elif schema == "phi":
                    tokenized_event.append(custom_searchsorted(phi, dictionary.phi_bins) + dictionary.PHI_OFFSET)
                elif schema == "particle_start":
                    tokenized_event.append(dictionary.particle_start_token)
                elif schema == "particle_end":
                    tokenized_event.append(dictionary.particle_end_token)
                else:
                    raise RuntimeError(f"pTokenizer: Unknown tokenization schema: {schema}")
        
        tokenized_event.append(dictionary.event_end_token)
        return tokenized_event

    def encode_event(self, tokens: list[float]) -> list[int] | None:
        """
        Tokenize one event, but do NOT pad it.
        """
        return self.tokenize_event(tokens)

    def postprocess_data(self):
        """
        For this tokenizer, no post processing is required since the encoded events are the data.
        The encoded events will be packed anyway so no padding is required to the correct sequence
        length.
        Important: a meta-data file needs to be written so the correct sequence length is used
        """

        # Simply copy the flattened file to the final filepath
        concat_data_filepath = self.in_paths.temp_data_dir / 'concatenated_data.bin'
        shutil.copyfile(concat_data_filepath, self.in_paths.tokenized_data_filepath)


class EventPerSequenceWholeParticleTokenizer(BaseTokenizer):
    
    def __init__(
        self,
        dictionary: Dictionary,
        in_paths: Paths,
        dtype=np.uint16,
        flush_tokens: int = 1024 ** 2,
        clean_temp_dir: bool = False,
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
        
        token = base_offset + eta_bin + (offset_due_to_eta * pt_bin) + (offset_due_to_eta * offset_due_to_pt * phi_bin)
        
        Notably, the PDGID is ignored.
        """
        tokenized_event = [dictionary.event_start_token]
        num_particles = len(event) // NUM_FEATURES_PER_PARTICLE_RAW
        
        for particle_idx in range(num_particles):
            pdgid  = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0]
            energy = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1]
            px     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2]
            py     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3]
            pz     = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4]
            
            r      = math.sqrt(px ** 2 + py ** 2 + pz ** 2)
            pt     = math.sqrt(px ** 2 + py ** 2)
            theta  = math.acos(pz / r) if r != 0 else 0.0
            phi    = math.atan2(py, px)
            eta    = -math.log(math.tan(theta / 2)) if theta != 0 else 0.0
            
            # Eta constrain
            if abs(eta) > 4:
                return []
            
            pt_bin = custom_searchsorted(pt, dictionary.pt_bins)
            eta_bin = custom_searchsorted(eta, dictionary.eta_bins)
            phi_bin = custom_searchsorted(phi, dictionary.phi_bins)
            
            token = dictionary.ETA_OFFSET + eta_bin + (dictionary.ETA_OFFSET * pt_bin) + (dictionary.ETA_OFFSET * dictionary.PT_OFFSET * phi_bin)
            tokenized_event.append(token)
        
        tokenized_event.append(dictionary.event_end_token)
        return tokenized_event

    def pad_sequence(self, event: list[float]) -> list[float]:
        """
        event: tokenzied, unpadded event
        num_particles_max: used to determine padding length to ensure uniform sequence lengths
        
        Pads a single event to the maximum number of particles. Padding sequence used is derived
        from the dictionary's tokenization schema.
        """
        
        if len(event) == 0:
            return None

        len_pad_required = self.sequence_length - len(event)
        if len_pad_required < 0:
            raise RuntimeError(
                f"Tokenized event length {len(event)} exceeds "
                f"sequence_length {self.sequence_length}"
            )

        num_padding_sequences_required = len_pad_required // len(self.padding_sequence)
        padded_event = event + self.padding_sequence * num_padding_sequences_required
        return padded_event

    def encode_event(self, tokens: list[float]) -> list[int]:
        tokenized_event = self.tokenize_event(tokens)
        padded_event = self.pad_sequence(tokenized_event)
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
        shutil.copyfile(concat_data_filepath, self.in_paths.tokenized_data_filepath)

    def write_metadata(self):
        """
        Write sidecar metadata needed to safely reload the raw binary file.
        """
        vocab_size = dictionary.ETA_OFFSET + (len(dictionary.eta_bins) * len(dictionary.pt_bins) * len(dictionary.phi_bins))
        metadata = {
            "format": "whole_particle",
            "dtype": self.dtype.name,
            "vocab_size": vocab_size,
            "sequence_length": self.sequence_length,
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
        input_data_filepath     = script_dir / 'data' / 'raw' / dictionary.dataset_name,
        tokenized_data_filepath = script_dir / 'data' / 'tokenized' / dictionary.tokenization_name / 'tokenized_data.bin',
        temp_data_dir           = script_dir / 'data' / 'tokenized' / dictionary.tokenization_name / 'temp'
    )
    humanized_dictionary_filepath = script_dir / 'data' / 'tokenized' / dictionary.tokenization_name / 'humanized_dictionary.txt'
    
    # dictionary.update_dictionary_particle_list(relevant_paths.input_data_filepath, dictionary_filepath)
    dictionary.output_humanized_dictionary(humanized_dictionary_filepath)
    
    tokenizer = EventPerSequenceWholeParticleTokenizer(
        dictionary=dictionary,
        in_paths=relevant_paths,
        dtype=np.uint16,
        clean_temp_dir=True
    )
    tokenizer.batch_encode_byte_ranges()
    tokenizer.postprocess_data()
    tokenizer.write_metadata()