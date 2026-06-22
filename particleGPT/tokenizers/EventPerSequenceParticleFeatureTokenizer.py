
import os
import shutil
import math
import numpy as np
import paths as paths
from numba import njit
from tqdm import tqdm
from dataclasses import dataclass
import warnings

from particleGPT.dictionary import Dictionary
from particleGPT.quickmaths import custom_searchsorted
from particleGPT.tokenizers.tokenizer_paths import TokenizerPaths
from particleGPT.tokenizers.base_tokenizer import analyze_dataset, BaseTokenizer

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = os.cpu_count() # 256--old hardcoded value
IO_BUFFER = 16 * 1024 * 1024  # 16 MB
COPY_BUFFER = 256 * 1024 * 1024  # 256 MB

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
        in_paths: TokenizerPaths,
        dtype=np.uint16,
        flush_tokens: int = 1024 ** 2,
        flush_lens: int = 1024 ** 2,
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
        
        self.padding_sequence = dictionary.padding_sequence
        if len(self.padding_sequence) == 0:
            raise ValueError("padding_sequence cannot be empty.")
        
        # This tokenizer pads to the max possible particles. +2 to handle EVENT_START and EVENT_END tokens.
        sequence_length = num_particles_max * dictionary.num_tokens_per_particle + 2
        super().__init__(dictionary, in_paths, dtype, sequence_length, flush_tokens, flush_lens, clean_temp_dir)
        
    def tokenize_event(self, event: list[float]) -> list[int] | None:
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
                    local_token = custom_searchsorted(pt, self.dictionary.pt_bins.thresholds)
                    global_token = local_token+ self.dictionary.PT_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "px":
                    local_token = custom_searchsorted(px, self.dictionary.px_bins.thresholds)
                    global_token = local_token+ self.dictionary.PX_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "py":
                    local_token = custom_searchsorted(py, self.dictionary.py_bins.thresholds)
                    global_token = local_token+ self.dictionary.PY_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "pz":
                    local_token = custom_searchsorted(pz, self.dictionary.pz_bins.thresholds)
                    global_token = local_token+ self.dictionary.PZ_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "e":
                    local_token = custom_searchsorted(energy, self.dictionary.e_bins.thresholds)
                    global_token = local_token+ self.dictionary.ENERGY_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "eta":
                    local_token = custom_searchsorted(eta, self.dictionary.eta_bins.thresholds)
                    global_token = local_token+ self.dictionary.ETA_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "theta":
                    local_token = custom_searchsorted(theta, self.dictionary.theta_bins.thresholds)
                    global_token = local_token+ self.dictionary.THETA_OFFSET
                    tokenized_event.append(global_token)
                elif schema == "phi":
                    local_token = custom_searchsorted(phi, self.dictionary.phi_bins.thresholds)
                    global_token = local_token+ self.dictionary.PHI_OFFSET
                    tokenized_event.append(global_token)
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

    def encode_event(self, tokens: list[float]) -> list[int] | None:
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
