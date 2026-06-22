
import os
import shutil
import math
import numpy as np
import paths as paths
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import warnings

from particleGPT.dictionary import Dictionary
from particleGPT.quickmaths import custom_searchsorted
from particleGPT.tokenizers.tokenizer_paths import TokenizerPaths
from particleGPT.tokenizers.base_tokenizer import BaseTokenizer

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = os.cpu_count() # 256--old hardcoded value
IO_BUFFER = 16 * 1024 * 1024  # 16 MB
COPY_BUFFER = 256 * 1024 * 1024  # 256 MB

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
