
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
from particleGPT.tokenizers.base_tokenizer import analyze_dataset, BaseTokenizer

NUM_FEATURES_PER_PARTICLE_RAW = 5
N_WORKERS = os.cpu_count() # 256--old hardcoded value
IO_BUFFER = 16 * 1024 * 1024  # 16 MB
COPY_BUFFER = 256 * 1024 * 1024  # 256 MB

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
        # This tokenizer pads to the max possible particles. +2 to handle EVENT_START and EVENT_END tokens.
        sequence_length = num_particles_max * dictionary.num_tokens_per_particle + 2
        super().__init__(dictionary, in_paths, dtype, sequence_length, flush_tokens, flush_lens, clean_temp_dir)
        
    def tokenize_event(self, event: list[float]) -> list[int] | None:
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
            
            pt_bin = custom_searchsorted(pt, self.dictionary.pt_bins.thresholds)
            eta_bin = custom_searchsorted(eta, self.dictionary.eta_bins.thresholds)
            phi_bin = custom_searchsorted(phi, self.dictionary.phi_bins.thresholds)
            
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
