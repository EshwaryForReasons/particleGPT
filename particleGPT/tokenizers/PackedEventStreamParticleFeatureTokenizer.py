
import numpy as np
import paths as paths
from pathlib import Path
import warnings

from particleGPT.dictionary import Dictionary
from particleGPT.tokenizers.base_tokenizer import analyze_dataset, DecodeEventResult
from particleGPT.tokenizers.EventPerSequenceParticleFeatureTokenizer import EventPerSequenceParticleFeatureTokenizer

NUM_FEATURES_PER_PARTICLE_RAW = 5

class PackedEventStreamParticleFeatureTokenizer(EventPerSequenceParticleFeatureTokenizer):
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
    
    def encode_dataset(self, input_data_filepath: Path):
        # Run some checks before calling super().encode_dataset(...)
        num_events, num_particles_max = analyze_dataset(input_data_filepath)
        print(f"Found {num_events:,} events in input dataset.")
        print(f"Max particles/event: {num_particles_max:,}")
        
        # Handle num_particles_max override
        if self.dictionary.particle_count_override is not None:
            if self.dictionary.particle_count_override < num_particles_max:
                raise ValueError(
                    "The particle_count_override in the dictionary must be >= maximum "
                    "number of particles found in the dataset."
                )
            num_particles_max = self.dictionary.particle_count_override
        
        # Packed tokenzier must be provided a sequence length
        if self.sequence_length is None:
            raise ValueError("Packed tokenzier must be provided a sequence length!")

        return super().encode_dataset(input_data_filepath)

    def encode_event(self, tokens: list[float]) -> DecodeEventResult:
        """
        Tokenize one event, but do NOT pad it.
        """
        return self._tokenize_event(tokens)

    def postprocess_data(self):
        """
        Pad the final sequence such that the data is divisible by sequence_length.
        
        Important: a meta-data file needs to be written so the correct sequence length is used
        """

        concat_data_filepath = self.temp_dir / 'concatenated_data.bin'
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
        
        if new_len % self.sequence_length != 0:
            raise RuntimeError(
                f"Final token count {new_len} is not divisible by sequence_length {self.sequence_length}. "
                f"Some tokens will be dropped when reshaping into sequences."
            )
        self.total_sequences_written = int(new_len // self.sequence_length)