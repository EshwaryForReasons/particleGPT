
import math
import numpy as np
import paths as paths
from dataclasses import dataclass
from pathlib import Path
from numba import njit, types

from particle import PDGID, Particle

from particleGPT.dictionary import Dictionary
from particleGPT.quickmaths import custom_searchsorted
from particleGPT.tokenizers.base_tokenizer import analyze_dataset, DecodeEventResult, BaseTokenizer

NUM_FEATURES_PER_PARTICLE_RAW = 5

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
        
        # Derive sequence length
        sequence_length = num_particles_max * self.dictionary.num_tokens_per_particle + 2
        if self.sequence_length is not None and self.sequence_length != sequence_length:
            raise ValueError(
                "Sequence length provided in config does not match calculated sequence length.\n"
                f"config sequence length: {self.sequence_length}; calculated sequence length: {sequence_length}"
            )
        self.sequence_length = sequence_length
        
        return super().encode_dataset(input_data_filepath)
        
    def _tokenize_event(self, event: list[float]) -> list[int] | None:
        """
        Tokenizes and returns a single event.
        """
        tokenized_event = [self.dictionary.event_start_token]
        if len(event) % NUM_FEATURES_PER_PARTICLE_RAW != 0:
            raise RuntimeError(f"Malformed event: got {len(event)} float, which is not divisible by {NUM_FEATURES_PER_PARTICLE_RAW}")
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

    def _pad_sequence(self, event: list[float]) -> list[float]:
        """
        event: tokenzied, unpadded event
        num_particles_max: used to determine padding length to ensure uniform sequence lengths
        
        Pads a single event to the maximum number of particles. Padding sequence used is derived
        from the dictionary's tokenization schema.
        """

        len_pad_required = self.sequence_length - len(event)
        if len_pad_required < 0:
            raise RuntimeError(f"Tokenized event length {len(event)} exceeds sequence_length {self.sequence_length}")
        if len_pad_required % len(self.cached_padding_sequence) != 0:
            raise RuntimeError(f"Cannot pad exactly: need {len_pad_required} tokens, but padding_sequence has length {len(self.cached_padding_sequence)}.")

        num_padding_sequences_required = len_pad_required // len(self.cached_padding_sequence)
        padded_event = event + self.cached_padding_sequence * num_padding_sequences_required
        return padded_event

    def encode_event(self, event: list[float]) -> list[int] | None:
        tokenized_event = self._tokenize_event(event)
        if tokenized_event is None:
            return None
            
        padded_event = self._pad_sequence(tokenized_event)
        if padded_event is not None and len(padded_event) != self.sequence_length:
            raise RuntimeError(f"Padding failed: got {len(padded_event)}, expected {self.sequence_length}.")
        
        return padded_event
    
    
    @staticmethod
    def _sqrt_nonnegative(x: float, tol: float = 1e-12) -> float:
        if x < 0.0:
            if x > -tol:
                return 0.0
        return np.sqrt(x)
    
    @staticmethod
    def _vector_calc_pt_eta_phi(mass, features):
        pt = features["pt"]
        eta = features["eta"]
        phi = features["phi"]

        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = EventPerSequenceParticleFeatureTokenizer._sqrt_nonnegative(mass * mass + px * px + py * py + pz * pz)
        return energy, px, py, pz

    @staticmethod
    def _vector_calc_e_theta_phi(mass, features):
        energy = features["e"]
        theta = features["theta"]
        phi = features["phi"]
        momentum = EventPerSequenceParticleFeatureTokenizer._sqrt_nonnegative(energy * energy - mass * mass)

        pt = momentum * np.sin(theta)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = momentum * np.cos(theta)
        return energy, px, py, pz

    @staticmethod
    def _vector_calc_e_px_py_pz(mass, features):
        energy = features["e"]
        px = features["px"]
        py = features["py"]
        pz = features["pz"]
        return energy, px, py, pz
    
    
    # @staticmethod
    # # @njit("int64(float64, float64[:])", cache=True, nogil=True)
    # @njit(types.Tuple((types.float64, types.float64, types.float64, types.float64))(
    #     types.float64, types.float64, types.float64, types.float64, types.float64
    # ), cache=True, nogil=True)
    # def _vector_calc_pt_eta_phi(mass, f1, f2, f3, opt_f4):
    #     pt = f1
    #     eta = f2
    #     phi = f3

    #     px = pt * np.cos(phi)
    #     py = pt * np.sin(phi)
    #     pz = pt * np.sinh(eta)
    #     energy = np.sqrt(mass * mass + px * px + py * py + pz * pz)
    #     return energy, px, py, pz

    # @staticmethod
    # def _vector_calc_e_theta_phi(mass, f1, f2, f3, opt_f4):
    #     energy = f1
    #     theta = f2
    #     phi = f3
    #     momentum = np.sqrt(energy * energy - mass * mass)

    #     pt = momentum * np.sin(theta)
    #     px = pt * np.cos(phi)
    #     py = pt * np.sin(phi)
    #     pz = momentum * np.cos(theta)
    #     return energy, px, py, pz

    # @staticmethod
    # def _vector_calc_e_px_py_pz(mass, f1, f2, f3, opt_f4):
    #     energy = opt_f4
    #     px = f1
    #     py = f2
    #     pz = f3
    #     return energy, px, py, pz
    

    def build_caches(self):
        self.cached_pdgid_to_mass = {}
        for idx, pdgid in self.dictionary.pdgids.items():
            if pdgid == 0:
                # 0 is for reserved PDGIDs
                continue
            particle = Particle.from_pdgid(pdgid)
            self.cached_pdgid_to_mass[pdgid] = particle.mass
        
        # Profiling revealed ~3% of decoding runtime is spent doing this. i.e this is worth caching.
        self.cached_tokenization_schema_len = len(self.dictionary.tokenization_schema)
        
        self.cached_feature_converter_func = None
        ts = set(self.dictionary.tokenization_schema)
        if {"pt", "eta", "phi"} <= ts:
            self.cached_feature_converter_func = EventPerSequenceParticleFeatureTokenizer._vector_calc_pt_eta_phi
        elif {"e", "theta", "phi"} <= ts:
            self.cached_feature_converter_func = EventPerSequenceParticleFeatureTokenizer._vector_calc_e_theta_phi
        elif {"e", "px", "py", "pz"} <= ts:
            self.cached_feature_converter_func = EventPerSequenceParticleFeatureTokenizer._vector_calc_e_px_py_pz
        else:
            raise ValueError("Particle setup did not match any known method of constructing a full particle vector.")
        
        # Cache here as an optimization, accessing through a different class can be
        # terrible for performance in python
        self.cached_padding_sequence = self.dictionary.padding_sequence
        if len(self.cached_padding_sequence) == 0:
            raise ValueError("padding_sequence cannot be empty.")
        
        super().build_caches()

    
    def _untokenize_event(self, tokens: list[int]) -> DecodeEventResult:
        """
        Padding should be stripped before invoking this function. This does not handle padding.
        This function will not check for event malformity it will simply fail horrible. Event
        malformity checks should be done before invocation.
        
        @TODO: implement untokenization failure reasons, currently just returns None, which is not
        helpful when iterating models and nagivating failures.
        """
        
        result = DecodeEventResult()
        
        # -2 for EVENT_START and EVENT_END tokens
        num_particles = (len(tokens) - 2) // len(self.dictionary.tokenization_schema)
        
        if tokens[0] != self.dictionary.event_start_token or tokens[-1] != self.dictionary.event_end_token:
            result.success = False
            result.failure_reason = (
                f"Event is not currently delineated with EVENT_START and EVENT_END tokens. Expected start and end tokens to be "
                f"{self.dictionary.event_start_token} and {self.dictionary.event_end_token}, got {tokens[0]} and {tokens[-1]} instead."
            )
            return result
        
        for particle_idx in range(num_particles):
            this_particle_start = particle_idx * self.cached_tokenization_schema_len + 1
            this_particle_end = (particle_idx + 1) * self.cached_tokenization_schema_len + 1
            particle_tokens = tokens[this_particle_start:this_particle_end]
            
            for schema, token in zip(self.dictionary.tokenization_schema, particle_tokens, strict=True):
                if schema == "particle_start":
                    if token != self.dictionary.particle_start_token:
                        result.success = False
                        result.failure_reason = f"Expected {schema} token of {self.dictionary.particle_start_token}, got token {token}."
                        return result
                elif schema == "particle_end":
                    if token != self.dictionary.particle_end_token:
                        result.success = False
                        result.failure_reason = f"Expected {schema} token of {self.dictionary.particle_end_token}, got token {token}."
                        return result
                elif schema == "pdgid":
                    offset = self.dictionary.PDGID_OFFSET
                    # @TODO: maybe make this more robust? Works for now, I guess..
                    pdgid_index = token - offset
                    if pdgid_index < 0 or pdgid_index >= self.dictionary.num_particles:
                        result.success = False
                        result.failure_reason = f"Expected pdgid token in [{offset}, {offset + self.dictionary.num_particles}], got token {token}."
                        return result
                    pdgid = self.dictionary.pdgids[pdgid_index]
                    if pdgid == 0:
                        result.success = False
                        result.failure_reason = f"PDGID token {token} maps to reserved PDG slot {pdgid_index}."
                        return result
                    result.event.append(pdgid)
                else:
                    offset = self.dictionary.feature_offsets[schema]
                    # bin_values = np.asarray(self.dictionary.feature_bins[schema].centers, dtype=np.float64)
                    # @TODO: ensure dictionary uses np.float64 for these
                    bin_values = self.dictionary.feature_bins[schema].centers
                    bin_idx = token - offset
                    if bin_idx < 0 or bin_idx >= len(bin_values):
                        result.success = False
                        result.failure_reason = f"Expected {schema} token in [{offset}, {offset + len(bin_values)}), got token {token}."
                        return result
                    result.event.append(bin_values[bin_idx])
        
        result.success = True
        return result

    def _conv_particles_to_raw_style(self, untokenized_event_res):
        """
        Convert decoded feature values into raw-style pdgid, energy, px, py, pz.
        """
        
        result = DecodeEventResult()
        num_particles = len(untokenized_event_res.event) // self.cached_tokenization_schema_len
        for particle_idx in range(num_particles):
            particle_start_idx = particle_idx * self.cached_tokenization_schema_len
            particle_end_idx = particle_start_idx + self.cached_tokenization_schema_len # more precisely, next particle start index
            particle_tokens = untokenized_event_res.event[particle_start_idx:particle_end_idx]
            # @TODO: this doesn't work if we have a particle_start or end token as part of the particle, fix this issue
            #   easy way would just be to remove them before the features = ... line...
            filtered_schema = [schema for schema in self.dictionary.tokenization_schema if schema != 'particle_start' and schema != 'particle_end']
            features = dict(zip(filtered_schema, particle_tokens, strict=True))
            
            # PDGID is required to have a valid particle, so it is reasonable to assume it will exist in any given schema.
            # untokenize_event already gets the actual PDGID, not just a token
            pdgid = features['pdgid']
            
            mass = self.cached_pdgid_to_mass[pdgid]
            energy, px, py, pz = self.cached_feature_converter_func(mass, features)
            result.event += [int(pdgid), energy, px, py, pz]
        
        result.success = True
        return result
    
    # def _conv_particles_to_raw_style(self, untokenized_event_res):
    #     """
    #     Convert decoded feature values into raw-style pdgid, energy, px, py, pz.
    #     """
        
    #     result = DecodeEventResult()
    #     num_particles = len(untokenized_event_res.event) // self.cached_tokenization_schema_len
    #     for particle_idx in range(num_particles):
    #         particle_start_idx = particle_idx * self.cached_tokenization_schema_len
    #         particle_end_idx = particle_start_idx + self.cached_tokenization_schema_len # more precisely, next particle start index
    #         particle_tokens = untokenized_event_res.event[particle_start_idx:particle_end_idx]
            
    #         # PDGID is required to have a valid particle, so it is reasonable to assume it will exist in any given schema.
    #         # untokenize_event already gets the actual PDGID, not just a token
    #         pdgid, f1, f2, f3 = particle_tokens
    #         mass = self.cached_pdgid_to_mass[pdgid]
    #         energy, px, py, pz = self.cached_feature_converter_func(mass, f1, f2, f3, 0.0)
    #         result.event += [pdgid, energy, px, py, pz]
        
    #     result.success = True
    #     return result

    def decode_event(self, tokens: list[int]) -> DecodeEventResult:
        # @TODO implement some well formity checks here
        
        # Untokenized decodes tokens back to raw values but in the form of tokenization_schema
        result = self._untokenize_event(tokens)
        if result.success == False:
            return result
        
        # Convert to (pdgid, energy, px, py, pz) format
        result = self._conv_particles_to_raw_style(result)
        return result
    