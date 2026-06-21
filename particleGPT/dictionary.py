# Utility for retrieving bins, tokens, vocabulary, and humanizing the dictionary

import json
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from scipy.stats import norm
from scipy.interpolate import interp1d
from pydantic import validate_call

from particle import Particle

import paths

class ETokenTypes(Enum):
    PADDING = 0
    SPECIAL = 1
    PDGID = 2
    MATERIAL = 3
    ENERGY = 4
    ETA = 5
    THETA = 6
    PHI = 7
    PT = 8
    PX = 9
    PY = 10
    PZ = 11
    
FEATURE_TYPES = [
    ("e", "Energy", ETokenTypes.ENERGY),
    ("eta", "Eta", ETokenTypes.ETA),
    ("theta", "Theta", ETokenTypes.THETA),
    ("phi", "Phi", ETokenTypes.PHI),
    ("pt", "Pt", ETokenTypes.PT),
    ("px", "Px", ETokenTypes.PX),
    ("py", "Py", ETokenTypes.PY),
    ("pz", "Pz", ETokenTypes.PZ),
]

# Detokenization of gaussian bins using bin medians
def detokenize_gaussian_bins(tokens, bin_edges):
    """
    Decode tokenized values back to approximate original values
    using bin centers.

    Parameters:
        tokens (np.ndarray): Array of token indices (0-based).
        bin_edges (np.ndarray): Bin edges used during digitization.

    Returns:
        np.ndarray: Decoded values.
    """
    # Bin centers: len = len(bin_edges) + 1 - 1 = len(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Clip tokens to valid range
    tokens = np.clip(tokens, 0, len(bin_centers) - 1)
    return bin_centers[tokens]

# Helper for detokenization of gaussian bins using bin means
def build_gaussian_bin_means(values, bin_edges):
    bin_indices = np.digitize(values, bin_edges[1:-1], right=True)  # tokens range from 0 to len(bin_edges) - 2
    num_bins = len(bin_edges) - 1  # Number of actual bins
    bin_means = np.zeros(num_bins)
    for i in range(num_bins):
        in_bin = values[bin_indices == i]
        if len(in_bin) > 0:
            bin_means[i] = in_bin.mean()
        else:
            bin_means[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
    return bin_means


def linear_edges(start: float, stop: float, n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError(f"start and stop must be finite, got start={start}, stop={stop}")
    if stop <= start:
        raise ValueError(f"stop must be greater than start, got start={start}, stop={stop}")

    return np.linspace(start, stop, n_bins + 1, dtype=np.float64)

def step_edges(start: float, stop: float, step_size: float) -> np.ndarray:
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError(f"start and stop must be finite, got start={start}, stop={stop}")
    if stop <= start:
        raise ValueError(f"stop must be greater than start, got start={start}, stop={stop}")

    # Number of actual bins needed to cover [start, stop].
    n_bins = int(np.ceil((stop - start) / step_size))

    edges = start + np.arange(n_bins + 1, dtype=np.float64) * step_size

    # Force exact upper endpoint. This gives a shorter final bin if the range
    # is not exactly divisible by step_size, which is usually what you want.
    edges[-1] = stop

    return edges

def gaussian_edges(start: float, stop: float, n_bins: int, center: float, sigma: float = 1.0) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not np.isfinite(start) or not np.isfinite(stop) or not np.isfinite(center):
        raise ValueError(
            f"start, stop, and center must be finite, got "
            f"start={start}, stop={stop}, center={center}"
        )
    if stop <= start:
        raise ValueError(f"stop must be greater than start, got start={start}, stop={stop}")

    x_fine = np.linspace(start, stop, 10_000)
    pdf = norm.pdf(x_fine, loc=center, scale=sigma)
    cdf = np.cumsum(pdf)

    denominator = cdf.max() - cdf.min()
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError(
            f"Invalid Gaussian binning for start={start}, stop={stop}, "
            f"center={center}, sigma={sigma}"
        )

    cdf = (cdf - cdf.min()) / denominator

    inv_cdf = interp1d(
        cdf,
        x_fine,
        bounds_error=False,
        fill_value=(start, stop),
        assume_sorted=False,
    )

    uniform_probs = np.linspace(0, 1, n_bins + 1)
    edges = np.asarray(inv_cdf(uniform_probs), dtype=np.float64)

    # Enforce exact endpoints.
    edges[0] = start
    edges[-1] = stop

    # It is possible for aggressive Gaussian settings to create duplicate edges
    # due to numerical precision. FeatureBins will catch that.
    return edges


@dataclass(frozen=True)
class FeatureBins:
    name: str
    edges: np.ndarray

    def __post_init__(self):
        """
        Ensures that the edges are valid and properly formatted. Makes sure edges is truly immutable.
        the frozen=True flag only ensure it can't be reassigned, not that the contents are immutable.
        """
        edges = np.asarray(self.edges, dtype=np.float64, copy=True)

        if edges.ndim != 1:
            raise ValueError(f"{self.name}: edges must be 1D, got shape {edges.shape}")

        if len(edges) == 0:
            edges.setflags(write=False)
            # This supports optional features that are absent from the dictionary.
            
            empty = np.array([], dtype=np.float64)
            empty.setflags(write=False)

            object.__setattr__(self, "edges", edges)
            object.__setattr__(self, "_centers", empty)
            object.__setattr__(self, "_thresholds", empty)
            return

        if len(edges) < 2:
            raise ValueError(f"{self.name}: edges must contain at least two values")
        if not np.all(np.isfinite(edges)):
            raise ValueError(f"{self.name}: edges must all be finite")
        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"{self.name}: edges must be strictly increasing: {edges}")

        centers = 0.5 * (edges[:-1] + edges[1:])
        thresholds = edges[1:-1].copy()

        edges.setflags(write=False)
        centers.setflags(write=False)
        thresholds.setflags(write=False)
        
        # Cache these so they are not recreated every time they are accessed
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "_centers", centers)
        object.__setattr__(self, "_thresholds", thresholds)
        
    @property
    def centers(self) -> np.ndarray:
        return self._centers
    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds
    @property
    def exists(self) -> bool:
        return len(self.edges) > 0
    @property
    def num_tokens(self) -> int:
        if not self.exists:
            return 0
        return len(self.edges) - 1
    
    def tokenize_values(self, values: np.ndarray) -> np.ndarray:
        if not self.exists:
            raise ValueError(f"{self.name}: cannot tokenize values because feature has no bins")
    
        values = np.asarray(values, dtype=np.float64)
        invalid = (
            ~np.isfinite(values)
            | (values < self.edges[0])
            | (values > self.edges[-1])
        )

        if np.any(invalid):
            bad_values = values[invalid][:10]
            raise ValueError(f"{self.name}: invalid values for tokenization: {bad_values}")

        return np.searchsorted(self._thresholds, values, side="right").astype(np.int64)

    def contains_global_token(self, token: int, offset: int) -> bool:
        return offset <= token < offset + self.num_tokens

    def global_to_local(self, token: int, offset: int) -> int:
        local_token = int(token) - offset
        if local_token < 0 or local_token >= self.num_tokens:
            raise ValueError(
                f"{self.name}: token {token} is outside token range "
                f"[{offset}, {offset + self.num_tokens})"
            )
        return local_token

    def local_to_global(self, local_token: int, offset: int) -> int:
        local_token = int(local_token)
        if local_token < 0 or local_token >= self.num_tokens:
            raise ValueError(
                f"{self.name}: local token {local_token} is outside range "
                f"[0, {self.num_tokens})"
            )
        return offset + local_token

    def tokenize_value(self, value: float) -> int:
        return int(self.tokenize_values(np.array([value]))[0])

    def detokenize_local(self, local_token: int) -> float:
        local_token = int(local_token)

        if local_token < 0 or local_token >= self.num_tokens:
            raise ValueError(
                f"{self.name}: local token {local_token} is outside range "
                f"[0, {self.num_tokens})"
            )

        return float(self.centers[local_token])

    def detokenize_global(self, token: int, offset: int) -> float:
        return self.detokenize_local(self.global_to_local(token, offset))


class Dictionary():
    
    def __init__(self, dictionary_filename):
        self.dictionary_filename = Path(dictionary_filename).resolve()
        with open(self.dictionary_filename, 'r') as f:
            self.dictionary_data = json.load(f)
            
        # ===== Validate required values exist and point to good data =====
        
        self.dataset_name = self.dictionary_data.get('dataset', None)
        if self.dataset_name is None:
            raise RuntimeError("dataset_name must be defined in the dictionary.")
        
        self.dataset_filepath = paths.PROJECT_DIR / 'data' / 'raw' / self.dictionary_data['dataset']
        if not self.dataset_filepath.exists():
            raise RuntimeError(f"Dataset file does not exist at expected path: {self.dataset_filepath}")
            
        self.tokenization_name = self.dictionary_data.get('tokenization_name', None)
        if self.tokenization_name is None:
            raise RuntimeError("tokenization_name must be defined in the dictionary.")
        
        self.tokenizer_class_str = self.dictionary_data.get('tokenizer_class', None)
        if self.tokenizer_class_str is None:
            raise ValueError("tokenizer_class must be defined in the dictionary.")
        
        # ===== Load and validate data =====
        
        # Special tokens
        self.special_tokens = self._load_contiguous_int_values(self.dictionary_data["special_tokens"], "special_tokens")
        self.num_special_tokens = len(self.special_tokens)
        # Materials
        self.materials_named = {}
        if 'materials_named' in self.dictionary_data:
            self.materials_named = self._load_contiguous_int_values(self.dictionary_data["materials_named"], "materials_named")
        self.num_materials = len(self.materials_named)
        # tokenization schema
        tokenization_by_position = self._load_contiguous_int_keys(self.dictionary_data["tokenization"], "tokenization", value_type=str)
        self.tokenization_schema = [tokenization_by_position[idx] for idx in range(len(tokenization_by_position))]
        self.num_tokens_per_particle = len(self.tokenization_schema)
        # padding schema
        padding_by_position = self._load_contiguous_int_keys(self.dictionary_data["padding"], "padding", value_type=str)
        self.padding_sequence_names = [padding_by_position[idx] for idx in range(len(padding_by_position))]
        self._validate_padding_sequence()
        # overrides
        self.particle_count_override = self.dictionary_data.get('particle_count_override', None)
        self.num_train_events_override = self.dictionary_data.get('num_train_events_override', None)
        self.num_val_events_override = self.dictionary_data.get('num_val_events_override', None)
        self.num_test_events_override = self.dictionary_data.get('num_test_events_override', None)
        
        if len({x is None for x in (
            self.num_train_events_override,
            self.num_val_events_override,
            self.num_test_events_override,
        )}) != 1:
            raise ValueError("Event count overrides must be either all set or all None (unset).")
        
        # PDGIDs
        self.pdgids = self._load_contiguous_int_keys(self.dictionary_data["pdgids"], "pdgids", value_type=int)
        self._pdgids_to_index = {
            pdgid: slot
            for slot, pdgid in self.pdgids.items()
            if pdgid != 0
        }
        self.num_particles = len(self.pdgids)

        # ===== Create bins =====
        
        self.feature_bins = {}
        for feature_name, _, _ in FEATURE_TYPES:
            self.feature_bins[feature_name] = self._create_feature_bins(feature_name)

        self.e_bins = self.feature_bins["e"]
        self.eta_bins = self.feature_bins["eta"]
        self.theta_bins = self.feature_bins["theta"]
        self.phi_bins = self.feature_bins["phi"]
        self.pt_bins = self.feature_bins["pt"]
        self.px_bins = self.feature_bins["px"]
        self.py_bins = self.feature_bins["py"]
        self.pz_bins = self.feature_bins["pz"]
        
        # ===== Calculate offsets =====
        
        self.SPECIAL_TOKENS_OFFSET = 0
        self.PDGID_OFFSET = self.SPECIAL_TOKENS_OFFSET + self.num_special_tokens
        self.MATERIAL_OFFSET = self.PDGID_OFFSET + self.num_particles

        current_offset = self.MATERIAL_OFFSET + self.num_materials

        self.feature_offsets = {}
        for feature_name, _, _ in FEATURE_TYPES:
            self.feature_offsets[feature_name] = current_offset
            current_offset += self.feature_bins[feature_name].num_tokens

        self.vocab_size = current_offset
        self._validate_token_ranges()
        
        self.table_data = [
            ["Type", "Num", "Token Range", "Min", "Max", "Step Size"],
            [
                "Special tokens",
                self.num_special_tokens,
                self.token_range_str(self.SPECIAL_TOKENS_OFFSET, self.num_special_tokens),
                "N/A",
                "N/A",
                "N/A",
            ],
            [
                "Particles",
                self.num_particles,
                self.token_range_str(self.PDGID_OFFSET, self.num_particles),
                "N/A",
                "N/A",
                "N/A",
            ],
            [
                "Materials",
                self.num_materials,
                self.token_range_str(self.MATERIAL_OFFSET, self.num_materials),
                "N/A",
                "N/A",
                "N/A",
            ],
        ]

        for feature_name, display_name, _ in FEATURE_TYPES:
            feature_bins = self.feature_bins[feature_name]
            offset = self.feature_offsets[feature_name]

            self.table_data.append([
                f"{display_name} bins",
                feature_bins.num_tokens,
                self.token_range_str(offset, feature_bins.num_tokens),
                self.token_min(feature_name),
                self.token_max(feature_name),
                self.token_step_size(feature_name),
            ])

    @staticmethod
    @validate_call
    def _load_contiguous_int_values(raw: dict, name: str) -> dict[str, int]:
        """
        Load mappings where keys are names and values are local token IDs.
        Ensures values must be exactly 0..N-1, unique, integer, and contiguous.
        """
        parsed = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValueError(f"{name}: key {key!r} must be a string")
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}: value for {key!r} must be an integer, got {value!r}")
            parsed[key] = value

        if not parsed:
            raise ValueError(f"{name}: must contain at least one entry")

        values = list(parsed.values())
        expected = set(range(len(values)))
        actual = set(values)

        if len(values) != len(actual):
            raise ValueError(f"{name}: values must be unique, got {values}")
        if actual != expected:
            raise ValueError(
                f"{name}: values must be contiguous 0..N-1. "
                f"Expected={sorted(expected)}, got={sorted(actual)}"
            )

        return parsed

    @staticmethod
    @validate_call
    def _load_contiguous_int_keys(raw: dict, name: str, value_type=None) -> dict[int, object]:
        """
        Load mappings where keys are ordered integer positions/slots.
        Ensures keys must normalize to exactly 0..N-1, unique, integer, and contiguous.
        """
        parsed = {}
        for key, value in raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: key {key!r} must be an integer string") from exc

            if idx < 0:
                raise ValueError(f"{name}: key {key!r} must be non-negative")

            if idx in parsed:
                raise ValueError(
                    f"{name}: duplicate normalized key {idx}; "
                    f"key {key!r} collides with an earlier key"
                )

            if value_type is not None:
                if value_type is int:
                    if isinstance(value, bool):
                        raise ValueError(f"{name}: value at key {key!r} must be an integer, got bool {value}")
                    
                    try:
                        value = int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{name}: value at key {key!r} must be an integer, got {value!r}") from exc
                elif value_type is str:
                    if not isinstance(value, str):
                        raise ValueError(f"{name}: value at key {key!r} must be a string, got {value!r}")
                else:
                    try:
                        value = value_type(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{name}: value at key {key!r} could not be converted to {value_type}: {value!r}") from exc

            parsed[idx] = value

        if not parsed:
            raise ValueError(f"{name}: must contain at least one entry")

        expected = set(range(max(parsed) + 1))
        actual = set(parsed)

        if actual != expected:
            raise ValueError(
                f"{name}: keys must be contiguous 0..N-1. "
                f"Missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )

        return {idx: parsed[idx] for idx in range(len(parsed))}

    def _validate_token_ranges(self) -> None:
        """
        Validate that all top-level token ranges are contiguous, non-overlapping,
        and exactly cover [0, vocab_size).
        """
        ranges = [
            ("special_tokens", self.SPECIAL_TOKENS_OFFSET, self.num_special_tokens),
            ("pdgids", self.PDGID_OFFSET, self.num_particles),
            ("materials", self.MATERIAL_OFFSET, self.num_materials),
        ]

        for feature_name, _, _ in FEATURE_TYPES:
            ranges.append((
                feature_name,
                self.feature_offsets[feature_name],
                self.feature_bins[feature_name].num_tokens,
            ))

        expected_start = 0
        for name, start, count in ranges:
            if count < 0:
                raise RuntimeError(f"{name}: token count cannot be negative, got {count}")
            if start != expected_start:
                raise RuntimeError(f"Token range gap/overlap before {name}: expected start {expected_start}, got {start}")
            expected_start = start + count

        if self.vocab_size != expected_start:
            raise RuntimeError(f"Vocabulary size mismatch: got {self.vocab_size}, expected {expected_start}")
    
    def _validate_padding_sequence(self) -> None:
        valid_padding_names = set(self.special_tokens)
        for idx, padding_name in enumerate(self.padding_sequence_names):
            if padding_name not in valid_padding_names:
                raise ValueError(
                    f"padding: unknown token name at position {idx}: {padding_name!r}. "
                    f"Expected one of {sorted(valid_padding_names)}"
                )
                
    # =====================
    # Binning functions
    # =====================
    
    def _create_linear_edges(self, type_str: str, token_bin_key_name: str) -> np.ndarray:
        config = self.dictionary_data[token_bin_key_name]

        start = self.token_min(type_str)
        stop = self.token_max(type_str)

        if "step_size" in config:
            step_size = config["step_size"]
            return step_edges(start, stop, step_size)

        if "n_bins" in config:
            n_bins = config["n_bins"]
            return linear_edges(start, stop, n_bins)

        raise ValueError(
            f"Linear tokenization for {type_str} requires either step_size or n_bins"
        )

    def _create_gaussian_edges(self, type_str: str, token_bin_key_name: str) -> np.ndarray:
        config = self.dictionary_data[token_bin_key_name]

        gaussian_center = config.get("gaussian_center")
        gaussian_sigma = config.get("gaussian_sigma", 1.0)
        n_gaussian_bins = config.get("n_bins")

        if gaussian_center is None:
            raise ValueError(f"Missing Gaussian center for {type_str}")
        if n_gaussian_bins is None or n_gaussian_bins <= 0:
            raise ValueError(f"Missing or invalid number of Gaussian bins for {type_str}: {n_gaussian_bins}")

        return gaussian_edges(
            self.token_min(type_str),
            self.token_max(type_str),
            n_gaussian_bins,
            gaussian_center,
            gaussian_sigma,
        )
    
    def _create_feature_bins(self, type_str):
        token_bin_key_name = f"{type_str}_bin_data"

        if token_bin_key_name not in self.dictionary_data:
            return FeatureBins(name=type_str, edges=np.array([], dtype=np.float64))

        tokenization_config = self.dictionary_data[token_bin_key_name]
        tokenization_function = tokenization_config.get("tokenization", "linear")

        if tokenization_function == "linear":
            edges = self._create_linear_edges(type_str, token_bin_key_name)
        elif tokenization_function == "gaussian":
            edges = self._create_gaussian_edges(type_str, token_bin_key_name)
        else:
            raise ValueError(
                f"Unsupported tokenization function for {type_str}: {tokenization_function!r}"
            )

        return FeatureBins(name=type_str, edges=edges)
    
    # Functions to make the table look nicer
    def token_min(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data']['min']
    def token_max(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data']['max']
    def token_step_size(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data'].get('step_size', None)
    def token_n_bins(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.dictionary_data[f'{type_str}_bin_data']['n_bins']
    def token_range(self, type_str):
        if f'{type_str}_bin_data' not in self.dictionary_data:
            return 0
        return self.token_max(type_str) - self.token_min(type_str)
    def token_range_str(self, offset, num_tokens):
        if num_tokens == 0:
            return "N/A"
        return f"{offset} - {offset + num_tokens - 1}"
    
    # =====================
    # Special token retrieval as properties
    # =====================
    
    @property
    def padding_token(self):
        return self.special_tokens["padding"]
    @property
    def event_start_token(self):
        return self.special_tokens["event_start"]
    @property
    def event_end_token(self):
        return self.special_tokens["event_end"]
    @property
    def particle_start_token(self):
        return self.special_tokens["particle_start"]
    @property
    def particle_end_token(self):
        return self.special_tokens["particle_end"]
    
    # Offset properties; these functions are here for legacy reasons.
    # I wonder if it is better to keep them? in principle forcing the use of hard-coded
    # function calls might be more robust, but allowing the map to be accessed directly
    # will be more flexible.
    # In principle, we should only allow one or the other, no mixing.
    # @TODO: Personally, I am leaning towards removing these and just accessing feature_offsets directly.
    @property
    def ENERGY_OFFSET(self):
        return self.feature_offsets["e"]
    @property
    def ETA_OFFSET(self):
        return self.feature_offsets["eta"]
    @property
    def THETA_OFFSET(self):
        return self.feature_offsets["theta"]
    @property
    def PHI_OFFSET(self):
        return self.feature_offsets["phi"]
    @property
    def PT_OFFSET(self):
        return self.feature_offsets["pt"]
    @property
    def PX_OFFSET(self):
        return self.feature_offsets["px"]
    @property
    def PY_OFFSET(self):
        return self.feature_offsets["py"]
    @property
    def PZ_OFFSET(self):
        return self.feature_offsets["pz"]
    
    @property
    def pdgids_to_index(self) -> dict[int, int]:
        return self._pdgids_to_index
    
    @property
    def padding_sequence(self) -> list[int]:
        return [self.special_tokens[token_name] for token_name in self.padding_sequence_names]
    
    @validate_call
    def get_token_type(self, token: int) -> ETokenTypes:
        if token == self.padding_token:
            return ETokenTypes.PADDING
        if token in self.special_tokens.values():
            return ETokenTypes.SPECIAL
        if self.PDGID_OFFSET <= token < self.PDGID_OFFSET + self.num_particles:
            return ETokenTypes.PDGID
        if self.MATERIAL_OFFSET <= token < self.MATERIAL_OFFSET + self.num_materials:
            return ETokenTypes.MATERIAL

        for feature_name, _, token_type in FEATURE_TYPES:
            feature_bins = self.feature_bins[feature_name]
            offset = self.feature_offsets[feature_name]
            if feature_bins.contains_global_token(token, offset):
                return token_type

        raise ValueError(f"Token {token} is outside vocabulary range [0, {self.vocab_size})")

    # =====================
    # Markdown output functions
    # =====================
    
    @staticmethod
    def _markdown_table(headers, rows):
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")

        return "\n".join(lines)

    @staticmethod
    @validate_call
    def _particle_name(pdgid: int):
        if pdgid == 0:
            return "reserved"
        # Don't try/except; if an unknown particle is encountered, that is a issue that should be debugged
        return Particle.from_pdgid(pdgid).name
    
    @validate_call
    def output_humanized_dictionary(self, output_file_path: Path):
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        active_particles = []
        reserved_particles = []

        for idx_str, pdgid in self.pdgids.items():
            idx = int(idx_str)
            pdgid = int(pdgid)
            token = self.PDGID_OFFSET + idx
            row = [idx, token, pdgid, self._particle_name(pdgid)]

            if pdgid == 0:
                reserved_particles.append(row)
            else:
                active_particles.append(row)

        special_rows = [
            [token_name, token_value + self.SPECIAL_TOKENS_OFFSET]
            for token_name, token_value in self.special_tokens.items()
        ]
        material_rows = [
            [material_name, token_value + self.MATERIAL_OFFSET]
            for material_name, token_value in self.materials_named.items()
        ]
        
        feature_rows = []
        for feature_name, display_name, _ in FEATURE_TYPES:
            feature_bins = self.feature_bins[feature_name]
            offset = self.feature_offsets[feature_name]

            feature_rows.append([
                f"{display_name} bins",
                feature_bins.num_tokens,
                self.token_range_str(offset, feature_bins.num_tokens),
                self.token_min(feature_name),
                self.token_max(feature_name),
                self.token_step_size(feature_name),
            ])

        token_range_rows = [
            [
                "Special tokens",
                self.num_special_tokens,
                self.token_range_str(self.SPECIAL_TOKENS_OFFSET, self.num_special_tokens),
                "N/A",
                "N/A",
                "N/A",
            ],
            [
                "Particles",
                self.num_particles,
                self.token_range_str(self.PDGID_OFFSET, self.num_particles),
                "N/A",
                "N/A",
                "N/A",
            ],
            [
                "Materials",
                self.num_materials,
                self.token_range_str(self.MATERIAL_OFFSET, self.num_materials),
                "N/A",
                "N/A",
                "N/A",
            ],
            *feature_rows,
        ]

        warnings = []

        for feature_name, display_name, _ in FEATURE_TYPES:
            if self.feature_bins[feature_name].num_tokens == 0:
                warnings.append(f"- No {display_name} bins configured.")

        if reserved_particles:
            warnings.append(f"- {len(reserved_particles)} particle slots remain reserved.")

        lines = [
            "# Dictionary Report",
            "",
            "## Summary",
            "",
            f"- Dataset: `{self.dataset_name}`",
            f"- Dataset path: `{self.dataset_filepath}`",
            f"- Tokenization name: `{self.tokenization_name}`",
            f"- Tokenizer class: `{self.tokenizer_class_str}`",
            f"- Vocabulary size: `{self.vocab_size}`",
            f"- Tokens per particle: `{self.num_tokens_per_particle}`",
            f"- Particle slots used: `{len(active_particles)} / {self.num_particles}`",
            f"- Particle slots reserved: `{len(reserved_particles)} / {self.num_particles}`",
            "",
            "## Token ranges",
            "",
            self._markdown_table(
                ["Type", "Count", "Token range", "Min", "Max", "Step size"],
                token_range_rows,
            ),
            "",
            "## Special tokens",
            "",
            self._markdown_table(
                ["Token name", "Token value"],
                special_rows,
            ),
            "",
            "## Materials",
            "",
            self._markdown_table(
                ["Material", "Token value"],
                material_rows,
            ),
            "",
            "## Particles",
            "",
            f"Used particle slots: `{len(active_particles)} / {self.num_particles}`  ",
            f"Reserved particle slots: `{len(reserved_particles)} / {self.num_particles}`",
            "",
            "### Active particle slots",
            "",
            self._markdown_table(
                ["Slot", "Token", "PDGID", "Particle name"],
                active_particles,
            ),
            "",
            "### Reserved particle slots",
            "",
            self._markdown_table(
                ["Slot", "Token", "PDGID", "Meaning"],
                [
                    [slot, token, pdgid, "reserved"]
                    for slot, token, pdgid, _ in reserved_particles
                ],
            ),
            "",
        ]

        if warnings:
            lines.extend([
                "## Warnings",
                "",
                *warnings,
                "",
            ])

        output_file_path.write_text("\n".join(lines))

    # =====================
    # Update particle list using dataset
    # =====================

    @validate_call
    def update_dictionary_particle_list(self, input_data_filepath: Path):
        """
        Extend dictionary['pdgids'] using first appearance order from input data.

        Existing nonzero PDGID slots are preserved exactly. New PDGIDs are assigned
        to currently unused slots, where unused means value == 0. This allows larger
        datasets to extend dictionaries from smaller datasets without changing the
        meaning of existing particle tokens.
        """
        with self.dictionary_filename.open("r") as f:
            dictionary = json.load(f)

        existing_pdgids = dictionary.get("pdgids")
        if not isinstance(existing_pdgids, dict):
            raise ValueError("dictionary['pdgids'] must be a dictionary")

        # Normalize all keys and values to integers
        normalized_pdgids = self._load_contiguous_int_keys(existing_pdgids, "dictionary['pdgids']", value_type=int)
        max_slots = len(normalized_pdgids)

        existing_nonzero = {
            pdgid
            for pdgid in normalized_pdgids.values()
            if pdgid != 0
        }

        free_slots = [
            idx
            for idx in range(max_slots)
            if normalized_pdgids[idx] == 0
        ]

        observed_new_pdgids = []
        observed_seen = set()

        with open(input_data_filepath, "r") as f:
            for line_number, event in enumerate(f, start=1):
                for particle_number, particle in enumerate(event.split(";"), start=1):
                    fields = particle.split()

                    # Handles blank lines and trailing semicolons.
                    if not fields:
                        continue

                    try:
                        pdgid = int(fields[0])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid PDGID on line {line_number}, "
                            f"particle {particle_number}: {fields[0]!r}"
                        ) from exc

                    if pdgid == 0:
                        raise ValueError(
                            f"Found PDGID 0 in input data on line {line_number}, "
                            f"particle {particle_number}. PDGID 0 is reserved for unused slots."
                        )

                    if pdgid in existing_nonzero:
                        continue

                    if pdgid in observed_seen:
                        continue

                    observed_seen.add(pdgid)
                    observed_new_pdgids.append(pdgid)

        if len(observed_new_pdgids) > len(free_slots):
            raise ValueError(
                f"Dictionary has {len(free_slots)} free PDGID slots, but input data contains "
                f"{len(observed_new_pdgids)} new PDGIDs. "
                f"New PDGIDs in first-appearance order: {observed_new_pdgids}"
            )

        for slot, pdgid in zip(free_slots, observed_new_pdgids, strict=True):
            normalized_pdgids[slot] = pdgid

        # Update on-file dictionary; conv int keys to str keys first
        dictionary["pdgids"] = {
            str(idx): normalized_pdgids[idx]
            for idx in range(max_slots)
        }
        with self.dictionary_filename.open("w") as f:
            json.dump(dictionary, f, indent=2)
        
        # Reinitialize dictionary since data might have been updated
        self.__init__(self.dictionary_filename)