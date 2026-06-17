"""
particleGPT untokenizer entrypoint.

This script untokenizes generated token samples using the current particleGPT
pipeline:
  - one flat tokenized binary dataset
  - one tokenized metadata JSON file
  - one preparation JSON file that selects train/validation/test ranges
  - generated sample directories created by sample.py

Default input:
    PROJECT_DIR/generated_samples/<model_name>/sampling_<sample_idx>/samples.csv

Default output:
    PROJECT_DIR/generated_samples/<model_name>/sampling_<sample_idx>/untokenized_samples.csv

Default invalid-token debug output:
    PROJECT_DIR/generated_samples/<model_name>/sampling_<sample_idx>/invalid_token_events.json

The generated samples file is expected to contain one generated sample per row.
Each row must be whitespace-separated token ids. Comma-separated samples are not
supported because sample.py writes whitespace-separated rows. Generated rows with
invalid tokens are dropped from the untokenized CSV and recorded in the debug JSON.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

import pUtil
import paths as paths
import particleGPT.configurator as conf
from particleGPT.dictionary import Dictionary
from train import DataloaderSplitConfig, ESplitTypes, TokenizedMetadataConfig

PROJECT_DIR = paths.PROJECT_DIR
DEFAULT_INPUT_CSV_NAME = "samples.csv"
DEFAULT_OUTPUT_CSV_NAME = "untokenized_samples.csv"
DEFAULT_METADATA_NAME = "untokenizing_metadata.json"
DEFAULT_INVALID_TOKENS_NAME = "invalid_token_events.json"

class BaseUntokenizer():
    """
    Base class for converting generated token ids back into raw-style particles.

    The output format matches the raw particle layout used by tokenizer.py:
        pdgid energy px py pz;pdgid energy px py pz;...

    Energy is reconstructed as a massless approximation when the tokenization
    schema does not include an explicit energy token. This is unavoidable for
    schemas like whole-particle eta/pt/phi encoding because the original energy
    was not retained in the token stream.
    """
    
    def __init__(
        self,
        dictionary: Dictionary,
        input_samples_filepath: Path,
        output_samples_filepath: Path,
        output_metadata_filepath: Path,
        output_invalid_tokens_filepath: Path,
        tokenized_metadata: dict[str, Any]
    ):
        self.dictionary = dictionary
        self.input_samples_filepath = input_samples_filepath
        self.output_samples_filepath = output_samples_filepath
        self.output_metadata_filepath = output_metadata_filepath
        self.output_invalid_tokens_filepath = output_invalid_tokens_filepath
        self.tokenized_metadata = tokenized_metadata
        self.tokenization_format = str(tokenized_metadata.get("format", "base_tokenizer"))
        self.tokenization_schema = list(tokenized_metadata.get("tokenization_schema", getattr(dictionary, "tokenization_schema", [])))
        self.default_pdgid = 0
        self.stop_at_event_end = conf.sampling.stop_at_event_end
        self.stop_at_padding = conf.sampling.stop_at_padding
        self.float_precision = conf.sampling.float_precision
        self.total_samples_read = 0
        self.total_samples_written = 0
        self.total_empty_samples = 0
        self.total_invalid_tokens = 0
        self.total_invalid_samples = 0
        
        if self.float_precision < 0:
            raise ValueError(f"float_precision must be non-negative, got {self.float_precision}.")
        if len(self.tokenization_schema) == 0 and self.tokenization_format != "whole_particle":
            raise ValueError(
                "Could not determine tokenization_schema from tokenized metadata or dictionary. "
                "Make sure the tokenized metadata was written by the current tokenizer."
            )
        
        self.index_to_pdgid = {}
        pdgids_to_index = getattr(dictionary, "pdgids_to_index", None)
        if pdgids_to_index is not None:
            self.index_to_pdgid = {int(index): int(pdgid) for pdgid, index in pdgids_to_index.items()}

    def untokenize_file(self) -> None:
        """
        Stream the generated samples file and write untokenized raw-style rows.

        Invalid generated rows are dropped from the untokenized CSV. A separate
        JSON file records the sample row, zero-based event index, first token,
        and reason that made the row invalid so generation failures can be
        debugged later.
        """
        self.output_samples_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.output_invalid_tokens_filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_filepath = self.output_samples_filepath.with_suffix(self.output_samples_filepath.suffix + ".tmp")
        tmp_invalid_tokens_filepath = self.output_invalid_tokens_filepath.with_suffix(self.output_invalid_tokens_filepath.suffix + ".tmp")
        invalid_records = []
        
        with self.input_samples_filepath.open("r", encoding="utf-8") as in_file, \
             tmp_output_filepath.open("w", encoding="utf-8", newline="") as out_file, \
             tmp_invalid_tokens_filepath.open("w", encoding="utf-8") as invalid_file:
            for event_idx, line in enumerate(in_file):
                line = line.strip()
                if not line:
                    continue
                
                tokens = self.parse_token_row(line, event_idx + 1)
                particles, invalid_token, invalid_reason = self.decode_token_row(tokens)
                self.total_samples_read += 1
                
                if invalid_token is not None:
                    self.total_invalid_tokens += 1
                    self.total_invalid_samples += 1
                    invalid_records.append({
                        "event_index": event_idx,
                        "invalid_token": int(invalid_token),
                        "reason": invalid_reason,
                        "event_row": line,
                    })
                    continue
                
                out_file.write(self.format_particles(particles) + "\n")
                self.total_samples_written += 1
                if len(particles) == 0:
                    self.total_empty_samples += 1
            
            json.dump(invalid_records, invalid_file, indent=4)
            invalid_file.write("\n")
        
        tmp_output_filepath.replace(self.output_samples_filepath)
        tmp_invalid_tokens_filepath.replace(self.output_invalid_tokens_filepath)
        self.write_metadata()

    def parse_token_row(self, row: str, line_idx: int) -> list[int]:
        """
        Parse one whitespace-separated row from sample.py.

        Comma-separated rows are intentionally rejected so stale outputs are not
        silently interpreted incorrectly.
        """
        if "," in row:
            raise ValueError(
                f"Found comma-separated tokens on line {line_idx}. "
                "Current sample.py writes whitespace-separated tokens only."
            )
        try:
            return [int(token) for token in row.split()]
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse token ids on line {line_idx}: {row[:500]}") from exc

    def decode_token_row(self, tokens: list[int]) -> tuple[list[dict[str, float | int]], int | None, str | None]:
        raise NotImplementedError("Subclasses must implement this method.")

    def format_particles(self, particles: list[dict[str, float | int]]) -> str:
        """
        Format decoded particles as a semicolon-separated raw-style event row.
        """
        formatted_particles = []
        for particle in particles:
            pdgid = int(particle.get("pdgid", self.default_pdgid))
            energy = self.format_float(float(particle.get("energy", 0.0)))
            px = self.format_float(float(particle.get("px", 0.0)))
            py = self.format_float(float(particle.get("py", 0.0)))
            pz = self.format_float(float(particle.get("pz", 0.0)))
            formatted_particles.append(f"{pdgid} {energy} {px} {py} {pz}")
        return ";".join(formatted_particles)

    def format_float(self, value: float) -> str:
        """Format floats with a fixed number of digits after the decimal point."""
        return f"{value:.{self.float_precision}f}"

    def token_is_padding(self, token: int) -> bool:
        """Return True if token is any padding token used by the dictionary."""
        padding_tokens = {0}
        padding_token = getattr(self.dictionary, "padding_token", None)
        if padding_token is not None:
            padding_tokens.add(int(padding_token))
        padding_sequence = getattr(self.dictionary, "get_padding_sequence", None)
        if callable(padding_sequence):
            padding_tokens.update(int(x) for x in padding_sequence())
        return int(token) in padding_tokens

    def write_metadata(self) -> None:
        """Write a small sidecar JSON file describing the untokenized output."""
        metadata = {
            "model_name": conf.generic.model_name,
            "input_samples_filepath": paths.project_relative_path(self.input_samples_filepath),
            "output_samples_filepath": paths.project_relative_path(self.output_samples_filepath),
            "tokenization_format": self.tokenization_format,
            "tokenization_schema": self.tokenization_schema,
            "default_pdgid": self.default_pdgid,
            "stop_at_event_end": self.stop_at_event_end,
            "stop_at_padding": self.stop_at_padding,
            "float_precision": self.float_precision,
            "total_samples_read": self.total_samples_read,
            "total_samples_written": self.total_samples_written,
            "total_empty_samples": self.total_empty_samples,
            "total_invalid_tokens": self.total_invalid_tokens,
            "total_invalid_samples": self.total_invalid_samples,
        }
        with self.output_metadata_filepath.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

class ParticleFeatureUntokenizer(BaseUntokenizer):
    """
    Untokenizer for feature-token particle data.

    This handles token streams produced by EventPerSequenceParticleFeatureTokenizer.
    It uses the dictionary schema to group per-particle tokens and then
    reconstructs raw-style particles.
    """
    
    def _is_event_start_token(self, token: int) -> bool:
        return token == int(self.dictionary.event_start_token)

    def _is_event_end_token(self, token: int) -> bool:
        return token == int(self.dictionary.event_end_token)
    
    def is_event_start_token(self, token: int) -> bool:
        return self._is_event_start_token(token)

    def is_event_end_token(self, token: int) -> bool:
        return self._is_event_end_token(token)
    
    def decode_token_row(self, tokens: list[int]) -> tuple[list[dict[str, float | int]], int | None, str | None]:
        """
        Decode one generated token row into raw-style particles.

        The decoder consumes tokens according to dictionary.tokenization_schema.
        If any token is invalid for the schema position where it appears, the
        whole row is rejected and the first invalid token is returned.
        """
        particles = []
        feature_buffer = {}
        schema_idx = 0
        seen_event_start = False
        schema_to_offset_and_bins = {
            "pt": ("PT_OFFSET", "pt_bins"),
            "px": ("PX_OFFSET", "px_bins"),
            "py": ("PY_OFFSET", "py_bins"),
            "pz": ("PZ_OFFSET", "pz_bins"),
            "energy": ("ENERGY_OFFSET", "e_bins"),
            "eta": ("ETA_OFFSET", "eta_bins"),
            "theta": ("THETA_OFFSET", "theta_bins"),
            "phi": ("PHI_OFFSET", "phi_bins"),
        }
        
        for token in tokens:
            token = int(token)
            if self._is_event_start_token(token):
                if seen_event_start or particles or feature_buffer or schema_idx != 0:
                    return [], token, "encountered EVENT_START after event decoding had already started"
                seen_event_start = True
                feature_buffer = {}
                schema_idx = 0
                continue
            if self._is_event_end_token(token):
                if feature_buffer or schema_idx != 0:
                    return [], token, (
                        "encountered EVENT_END before completing the current particle; "
                        f"expected {self.tokenization_schema[schema_idx]!r} at schema position {schema_idx}"
                    )
                if self.stop_at_event_end:
                    break
                feature_buffer = {}
                schema_idx = 0
                continue
            if self.stop_at_padding and self.token_is_padding(token):
                if feature_buffer or schema_idx != 0:
                    return [], token, (
                        "encountered padding before completing the current particle; "
                        f"expected {self.tokenization_schema[schema_idx]!r} at schema position {schema_idx}"
                    )
                break
            
            schema_name = self.tokenization_schema[schema_idx]
            next_schema_idx = (schema_idx + 1) % len(self.tokenization_schema)
            
            if schema_name == "particle_start":
                if not self.is_particle_start_token(token):
                    expected = int(self.dictionary.particle_start_token)
                    return [], token, f"expected particle_start token {expected}, got token {token}"
                feature_buffer = {}
                schema_idx = next_schema_idx
                continue
            if schema_name == "particle_end":
                if not self.is_particle_end_token(token):
                    expected = int(self.dictionary.particle_end_token)
                    return [], token, f"expected particle_end token {expected}, got token {token}"
                particles.append(self.finalize_particle(feature_buffer))
                feature_buffer = {}
                schema_idx = next_schema_idx
                continue
            
            decoded_value, invalid_reason = self.decode_feature_token(schema_name, token, schema_to_offset_and_bins)
            if invalid_reason is not None:
                return [], token, invalid_reason
            feature_buffer[schema_name] = decoded_value
            schema_idx = next_schema_idx
            
            if schema_idx == 0:
                particles.append(self.finalize_particle(feature_buffer))
                feature_buffer = {}
        
        return particles, None, None

    def is_particle_start_token(self, token: int) -> bool:
        return hasattr(self.dictionary, "particle_start_token") and token == int(self.dictionary.particle_start_token)

    def is_particle_end_token(self, token: int) -> bool:
        return hasattr(self.dictionary, "particle_end_token") and token == int(self.dictionary.particle_end_token)

    def decode_feature_token(self, schema_name: str, token: int, schema_to_offset_and_bins: dict[str, tuple[str, str]]) -> tuple[float | int | None, str | None]:
        """
        Decode one feature token using the offset and bins for its schema field.
        """
        if schema_name == "pdgid":
            offset = int(getattr(self.dictionary, "PDGID_OFFSET"))
            particle_index = int(token) - offset
            if particle_index < 0:
                return None, f"expected pdgid token >= {offset}, got token {token}"
            if self.index_to_pdgid and particle_index not in self.index_to_pdgid:
                max_index = max(self.index_to_pdgid)
                return None, (
                    f"expected pdgid token in [{offset}, {offset + max_index}], "
                    f"got token {token}"
                )
            return self.index_to_pdgid.get(particle_index, self.default_pdgid), None
        
        if schema_name not in schema_to_offset_and_bins:
            raise RuntimeError(f"Untokenizer: Unknown tokenization schema: {schema_name}")
        
        offset_attr, bins_attr = schema_to_offset_and_bins[schema_name]
        offset = int(getattr(self.dictionary, offset_attr))
        bins = np.asarray(getattr(self.dictionary, bins_attr), dtype=np.float64)
        bin_idx = int(token) - offset
        valid_start = offset
        valid_end_exclusive = offset + len(bins)
        if bin_idx < 0 or bin_idx >= len(bins):
            return None, (
                f"expected {schema_name} token in [{valid_start}, {valid_end_exclusive}), "
                f"got token {token}"
            )
        return bin_value_from_index(bin_idx, bins), None

    def finalize_particle(self, feature_buffer: dict[str, float | int]) -> dict[str, float | int]:
        """
        Convert decoded feature values into raw-style pdgid, energy, px, py, pz.
        """
        pdgid = int(feature_buffer.get("pdgid", self.default_pdgid))
        px = feature_buffer.get("px", None)
        py = feature_buffer.get("py", None)
        pz = feature_buffer.get("pz", None)
        energy = feature_buffer.get("energy", None)
        
        if px is None or py is None or pz is None:
            px, py, pz = momentum_from_features(feature_buffer)
        if energy is None:
            energy = math.sqrt(float(px) ** 2 + float(py) ** 2 + float(pz) ** 2)
        
        return {
            "pdgid": pdgid,
            "energy": float(energy),
            "px": float(px),
            "py": float(py),
            "pz": float(pz),
        }

class PackedEventStreamParticleFeatureUntokenizer(BaseUntokenizer):
    """
    Untokenizer for packed particle-feature token streams.

    Packed samples can contain more than one generated event per row because the
    tokenizer concatenates events into fixed-length training sequences. Events
    may also cross row boundaries, so this class treats the input file as one
    continuous token stream and decodes complete EVENT_START...EVENT_END spans.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_source_rows_read = 0
        self.total_complete_events_found = 0
        self.total_partial_events = 0
        self.total_events_spanning_rows = 0
    
    def untokenize_file(self) -> None:
        """
        Stream packed generated samples and write one untokenized row per event.

        Invalid events are dropped independently. A malformed event in a packed
        stream does not poison other complete events from that same stream.
        """
        self.output_samples_filepath.parent.mkdir(parents=True, exist_ok=True)
        self.output_invalid_tokens_filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_filepath = self.output_samples_filepath.with_suffix(self.output_samples_filepath.suffix + ".tmp")
        tmp_invalid_tokens_filepath = self.output_invalid_tokens_filepath.with_suffix(self.output_invalid_tokens_filepath.suffix + ".tmp")
        invalid_records = []
        current_event = None
        current_event_start_row_idx = None
        current_event_start_token_idx = None
        last_source_row_idx = None
        
        with self.input_samples_filepath.open("r", encoding="utf-8") as in_file, \
             tmp_output_filepath.open("w", encoding="utf-8", newline="") as out_file, \
             tmp_invalid_tokens_filepath.open("w", encoding="utf-8") as invalid_file:
            for source_row_idx, line in enumerate(in_file):
                line = line.strip()
                if not line:
                    continue
                
                source_tokens = self.parse_token_row(line, source_row_idx + 1)
                self.total_source_rows_read += 1
                last_source_row_idx = source_row_idx
                
                for token_idx, token in enumerate(source_tokens):
                    token = int(token)
                    if self.stop_at_padding and self.token_is_padding(token):
                        if current_event is not None:
                            self.record_partial_event(
                                invalid_records,
                                current_event,
                                current_event_start_row_idx,
                                current_event_start_token_idx,
                                source_row_idx,
                                token_idx,
                                "encountered padding before EVENT_END",
                                invalid_token=token,
                            )
                            current_event = None
                            current_event_start_row_idx = None
                            current_event_start_token_idx = None
                        break
                    
                    if self._is_event_start_token(token):
                        if current_event is not None:
                            self.record_partial_event(
                                invalid_records,
                                current_event,
                                current_event_start_row_idx,
                                current_event_start_token_idx,
                                source_row_idx,
                                token_idx,
                                "encountered EVENT_START before previous EVENT_END",
                                invalid_token=token,
                            )
                        current_event = [token]
                        current_event_start_row_idx = source_row_idx
                        current_event_start_token_idx = token_idx
                        continue
                    
                    if current_event is None:
                        continue
                    
                    current_event.append(token)
                    if self._is_event_end_token(token):
                        self.write_complete_event(
                            out_file,
                            invalid_records,
                            current_event,
                            current_event_start_row_idx,
                            current_event_start_token_idx,
                            source_row_idx,
                            token_idx,
                        )
                        current_event = None
                        current_event_start_row_idx = None
                        current_event_start_token_idx = None
            
            if current_event is not None:
                self.record_partial_event(
                    invalid_records,
                    current_event,
                    current_event_start_row_idx,
                    current_event_start_token_idx,
                    last_source_row_idx,
                    None,
                    "reached end of file before EVENT_END",
                )
            
            json.dump(invalid_records, invalid_file, indent=4)
            invalid_file.write("\n")
        
        tmp_output_filepath.replace(self.output_samples_filepath)
        tmp_invalid_tokens_filepath.replace(self.output_invalid_tokens_filepath)
        self.write_metadata()
    
    def write_complete_event(
        self,
        out_file,
        invalid_records: list[dict[str, Any]],
        event_tokens: list[int],
        source_row_start_index: int,
        source_token_start_index: int,
        source_row_end_index: int,
        source_token_end_index: int,
    ) -> None:
        """Decode and write one complete event span."""
        event_index = self.total_samples_read
        self.total_samples_read += 1
        self.total_complete_events_found += 1
        if source_row_start_index != source_row_end_index:
            self.total_events_spanning_rows += 1
        
        particles, invalid_token, invalid_reason = self.decode_token_row(event_tokens)
        if invalid_token is not None:
            self.total_invalid_tokens += 1
            self.total_invalid_samples += 1
            invalid_records.append({
                "event_index": event_index,
                "source_row_start_index": source_row_start_index,
                "source_token_start_index": source_token_start_index,
                "source_row_end_index": source_row_end_index,
                "source_token_end_index": source_token_end_index,
                "invalid_token": int(invalid_token),
                "reason": invalid_reason,
                "event_row": " ".join(str(int(token)) for token in event_tokens),
            })
            return
        
        out_file.write(self.format_particles(particles) + "\n")
        self.total_samples_written += 1
        if len(particles) == 0:
            self.total_empty_samples += 1
    
    def record_partial_event(
        self,
        invalid_records: list[dict[str, Any]],
        event_tokens: list[int],
        source_row_start_index: int,
        source_token_start_index: int,
        source_row_end_index: int,
        source_token_end_index: int | None,
        reason: str,
        invalid_token: int | None = None,
    ) -> None:
        """Record an event span that never reached EVENT_END."""
        self.total_partial_events += 1
        self.total_invalid_samples += 1
        if invalid_token is not None:
            self.total_invalid_tokens += 1
        invalid_records.append({
            "event_index": None,
            "partial_event_index": self.total_partial_events - 1,
            "source_row_start_index": source_row_start_index,
            "source_token_start_index": source_token_start_index,
            "source_row_end_index": source_row_end_index,
            "source_token_end_index": source_token_end_index,
            "invalid_token": None if invalid_token is None else int(invalid_token),
            "reason": reason,
            "event_row": " ".join(str(int(token)) for token in event_tokens),
        })
    
    def split_packed_token_row(self, tokens: list[int]) -> tuple[list[list[int]], list[list[int]]]:
        """
        Split one packed generated row into complete EVENT_START...EVENT_END spans.

        This helper is retained for callers that only need row-local splitting.
        File untokenization uses a streaming parser so event spans can cross row
        boundaries.
        """
        event_spans = []
        partial_spans = []
        current_event = None
        
        for token in tokens:
            token = int(token)
            if self.stop_at_padding and self.token_is_padding(token):
                break
            
            if self._is_event_start_token(token):
                if current_event:
                    partial_spans.append(current_event)
                current_event = [token]
                continue
            
            if current_event is None:
                continue
            
            current_event.append(token)
            if self._is_event_end_token(token):
                event_spans.append(current_event)
                current_event = None
        
        if current_event:
            partial_spans.append(current_event)
        
        return event_spans, partial_spans
    
    def _is_event_start_token(self, token: int) -> bool:
        return token == int(self.dictionary.event_start_token)

    def _is_event_end_token(self, token: int) -> bool:
        return token == int(self.dictionary.event_end_token)
    
    def decode_token_row(self, tokens: list[int]) -> tuple[list[dict[str, float | int]], int | None, str | None]:
        """
        Decode one complete EVENT_START...EVENT_END token span.

        The decoder consumes tokens according to dictionary.tokenization_schema.
        If any token is invalid for the schema position where it appears, the
        event is rejected and the first invalid token is returned.
        """
        particles = []
        feature_buffer = {}
        schema_idx = 0
        seen_event_start = False
        schema_to_offset_and_bins = {
            "pt": ("PT_OFFSET", "pt_bins"),
            "px": ("PX_OFFSET", "px_bins"),
            "py": ("PY_OFFSET", "py_bins"),
            "pz": ("PZ_OFFSET", "pz_bins"),
            "energy": ("ENERGY_OFFSET", "e_bins"),
            "eta": ("ETA_OFFSET", "eta_bins"),
            "theta": ("THETA_OFFSET", "theta_bins"),
            "phi": ("PHI_OFFSET", "phi_bins"),
        }
        
        for token in tokens:
            token = int(token)
            if self._is_event_start_token(token):
                if seen_event_start or particles or feature_buffer or schema_idx != 0:
                    return [], token, "encountered EVENT_START after event decoding had already started"
                seen_event_start = True
                feature_buffer = {}
                schema_idx = 0
                continue
            if self._is_event_end_token(token):
                if feature_buffer or schema_idx != 0:
                    return [], token, (
                        "encountered EVENT_END before completing the current particle; "
                        f"expected {self.tokenization_schema[schema_idx]!r} at schema position {schema_idx}"
                    )
                break
            if self.stop_at_padding and self.token_is_padding(token):
                if feature_buffer or schema_idx != 0:
                    return [], token, (
                        "encountered padding before completing the current particle; "
                        f"expected {self.tokenization_schema[schema_idx]!r} at schema position {schema_idx}"
                    )
                break
            
            schema_name = self.tokenization_schema[schema_idx]
            next_schema_idx = (schema_idx + 1) % len(self.tokenization_schema)
            
            if schema_name == "particle_start":
                if not self.is_particle_start_token(token):
                    expected = int(self.dictionary.particle_start_token)
                    return [], token, f"expected particle_start token {expected}, got token {token}"
                feature_buffer = {}
                schema_idx = next_schema_idx
                continue
            if schema_name == "particle_end":
                if not self.is_particle_end_token(token):
                    expected = int(self.dictionary.particle_end_token)
                    return [], token, f"expected particle_end token {expected}, got token {token}"
                particles.append(self.finalize_particle(feature_buffer))
                feature_buffer = {}
                schema_idx = next_schema_idx
                continue
            
            decoded_value, invalid_reason = self.decode_feature_token(schema_name, token, schema_to_offset_and_bins)
            if invalid_reason is not None:
                return [], token, invalid_reason
            feature_buffer[schema_name] = decoded_value
            schema_idx = next_schema_idx
            
            if schema_idx == 0:
                particles.append(self.finalize_particle(feature_buffer))
                feature_buffer = {}
        
        return particles, None, None

    def is_particle_start_token(self, token: int) -> bool:
        return hasattr(self.dictionary, "particle_start_token") and token == int(self.dictionary.particle_start_token)

    def is_particle_end_token(self, token: int) -> bool:
        return hasattr(self.dictionary, "particle_end_token") and token == int(self.dictionary.particle_end_token)

    def decode_feature_token(self, schema_name: str, token: int, schema_to_offset_and_bins: dict[str, tuple[str, str]]) -> tuple[float | int | None, str | None]:
        """
        Decode one feature token using the offset and bins for its schema field.
        """
        if schema_name == "pdgid":
            offset = int(getattr(self.dictionary, "PDGID_OFFSET"))
            particle_index = int(token) - offset
            if particle_index < 0:
                return None, f"expected pdgid token >= {offset}, got token {token}"
            if self.index_to_pdgid and particle_index not in self.index_to_pdgid:
                max_index = max(self.index_to_pdgid)
                return None, (
                    f"expected pdgid token in [{offset}, {offset + max_index}], "
                    f"got token {token}"
                )
            return self.index_to_pdgid.get(particle_index, self.default_pdgid), None
        
        if schema_name not in schema_to_offset_and_bins:
            raise RuntimeError(f"Untokenizer: Unknown tokenization schema: {schema_name}")
        
        offset_attr, bins_attr = schema_to_offset_and_bins[schema_name]
        offset = int(getattr(self.dictionary, offset_attr))
        bins = np.asarray(getattr(self.dictionary, bins_attr), dtype=np.float64)
        bin_idx = int(token) - offset
        valid_start = offset
        valid_end_exclusive = offset + len(bins)
        if bin_idx < 0 or bin_idx >= len(bins):
            return None, (
                f"expected {schema_name} token in [{valid_start}, {valid_end_exclusive}), "
                f"got token {token}"
            )
        return bin_value_from_index(bin_idx, bins), None

    def finalize_particle(self, feature_buffer: dict[str, float | int]) -> dict[str, float | int]:
        """
        Convert decoded feature values into raw-style pdgid, energy, px, py, pz.
        """
        pdgid = int(feature_buffer.get("pdgid", self.default_pdgid))
        px = feature_buffer.get("px", None)
        py = feature_buffer.get("py", None)
        pz = feature_buffer.get("pz", None)
        energy = feature_buffer.get("energy", None)
        
        if px is None or py is None or pz is None:
            px, py, pz = momentum_from_features(feature_buffer)
        if energy is None:
            energy = math.sqrt(float(px) ** 2 + float(py) ** 2 + float(pz) ** 2)
        
        return {
            "pdgid": pdgid,
            "energy": float(energy),
            "px": float(px),
            "py": float(py),
            "pz": float(pz),
        }
    
    def write_metadata(self) -> None:
        """Write metadata including packed-stream source-row accounting."""
        super().write_metadata()
        with self.output_metadata_filepath.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata.update({
            "packed_event_stream": True,
            "total_source_rows_read": self.total_source_rows_read,
            "total_complete_events_found": self.total_complete_events_found,
            "total_partial_events": self.total_partial_events,
            "total_events_spanning_rows": self.total_events_spanning_rows,
        })
        with self.output_metadata_filepath.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

class WholeParticleUntokenizer(BaseUntokenizer):
    """
    Untokenizer for EventPerSequenceWholeParticleTokenizer.

    The tokenizer uses mixed-radix particle ids:
        token = ETA_OFFSET + eta_bin + (n_eta_bins * pt_bin) + (n_eta_bins * n_pt_bins * phi_bin)

    Because PDGID and energy are not encoded by this tokenizer, this untokenizer
    writes default_pdgid and reconstructs energy using a massless approximation.
    """
    
    def decode_token_row(self, tokens: list[int]) -> tuple[list[dict[str, float | int]], int | None, str | None]:
        particles = []
        base_offset = int(getattr(self.dictionary, "ETA_OFFSET"))
        n_eta_bins = len(self.dictionary.eta_bins)
        n_pt_bins = len(self.dictionary.pt_bins)
        n_phi_bins = len(self.dictionary.phi_bins)
        vocab_end = base_offset + (n_eta_bins * n_pt_bins * n_phi_bins)
        seen_event_start = False
        
        for token in tokens:
            token = int(token)
            if self.is_event_start_token(token):
                if seen_event_start or particles:
                    return [], token, "encountered EVENT_START after event decoding had already started"
                seen_event_start = True
                continue
            if self.is_event_end_token(token):
                if self.stop_at_event_end:
                    break
                continue
            if self.stop_at_padding and self.token_is_padding(token):
                break
            if token < base_offset or token >= vocab_end:
                return [], token, f"expected whole-particle token in [{base_offset}, {vocab_end}), got token {token}"
            
            local_token = token - base_offset
            eta_bin = local_token % n_eta_bins
            pt_bin = (local_token // n_eta_bins) % n_pt_bins
            phi_bin = local_token // (n_eta_bins * n_pt_bins)
            
            feature_buffer = {
                "eta": bin_value_from_index(eta_bin, np.asarray(self.dictionary.eta_bins, dtype=np.float64)),
                "pt": bin_value_from_index(pt_bin, np.asarray(self.dictionary.pt_bins, dtype=np.float64)),
                "phi": bin_value_from_index(phi_bin, np.asarray(self.dictionary.phi_bins, dtype=np.float64)),
            }
            px, py, pz = momentum_from_features(feature_buffer)
            energy = math.sqrt(float(px) ** 2 + float(py) ** 2 + float(pz) ** 2)
            particles.append({
                "pdgid": self.default_pdgid,
                "energy": float(energy),
                "px": float(px),
                "py": float(py),
                "pz": float(pz),
            })
        
        return particles, None, None

    def is_event_start_token(self, token: int) -> bool:
        return hasattr(self.dictionary, "event_start_token") and token == int(self.dictionary.event_start_token)

    def is_event_end_token(self, token: int) -> bool:
        return hasattr(self.dictionary, "event_end_token") and token == int(self.dictionary.event_end_token)

def bin_value_from_index(bin_idx: int, bins: np.ndarray) -> float:
    """
    Convert a tokenizer bin index back to a representative continuous value.

    tokenizer.py uses np.searchsorted(..., side='right') plus explicit underflow
    and overflow clipping. Interior bins are represented by the midpoint of the
    adjacent bin boundaries. Edge bins use the edge value itself because the true
    underflow/overflow value is not recoverable from the token id alone.
    """
    if len(bins) == 0:
        raise ValueError("Cannot decode from an empty bin array.")
    if bin_idx <= 0:
        return float(bins[0])
    if bin_idx >= len(bins) - 1:
        return float(bins[-1])
    return float(0.5 * (bins[bin_idx - 1] + bins[bin_idx]))

def momentum_from_features(feature_buffer: dict[str, float | int]) -> tuple[float, float, float]:
    """
    Reconstruct px, py, pz from the available kinematic feature tokens.

    Preferred reconstruction path:
      1) pt, eta, phi
      2) pt, theta, phi
      3) px, py, pz if already available
    """
    if all(key in feature_buffer for key in ("px", "py", "pz")):
        return float(feature_buffer["px"]), float(feature_buffer["py"]), float(feature_buffer["pz"])
    
    pt = float(feature_buffer.get("pt", 0.0))
    phi = float(feature_buffer.get("phi", 0.0))
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    
    if "eta" in feature_buffer:
        pz = pt * math.sinh(float(feature_buffer["eta"]))
    elif "theta" in feature_buffer:
        theta = float(feature_buffer["theta"])
        pz = 0.0 if abs(math.tan(theta)) < 1e-12 else pt / math.tan(theta)
    else:
        pz = float(feature_buffer.get("pz", 0.0))
    
    return px, py, pz

def resolve_sampling_idx(generated_samples_dir: Path) -> int:
    """
    Resolve the sampling_N directory to untokenize.

    If untokenizing.sample_idx or sampling_idx is set, that value is used.
    Otherwise, the highest existing sampling_N directory is used because the
    usual workflow is to untokenize the newest sampling run.
    """
    if conf.sampling.sampling_idx_override is not None:
        return int(conf.sampling.sampling_idx_override)
    
    max_idx = -1
    if generated_samples_dir.exists():
        for path in generated_samples_dir.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"sampling_(\d+)", path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
    
    if max_idx < 0:
        raise FileNotFoundError(f"No sampling_N directories found in {generated_samples_dir}.")
    
    return max_idx

def main() -> None:
    """Untokenize generated samples from the selected sampling_N directory."""
    if conf.generic.preparation_config_file is None:
        raise ValueError("preparation_config_file in configuration cannot be None!")
    
    preparation_config_filepath = paths.PROJECT_DIR / conf.generic.preparation_config_file
    dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, preparation_config_filepath)
    if not dls_conf.verify():
        raise RuntimeError("Failure when verifying test split config. Ensure all required arguments exist.")
    
    tokenized_metadata_filepath = paths.PROJECT_DIR / dls_conf.tokenized_metadata_filepath
    try:
        with tokenized_metadata_filepath.open("r", encoding="utf-8") as f:
            tokenized_metadata = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failure while trying to load json from {tokenized_metadata_filepath}! Exception:\n{exc}") from exc
    
    tmd_conf = TokenizedMetadataConfig(tokenized_metadata_filepath)
    if not tmd_conf.verify():
        raise RuntimeError("Failure when verifying tokenized metadata config. Ensure all required arguments exist.")
    
    dictionary = pUtil.get_dictionary(conf.generic.preparation_config_filepath)
    
    generated_samples_dir = PROJECT_DIR / "generated_samples" / conf.generic.model_name
    sample_idx = resolve_sampling_idx(generated_samples_dir)
    output_dir = generated_samples_dir / f"sampling_{sample_idx}"
    
    input_samples_filepath = output_dir / DEFAULT_INPUT_CSV_NAME
    if not input_samples_filepath.exists():
        raise FileNotFoundError(f"Input generated samples file does not exist: {input_samples_filepath}")
    
    output_samples_filepath = output_dir / DEFAULT_OUTPUT_CSV_NAME
    output_metadata_filepath = output_dir / DEFAULT_METADATA_NAME
    output_invalid_tokens_filepath = output_dir / DEFAULT_INVALID_TOKENS_NAME
    
    tokenization_format = str(tokenized_metadata.get("format", "base_tokenizer"))
    tokenizer_class = str(tokenized_metadata.get("tokenizer_class", ""))
    if tokenization_format == "whole_particle" or tokenizer_class == "EventPerSequenceWholeParticleTokenizer":
        untokenizer = WholeParticleUntokenizer(dictionary, input_samples_filepath, output_samples_filepath, output_metadata_filepath, output_invalid_tokens_filepath, tokenized_metadata)
    elif tokenizer_class == "PackedEventStreamParticleFeatureTokenizer":
        untokenizer = PackedEventStreamParticleFeatureUntokenizer(dictionary, input_samples_filepath, output_samples_filepath, output_metadata_filepath, output_invalid_tokens_filepath, tokenized_metadata)
    else:
        untokenizer = ParticleFeatureUntokenizer(dictionary, input_samples_filepath, output_samples_filepath, output_metadata_filepath, output_invalid_tokens_filepath, tokenized_metadata)
    untokenizer.untokenize_file()
    
    print(f"Untokenized samples from: {input_samples_filepath}")
    print(f"Untokenized samples written to: {output_samples_filepath}")
    print(f"Untokenizing metadata written to: {output_metadata_filepath}")
    print(f"Invalid token debug records written to: {output_invalid_tokens_filepath}")
    print(f"Samples read: {untokenizer.total_samples_read:,}")
    print(f"Samples written: {untokenizer.total_samples_written:,}")
    print(f"Samples dropped for invalid tokens: {untokenizer.total_invalid_samples:,}")
    print(f"Empty samples: {untokenizer.total_empty_samples:,}")

if __name__ == "__main__":
    main()