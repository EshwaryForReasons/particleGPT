"""
Use from main project directory as:

python -m analysis.analyzer config/model_config_file.json

additional optional arguments:
    --no-metrics: If provided, skip metric calculations.
    --no-distributions: If provided, skip distribution generation.
    --no-untokenize: If provided, skip untokenization of generated data.
"""

import sys
import json
import os
import numpy as np
from pathlib import Path
import argparse
import warnings

import particleGPT.configurator as conf
import paths
import data_manager
import matplotlib.pyplot as plt
from analysis.dataset import dataset
from analysis.metrics import metrics
from analysis.tables import tables
from analysis.plotting import plotting_v2 as plotting
import pUtil
import particleGPT.untokenizer as untokenizer
from train import DataloaderSplitConfig
from train import ESplitTypes
from train import TokenizedMetadataConfig
from particleGPT.dictionary import Dictionary

class Analyzer:
    
    @staticmethod
    def get_latest_sampling_dir(model_name):
        """
        Return the selected sampling directory for the current sampling pipeline.

        The sampler writes generated samples to:
            PROJECT_DIR/generated_samples/<model_name>/sampling_<idx>/

        untokenizer.resolve_sampling_idx handles explicit sample_idx config values
        and otherwise chooses the newest sampling_<idx> directory.
        """
        generated_samples_dir = paths.PROJECT_DIR / 'generated_samples' / model_name
        try:
            sampling_idx = untokenizer.resolve_sampling_idx(generated_samples_dir)
        except FileNotFoundError:
            return generated_samples_dir / 'sampling_0'
        return generated_samples_dir / f'sampling_{sampling_idx}'

    def __init__(self, model_name):
        self.model_name = model_name
        self.latest_sampling_dir = self.get_latest_sampling_dir(model_name)

        if conf.generic.preparation_config_file is None:
            raise ValueError("preparation_config_file in configuration cannot be None!")
        
        self.preparation_config_filename = paths.PROJECT_DIR / conf.generic.preparation_config_file
        self.dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, self.preparation_config_filename)
        if not self.dls_conf.verify():
            raise RuntimeError("Failure when verifying dataloader split config. Ensure all required arguments exist.")

        self.tokenized_metadata_filename = paths.PROJECT_DIR / self.dls_conf.tokenized_metadata_filepath
        try:
            with open(self.tokenized_metadata_filename, 'r') as f:
                self.tokenized_metadata = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failure while trying to load json from {self.tokenized_metadata_filename}! Exception:\n{exc}") from exc

        # TokenizedMetadataConfig is reused from train.py. Temporarily using the
        # project directory as cwd makes relative paths in tokenized metadata behave
        # the same way as scripts launched from PROJECT_DIR.
        current_dir = Path.cwd()
        try:
            os.chdir(paths.PROJECT_DIR)
            self.tmd_conf = TokenizedMetadataConfig(self.tokenized_metadata_filename)
        finally:
            os.chdir(current_dir)

        if not self.tmd_conf.verify():
            raise RuntimeError("Failure when verifying tokenized metadata config. Ensure all required arguments exist.")
        if not self.tmd_conf.tokenized_data_filepath.is_absolute():
            self.tmd_conf.tokenized_data_filepath = paths.PROJECT_DIR / self.tmd_conf.tokenized_data_filepath

        # The tokenized metadata stores the dictionary path relative to PROJECT_DIR.
        # Reuse the untokenizer's resolver so analysis and untokenization agree.
        self.real_test_tokens_filename                  = self.latest_sampling_dir / 'real_test_tokens.csv'
        self.real_test_untokenized_filename             = self.latest_sampling_dir / 'real_test_untokenized_samples.csv'
        self.real_test_untokenizing_metadata_filename   = self.latest_sampling_dir / 'real_test_untokenizing_metadata.json'
        self.real_test_invalid_tokens_filename          = self.latest_sampling_dir / 'real_test_invalid_token_events.json'
        self.generated_samples_filename                 = self.latest_sampling_dir / untokenizer.DEFAULT_INPUT_CSV_NAME
        self.filtered_samples_filename                  = self.latest_sampling_dir / 'filtered_samples.csv'
        self.sampled_leading_particles_filename         = self.latest_sampling_dir / 'sampled_leading_particles.csv'
        self.verbose_particles_filename                 = self.latest_sampling_dir / 'untokenized_samples_verbose.csv'
        self.untokenized_samples_filename               = self.latest_sampling_dir / untokenizer.DEFAULT_OUTPUT_CSV_NAME
        self.untokenizing_metadata_filename             = self.latest_sampling_dir / untokenizer.DEFAULT_METADATA_NAME
        self.invalid_tokens_filename                    = self.latest_sampling_dir / untokenizer.DEFAULT_INVALID_TOKENS_NAME
        self.metrics_results_filename                   = self.latest_sampling_dir / 'metrics_results.json'
        self.sampling_metadata_filename                 = self.latest_sampling_dir / 'sampling_metadata.json'
        self.plotted_distributions_dir                  = self.latest_sampling_dir / 'plotted_distributions'

        if self.sampling_metadata_filename.exists():
            try:
                with open(self.sampling_metadata_filename, 'r') as f:
                    self.sampling_metadata = json.load(f)
            except Exception as exc:
                raise RuntimeError(f"Failure while trying to load json from {self.sampling_metadata_filename}! Exception:\n{exc}") from exc

            final_csv_path = self.sampling_metadata.get('final_csv_path', None)
            if final_csv_path is not None:
                final_csv_path = paths.PROJECT_DIR / Path(final_csv_path)
                if final_csv_path.exists():
                    self.generated_samples_filename = final_csv_path
        else:
            self.sampling_metadata = {}

        if not self.generated_samples_filename.exists():
            raise FileNotFoundError(f'generated_samples_filename does not exist: {self.generated_samples_filename}')

        file_bytes = self.tmd_conf.tokenized_data_filepath.stat().st_size
        dtype_bytes = np.dtype(self.tmd_conf.dtype).itemsize
        if file_bytes % dtype_bytes != 0:
            raise ValueError(
                f"Size of tokenized data file ({file_bytes} bytes) is not divisible by dtype size {dtype_bytes}."
            )

        data_total_tokens = file_bytes // dtype_bytes
        if data_total_tokens != self.tmd_conf.total_tokens:
            raise ValueError(
                f"Tokenized data contains {data_total_tokens:,} tokens, but metadata expected {self.tmd_conf.total_tokens:,}."
            )

        raw_split_tokens = self.dls_conf.num_sequences * self.tmd_conf.sequence_length
        if self.dls_conf.from_end:
            raw_end = (self.tmd_conf.num_full_sequences - self.dls_conf.skip_sequences) * self.tmd_conf.sequence_length
            raw_start = raw_end - raw_split_tokens
        else:
            raw_start = self.dls_conf.skip_sequences * self.tmd_conf.sequence_length
            raw_end = raw_start + raw_split_tokens

        if raw_start < 0 or raw_end > self.tmd_conf.total_tokens or raw_start >= raw_end:
            raise ValueError(
                f"Invalid test split range: raw_start={raw_start}, raw_end={raw_end}, total_tokens={self.tmd_conf.total_tokens}."
            )

        self.test_split_start_token_idx = raw_start
        self.test_split_end_token_idx = raw_end
        self.dictionary = pUtil.get_dictionary(conf.generic.preparation_config_file)

    def build_untokenizer(self, input_samples_filepath, output_samples_filepath, output_metadata_filepath, output_invalid_tokens_filepath):
        """
        Build the matching untokenizer for this tokenized dataset.

        This is shared by generated samples and real test-token rows so both are
        decoded with the exact same tokenizer-format logic.
        """
        tokenization_format = str(self.tokenized_metadata.get('format', 'base_tokenizer'))
        tokenizer_class = str(self.tokenized_metadata.get('tokenizer_class', ''))
        
        selected_untokenizer = None
        if tokenizer_class == "EventPerSequenceWholeParticleTokenizer":
            selected_untokenizer = untokenizer.WholeParticleUntokenizer(
                self.dictionary, 
                input_samples_filepath, 
                output_samples_filepath, 
                output_metadata_filepath, 
                output_invalid_tokens_filepath,
                self.tokenized_metadata
            )
        elif tokenizer_class == "PackedEventStreamParticleFeatureTokenizer":
            selected_untokenizer = untokenizer.PackedEventStreamParticleFeatureUntokenizer(
                self.dictionary,
                input_samples_filepath, 
                output_samples_filepath, 
                output_metadata_filepath, 
                output_invalid_tokens_filepath,
                self.tokenized_metadata
            )
        else:
            selected_untokenizer = untokenizer.ParticleFeatureUntokenizer(
                self.dictionary, 
                input_samples_filepath, 
                output_samples_filepath, 
                output_metadata_filepath, 
                output_invalid_tokens_filepath,
                self.tokenized_metadata
            )

        if selected_untokenizer is None:
            raise ValueError(f"Could not determine untokenizer for tokenization format {tokenization_format} and tokenizer class {tokenizer_class}.")
        
        return selected_untokenizer

    def untokenize_generated_data(self):
        """
        Untokenize the generated samples from the selected sampling directory.

        Invalid generated rows are dropped by the current untokenizer and recorded
        in invalid_token_events.json with the invalid token and reason.
        """
        sample_untokenizer = self.build_untokenizer(
            self.generated_samples_filename,
            self.untokenized_samples_filename,
            self.untokenizing_metadata_filename,
            self.invalid_tokens_filename,
        )
        sample_untokenizer.untokenize_file()

    def generate_verbose_particle_information(self):
        # Output will be num_particles, pdgid, e, px, py, pz, pt, eta, theta, phi

        if not self.untokenized_samples_filename.exists():
            self.untokenize_generated_data()
        untokenized_data = data_manager.load_geant4_dataset(self.untokenized_samples_filename, pad_token=np.nan)

        NUM_FEATURES_PER_PARTICLE_VERBOSE = 9
        verbose_data = np.full(shape=(untokenized_data.shape[0], untokenized_data.shape[1], NUM_FEATURES_PER_PARTICLE_VERBOSE), fill_value=np.nan, dtype=np.float64)
        for idx_e, event in enumerate(untokenized_data):
            for idx_p, particle in enumerate(event):
                if particle[0] == np.nan:
                    verbose_data[idx_e, idx_p] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    continue

                pdgid, e, px, py, pz = particle
                r = np.sqrt(px * px + py * py + pz * pz)
                pt = np.sqrt(px * px + py * py)
                theta = np.arccos(pz / r) if r != 0 else 0
                phi = np.arctan2(py, px)
                eta = -np.log(np.tan(theta / 2))

                verbose_data[idx_e, idx_p] = [pdgid, e, px, py, pz, pt, eta, theta, phi]

        with open(self.verbose_particles_filename, 'w') as out_file:
            for event in verbose_data:
                for particle in event:
                    pdgid, e, px, py, pz, pt, eta, theta, phi = particle
                    if np.isnan(pdgid):
                        continue
                    out_file.write(f'{int(pdgid)} {e:.5f} {px:.5f} {py:.5f} {pz:.5f} {pt:.5f} {eta:.5f} {theta:.5f} {phi:.5f};')
                out_file.write('\n')

    def generate_real_test_data(self):
        """
        Materialize and untokenize the real test sequences matching the samples.

        The real comparison set comes from the same configured test_bin token
        range used by sample.py. It is written into the sampling directory so
        every sampling run has its own matching real-token and real-untokenized
        files.
        """
        total_starters = self.sampling_metadata.get('total_starters', None)
        if total_starters is None:
            with open(self.generated_samples_filename) as gen_samples_file:
                total_starters = sum(1 for event in gen_samples_file if event.strip() != '')
        num_test_sequences = min(int(total_starters), int(self.dls_conf.num_sequences))

        data = np.memmap(self.tmd_conf.tokenized_data_filepath, dtype=self.tmd_conf.dtype, mode='r')
        token_start = self.test_split_start_token_idx
        token_end = token_start + num_test_sequences * self.tmd_conf.sequence_length
        if token_end > self.test_split_end_token_idx:
            raise ValueError(
                f"Requested real test token_end={token_end}, but test split ends at {self.test_split_end_token_idx}."
            )

        tokenized_data = data[token_start:token_end].reshape(num_test_sequences, self.tmd_conf.sequence_length)
        with open(self.real_test_tokens_filename, 'w') as real_test_tokens_file:
            for event in tokenized_data:
                real_test_tokens_file.write(' '.join(str(int(token)) for token in event) + '\n')

        real_untokenizer = self.build_untokenizer(
            self.real_test_tokens_filename,
            self.real_test_untokenized_filename,
            self.real_test_untokenizing_metadata_filename,
            self.real_test_invalid_tokens_filename,
        )
        real_untokenizer.untokenize_file()

    def convert_to_verbose_particles(self, untokenized_data):
        """
        Convert raw-style particles into the verbose analysis layout.

        Input rows are expected to contain:
            pdgid, e, px, py, pz

        Output rows contain the columns expected by analysis_v2.plotting:
            pdgid, e, px, py, pz, pt, eta, theta, phi
        """
        NUM_FEATURES_PER_PARTICLE_VERBOSE = 9
        verbose_data = np.full(shape=(untokenized_data.shape[0], untokenized_data.shape[1], NUM_FEATURES_PER_PARTICLE_VERBOSE), fill_value=np.nan, dtype=np.float64)
        for idx_e, event in enumerate(untokenized_data):
            for idx_p, particle in enumerate(event):
                pdgid = particle[0]
                if np.isnan(pdgid):
                    continue

                pdgid, e, px, py, pz = particle
                r = np.sqrt(px * px + py * py + pz * pz)
                pt = np.sqrt(px * px + py * py)
                theta = np.arccos(np.clip(pz / r, -1.0, 1.0)) if r != 0 else 0
                phi = np.arctan2(py, px)
                eta = -np.log(np.tan(theta / 2)) if theta != 0 else np.inf

                verbose_data[idx_e, idx_p] = [pdgid, e, px, py, pz, pt, eta, theta, phi]
        
        return verbose_data

    def extract_energy_conservation_for_analysis(self, in_dataset):
        """
        Return |E_in - E_out| for raw-style or verbose particle arrays.

        This mirrors the old energy-conservation comparison without depending on
        analysis_v2.dataset.extract_ein_eout_for_analysis, which still resolves
        dictionaries through the old preparation-directory path.
        """
        e_in = in_dataset[:, 0, 1]
        e_out = np.nansum(in_dataset[:, 1:, 1], axis=1)
        return np.abs(e_in - e_out)

    def generate_distributions(self):
        """
        Generate distribution plots with the current analysis_v2.py API.

        analysis_v2.py exposes dataset-processing utilities and plotting
        methods directly. It no longer provides the old plotting() object with a
        load_data_by_model_names(...) step, so this method loads the real and
        generated files explicitly before plotting.
        """
        if not self.untokenized_samples_filename.exists():
            self.untokenize_generated_data()
        if not self.real_test_untokenized_filename.exists():
            self.generate_real_test_data()

        self.plotted_distributions_dir.mkdir(parents=True, exist_ok=True)

        real_data = data_manager.load_geant4_dataset(self.real_test_untokenized_filename, pad_token=np.nan)
        generated_data = data_manager.load_geant4_dataset(self.untokenized_samples_filename, pad_token=np.nan)
        real_verbose_data = self.convert_to_verbose_particles(real_data)
        generated_verbose_data = self.convert_to_verbose_particles(generated_data)
        model_legend_titles = ['Geant4', self.model_name]

        real_leading_pdgid = dataset.extract_single_column_for_analysis(real_verbose_data, 'pdgid', return_only_leading=True)
        generated_leading_pdgid = dataset.extract_single_column_for_analysis(generated_verbose_data, 'pdgid', return_only_leading=True)
        real_all_pdgid = dataset.extract_single_column_for_analysis(real_verbose_data, 'pdgid')
        generated_all_pdgid = dataset.extract_single_column_for_analysis(generated_verbose_data, 'pdgid')
        real_energy_conservation = self.extract_energy_conservation_for_analysis(real_verbose_data)
        generated_energy_conservation = self.extract_energy_conservation_for_analysis(generated_verbose_data)
        real_num_particles = dataset.extract_single_column_for_analysis(real_verbose_data, 'num_particles')
        generated_num_particles = dataset.extract_single_column_for_analysis(generated_verbose_data, 'num_particles')

        fig, _ = plotting.plot_dist_and_ratio_discrete_overlaid(
            column_name='pdgid',
            ref_vals=real_leading_pdgid,
            comp_vals_dict={self.model_name: generated_leading_pdgid},
            model_legend_titles=model_legend_titles,
            density=True,
            use_log=False,
            out_file=self.plotted_distributions_dir / 'distribution_leading_pdgid.png',
            show_output=False,
            sort_descending=True,
        )
        plt.close(fig)

        fig, _ = plotting.plot_dist_and_ratio_discrete_overlaid(
            column_name='pdgid',
            ref_vals=real_leading_pdgid,
            comp_vals_dict={self.model_name: generated_leading_pdgid},
            model_legend_titles=model_legend_titles,
            density=True,
            use_log=True,
            out_file=self.plotted_distributions_dir / 'distribution_leading_pdgid_log.png',
            show_output=False,
            sort_descending=True,
        )
        plt.close(fig)

        fig, _ = plotting.plot_dist_and_ratio_discrete_overlaid(
            column_name='pdgid',
            ref_vals=real_all_pdgid,
            comp_vals_dict={self.model_name: generated_all_pdgid},
            model_legend_titles=model_legend_titles,
            density=True,
            use_log=False,
            out_file=self.plotted_distributions_dir / 'distribution_all_pdgid.png',
            show_output=False,
            sort_descending=True,
        )
        plt.close(fig)

        fig, _ = plotting.plot_dist_and_ratio_discrete_overlaid(
            column_name='pdgid',
            ref_vals=real_all_pdgid,
            comp_vals_dict={self.model_name: generated_all_pdgid},
            model_legend_titles=model_legend_titles,
            density=True,
            use_log=True,
            out_file=self.plotted_distributions_dir / 'distribution_all_pdgid_log.png',
            show_output=False,
            sort_descending=True,
        )
        plt.close(fig)

        # fig, _ = analv2.plotting.plot_dist_and_ratio_cont(
        #     column_name='energy_conservation',
        #     ref_vals=real_energy_conservation,
        #     comp_vals_dict={self.model_name: generated_energy_conservation},
        #     model_legend_titles=model_legend_titles,
        #     density=True,
        #     use_log=True,
        #     out_file=self.plotted_distributions_dir / 'distribution_energy_conservation_log.png',
        #     show_output=False,
        # )
        # plt.close(fig)

        # fig, _ = analv2.plotting.plot_dist_and_ratio_cont(
        #     column_name='energy_conservation',
        #     ref_vals=real_energy_conservation,
        #     comp_vals_dict={self.model_name: generated_energy_conservation},
        #     model_legend_titles=model_legend_titles,
        #     density=True,
        #     use_log=False,
        #     out_file=self.plotted_distributions_dir / 'distribution_energy_conservation.png',
        #     show_output=False,
        # )
        # plt.close(fig)

        fig, _ = plotting.plot_dist_and_ratio_discrete_overlaid(
            column_name='num_particles',
            ref_vals=real_num_particles,
            comp_vals_dict={self.model_name: generated_num_particles},
            model_legend_titles=model_legend_titles,
            density=False,
            use_log=False,
            out_file=self.plotted_distributions_dir / 'distribution_num_particles.png',
            show_output=False,
        )
        plt.close(fig)

        for column_name in ['e', 'px', 'py', 'pz', 'pt', 'eta', 'theta', 'phi']:
            real_leading_values = dataset.extract_single_column_for_analysis(real_verbose_data, column_name, return_only_leading=True)
            generated_leading_values = dataset.extract_single_column_for_analysis(generated_verbose_data, column_name, return_only_leading=True)
            real_all_values = dataset.extract_single_column_for_analysis(real_verbose_data, column_name)
            generated_all_values = dataset.extract_single_column_for_analysis(generated_verbose_data, column_name)

            fig, _ = plotting.plot_dist_and_ratio_cont(
                column_name=column_name,
                ref_vals=real_leading_values,
                comp_vals_dict={self.model_name: generated_leading_values},
                model_legend_titles=model_legend_titles,
                out_file=self.plotted_distributions_dir / f'distribution_leading_{column_name}.png',
                show_output=False,
            )
            plt.close(fig)

            fig, _ = plotting.plot_dist_and_ratio_cont(
                column_name=column_name,
                ref_vals=real_leading_values,
                comp_vals_dict={self.model_name: generated_leading_values},
                model_legend_titles=model_legend_titles,
                use_log=True,
                out_file=self.plotted_distributions_dir / f'distribution_leading_log_{column_name}.png',
                show_output=False,
            )
            plt.close(fig)

            fig, _ = plotting.plot_dist_and_ratio_cont(
                column_name=column_name,
                ref_vals=real_all_values,
                comp_vals_dict={self.model_name: generated_all_values},
                model_legend_titles=model_legend_titles,
                out_file=self.plotted_distributions_dir / f'distribution_all_{column_name}.png',
                show_output=False,
            )
            plt.close(fig)

            fig, _ = plotting.plot_dist_and_ratio_cont(
                column_name=column_name,
                ref_vals=real_all_values,
                comp_vals_dict={self.model_name: generated_all_values},
                model_legend_titles=model_legend_titles,
                use_log=True,
                out_file=self.plotted_distributions_dir / f'distribution_all_log_{column_name}.png',
                show_output=False,
            )
            plt.close(fig)

    def get_real_jets(self):
        # =====================
        # Preparing real Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # =====================

        self.generate_real_test_data()
        untokenized_data = data_manager.load_geant4_dataset(self.real_test_untokenized_filename, pad_token=0.0)
        angular_real_data = data_manager.convert_data_4vector_to_features(untokenized_data, pad_token=0.0)

        accumulated_data = []
        for event in angular_real_data:
            single_jet = []
            for particle in event:
                pdgid, pt, eta, phi = particle
                features = [eta, phi, pt]
                single_jet.append(features)
            accumulated_data.append(single_jet)
        return np.array(accumulated_data, np.float64)

    def get_generated_jets(self):
        # =====================
        # Preparing generated Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # =====================

        if not self.untokenized_samples_filename.exists():
            self.untokenize_generated_data()

        # Since untokenized samples file uses the same format as Geant4
        untokenized_data = data_manager.load_geant4_dataset(self.untokenized_samples_filename, pad_token=0.0)
        angular_untokenized_data = data_manager.convert_data_4vector_to_features(untokenized_data, pad_token=0.0)

        accumulated_data = []
        for event in angular_untokenized_data:
            single_jet = []
            for particle in event:
                pdgid, pt, eta, phi = particle
                features = [eta, phi, pt]
                single_jet.append(features)
            accumulated_data.append(single_jet)
        return np.array(accumulated_data, np.float64)

    def calculate_metrics(self):
        real_jets = self.get_real_jets()
        generated_jets = self.get_generated_jets()

        # Make sure real and generated jets have the same num events in them.
        num_events_in_jets = min(len(real_jets), len(generated_jets))
        real_jets = real_jets[:num_events_in_jets]
        generated_jets = generated_jets[:num_events_in_jets]

        # =====================
        # Coverage and MMD
        # =====================

        cov, mmd = metrics.jetnet_eval_cov_mmd(real_jets, generated_jets)

        # =====================
        # FPD and KPD
        # =====================

        suggested_real_features = metrics.jetnet_get_suggested_kpd_fpd_features(real_jets)
        suggested_generated_features = metrics.jetnet_get_suggested_kpd_fpd_features(generated_jets)

        suggested_real_features = np.nan_to_num(suggested_real_features, nan=0.0)
        suggested_generated_features = np.nan_to_num(suggested_generated_features, nan=0.0)

        kpd_median, kpd_error = metrics.jetnet_eval_kpd(suggested_real_features, suggested_generated_features, num_threads=0)
        fpd_value, fpd_error = metrics.jetnet_eval_fpd(suggested_real_features, suggested_generated_features)

        # =====================
        # Wasserstein Distances
        # =====================

        # I don't know what this means so I am not using this one.
        # Wasserstein distances between Energy Flow Polynomials
        # w1_scores_avg_efp = analv2.metrics.jetnet_eval_w1efp(real_jets, generated_jets)

        # Wasserstein distance between masses of jets1 and jets2
        w1_mass_score = metrics.jetnet_eval_w1m(real_jets, generated_jets)
        # Wasserstein distances between particle features of jets1 and jets2
        w1_scores_avg_features = metrics.jetnet_eval_w1p(real_jets, generated_jets)

        w1m_score = w1_mass_score[0]
        w1m_score_std = w1_mass_score[1]

        w1p_avg_eta = w1_scores_avg_features[0][0]
        w1p_avg_phi = w1_scores_avg_features[0][1]
        w1p_avg_pt = w1_scores_avg_features[0][2]
        w1p_avg_eta_std = w1_scores_avg_features[1][0]
        w1p_avg_phi_std = w1_scores_avg_features[1][1]
        w1p_avg_pt_std = w1_scores_avg_features[1][2]

        metrics_results_dict = {
            "coverage": cov,
            "mmd": mmd,
            "kpd_median": kpd_median,
            "fpd_value": fpd_value,
            "w1m_score": w1m_score,
            "w1p_avg_eta": w1p_avg_eta,
            "w1p_avg_phi": w1p_avg_phi,
            "w1p_avg_pt": w1p_avg_pt,
            "kpd_error": kpd_error,
            "fpd_error": fpd_error,
            "w1m_score_std": w1m_score_std,
            "w1p_avg_eta_std": w1p_avg_eta_std,
            "w1p_avg_phi_std": w1p_avg_phi_std,
            "w1p_avg_pt_std": w1p_avg_pt_std,
        }

        with open(self.metrics_results_filename, "w") as opt_file:
            json.dump(metrics_results_dict, opt_file, indent=4)

if __name__ == "__main__":
    print(f"Generating distributions and metrics for config {getattr(conf.generic, 'preparation_config_file', 'unknown')}.")
    model_to_analyze = conf.generic.model_name
    
    parser = argparse.ArgumentParser(
        description="Handles the analysis of generated particle data. Assumes the model has been sampled already."
    )
    parser.add_argument("config_file", type=Path)
    parser.add_argument("--no-metrics", action="store_true", help="If provided, skip metric calculations.")
    parser.add_argument("--no-distributions", action="store_true", help="If provided, skip distribution generation.")
    parser.add_argument("--no-untokenize", action="store_true", help="If provided, skip untokenization of generated data. Assumes the data is already untokenized.")
    args = parser.parse_args()
    
    if args.no_metrics:
        print("Skipping metric calculations.")
    if args.no_distributions:
        print("Skipping distribution generation.")
    if args.no_untokenize:
        warnings.warn(
            "flag --no-untokenize is set. Will skip untokenize distribution."
            "This is fail horribly if the data is not already untokenized!",
            RuntimeWarning
        )
        print("Skipping untokenization of generated data.")

    print(f'Analyzing model {model_to_analyze}')

    sampling_dir = Analyzer.get_latest_sampling_dir(model_to_analyze)
    if not sampling_dir.exists():
        print(f'Analysis for model {model_to_analyze} cannot be performed, because no sampling data is available.')
        sys.exit()

    # Run the analysis
    dataset_analyzer = Analyzer(model_to_analyze)
    if not args.no_untokenize:
        dataset_analyzer.untokenize_generated_data()
    dataset_analyzer.generate_verbose_particle_information()
    if not args.no_distributions:
        dataset_analyzer.generate_distributions()
    if not args.no_metrics:
        dataset_analyzer.calculate_metrics()

    print('Analysis finished successfully.')
