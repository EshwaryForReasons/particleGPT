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
import matplotlib.pyplot as plt
import warnings

import paths
from paths import path_constants as pc
import pUtil
import data_manager
from analysis.dataset import dataset
from analysis.metrics import metrics
from analysis.tables import tables
from analysis.plotting import plotting_v2 as plotting
from particleGPT.dictionary import Dictionary
from particleGPT.preparation import ESplitTypes, DataloaderSplitConfig, TokenizedMetadataConfig
# import particleGPT.untokenizer as untokenizer
import particleGPT.configurator as conf
from particleGPT.tokenizers import (
    EventPerSequenceParticleFeatureTokenizer,
    # EventPerSequenceWholeParticleTokenizer,
    PackedEventStreamParticleFeatureTokenizer,
)
import particleGPT.quickmaths as quickmaths

class Analyzer:
    
    def __init__(self, metadata_filepath: Path):
        self.sampling_metadata_filepath = metadata_filepath
        if not self.sampling_metadata_filepath.exists():
            raise FileNotFoundError(f"Sampling metadata file not found at {self.sampling_metadata_filepath}")
        self.latest_sampling_dir = self.sampling_metadata_filepath.parent
        
        with self.sampling_metadata_filepath.open('r') as f:
            self.sampling_metadata = json.load(f)
        config_filepath = Path(self.sampling_metadata['config_filepath'])
        with config_filepath.open('r') as f:
            config_data = json.load(f)
        preparation_conf_filepath = Path(config_data['preparation_config_file'])
        self.dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, preparation_conf_filepath)
        
        self.dictionary = Dictionary(self.dls_conf.tmd_conf.dictionary_filepath)
        
        # @TODO: this is a horrible temporary solution to make the other libraries work for now
        conf.main(config_filepath)
        
        self.model_name = self.sampling_metadata["model_name"]
        
        raw_start, raw_end = self.dls_conf.get_raw_tokens_range()
        self.test_split_start_token_idx = raw_start
        self.test_split_end_token_idx = raw_end

        self.samples_decoded_filepath                   = self.latest_sampling_dir / paths.as_bin(pc.samples_decoded_filename)
        self.real_test_decoded_filepath                 = self.latest_sampling_dir / paths.as_bin(pc.real_test_decoded_filepath)
        self.real_test_decoded_invalid_events_filepath  = self.latest_sampling_dir / paths.as_json(pc.real_test_decoded_invalid_events_filepath)
        self.plotted_distributions_dir                  = self.latest_sampling_dir / 'plotted_distributions'

    def generate_real_test_data(self):
        """
        Materialize and untokenize the real test sequences matching the samples.

        The real comparison set comes from the same configured test_bin token
        range used by sample.py. It is written into the sampling directory so
        every sampling run has its own matching real-token and real-untokenized
        files.
        """
        
        num_sample_sequences = self.sampling_metadata['num_sample_sequences']
        if num_sample_sequences > self.dls_conf.num_sequences:
            raise ValueError("More sequences were sampled than the test split can provide! Please double check this.")

        token_start = self.test_split_start_token_idx
        token_end = token_start + num_sample_sequences * self.dls_conf.tmd_conf.sequence_length
        if token_end > self.test_split_end_token_idx:
            raise ValueError(f"Requested real test token_end={token_end}, but test split ends at {self.test_split_end_token_idx}.")
        tokenized_data = np.memmap(self.dls_conf.tmd_conf.tokenized_data_filepath, dtype=self.dls_conf.tmd_conf.dtype, mode='r')
        tokenized_data = tokenized_data[token_start:token_end]

        # Untokenize real test data
        tokenizer = self.dls_conf.tmd_conf.tokenizer_class(dictionary=self.dictionary, temp_dir=None)
        tokenizer.decode_dataset(tokenized_data)
        tokenizer.save_data(self.real_test_decoded_filepath, self.real_test_decoded_invalid_events_filepath, skip_write_metadata=True)

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
        if not self.samples_decoded_filepath.exists():
            raise FileNotFoundError(f'untokenized_samples_filepath does not exist: {self.samples_decoded_filepath}')
        # if not self.real_test_decoded_filepath.exists():
        self.generate_real_test_data()

        self.plotted_distributions_dir.mkdir(parents=True, exist_ok=True)

        real_data = data_manager.load_geant4_dataset(self.real_test_decoded_filepath, pad_token=np.nan)
        generated_data = data_manager.load_geant4_dataset(self.samples_decoded_filepath, pad_token=np.nan)
        real_verbose_data = quickmaths.convert_to_verbose_particles(real_data)
        generated_verbose_data = quickmaths.convert_to_verbose_particles(generated_data)
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

        fig, _ = plotting.plot_dist_and_ratio_cont(
            column_name='energy_conservation',
            ref_vals=real_energy_conservation,
            comp_vals_dict={self.model_name: generated_energy_conservation},
            model_legend_titles=model_legend_titles,
            density=True,
            use_log=True,
            out_file=self.plotted_distributions_dir / 'distribution_energy_conservation_log.png',
            show_output=False,
        )
        plt.close(fig)

        fig, _ = plotting.plot_dist_and_ratio_cont(
            column_name='energy_conservation',
            ref_vals=real_energy_conservation,
            comp_vals_dict={self.model_name: generated_energy_conservation},
            model_legend_titles=model_legend_titles,
            density=True,
            use_log=False,
            out_file=self.plotted_distributions_dir / 'distribution_energy_conservation.png',
            show_output=False,
        )
        plt.close(fig)

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
        untokenized_data = data_manager.load_geant4_dataset(self.real_test_decoded_filepath, pad_token=0.0)
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

        if not self.samples_decoded_filepath.exists():
            raise FileNotFoundError(f'untokenized_samples_filepath does not exist: {self.samples_decoded_filepath}')

        # Since untokenized samples file uses the same format as Geant4
        untokenized_data = data_manager.load_geant4_dataset(self.samples_decoded_filepath, pad_token=0.0)
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

        with open(self.latest_sampling_dir / paths.as_json(pc.samples_metrics_results_filename), "w") as opt_file:
            json.dump(metrics_results_dict, opt_file, indent=4)


def untokenize_generated_data(sampling_metadata_filepath: Path):
    # Load sampling metadata to get tokenized_metadata_file
    with sampling_metadata_filepath.open('r') as f:
        sampling_metadata = json.load(f)
    
    samples_filename = Path(sampling_metadata['output_filepath'])
    if not samples_filename.exists():
        raise FileNotFoundError("Generated samples.csv needs to exist before they can be untokenized!")
    
    tokenized_metadata_filepath = Path(sampling_metadata['tokenized_metadata_filepath'])
    tmd_conf = TokenizedMetadataConfig(tokenized_metadata_filepath)
    dictionary = Dictionary(tmd_conf.dictionary_filepath)
    
    samples_decoded_filepath = sampling_metadata_filepath.parent / paths.as_csv(pc.samples_decoded_filename)
    tokenizer = tmd_conf.tokenizer_class(dictionary=dictionary, temp_dir=None)
    tokenizer.decode_dataset_from_file(samples_filename)
    tokenizer.save_data(samples_decoded_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handles the analysis of generated particle data. Assumes the model has been sampled already.")
    parser.add_argument("metadata_file", type=Path)
    parser.add_argument("--no-metrics", action="store_true", help="If provided, skip metric calculations.")
    parser.add_argument("--no-distributions", action="store_true", help="If provided, skip distribution generation.")
    parser.add_argument("--no-untokenize", action="store_true", help="If provided, skip untokenization of generated data. Assumes the data is already untokenized.")
    args = parser.parse_args()
    
    sampling_metadata_filepath = Path(args.metadata_file)
    if not sampling_metadata_filepath.exists():
        raise FileNotFoundError(f"Samples metadata file not found at {sampling_metadata_filepath}. The metadata file is required to perform analysis.")
    
    if args.no_untokenize:
        warnings.warn(
            "flag --no-untokenize is set. Will skip untokenize distribution."
            "This is fail horribly if the data is not already untokenized!",
            RuntimeWarning
        )
        print("Skipping untokenization of generated data.")
    if args.no_distributions:
        print("Skipping distribution generation.")
    if args.no_metrics:
        print("Skipping metric calculations.")
    
    print(f'Analyzing sampled data at {sampling_metadata_filepath.parent}')

    # Untokenize data
    if not args.no_untokenize:
        print("Untokenizing data")
        untokenize_generated_data(sampling_metadata_filepath)
        
    dataset_analyzer = Analyzer(sampling_metadata_filepath)
    
    # Generate distributions
    if not args.no_distributions:
        print("Generating distributions")
        dataset_analyzer.generate_distributions()
        
    # Calculate metrics
    if not args.no_metrics:
        print("Calculating metrics")
        dataset_analyzer.calculate_metrics()

    print('Analysis finished successfully.')
