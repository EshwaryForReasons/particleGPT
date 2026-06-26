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

class Analyzer:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.latest_sampling_dir = pUtil.get_latest_sampling_dir(self.model_name)

        if conf.generic.preparation_config_file is None:
            raise ValueError("preparation_config_file in configuration cannot be None!")
        self.preparation_config_filepath = paths.PROJECT_DIR / conf.generic.preparation_config_file
        if not self.preparation_config_filepath.exists():
            raise FileNotFoundError(f"Preparation config file does not exist: {self.preparation_config_filepath}")
        self.dls_conf = DataloaderSplitConfig(ESplitTypes.TEST, self.preparation_config_filepath)

        # The tokenized metadata stores the dictionary path relative to PROJECT_DIR.
        # Reuse the untokenizer's resolver so analysis and untokenization agree.
        self.real_test_tokens_filepath                  = self.latest_sampling_dir / 'real_test_tokens.csv'
        self.real_test_untokenized_filepath             = self.latest_sampling_dir / 'real_test_untokenized_samples.csv'
        self.real_test_untokenizing_metadata_filepath   = self.latest_sampling_dir / 'real_test_untokenizing_metadata.json'
        self.real_test_invalid_tokens_filepath          = self.latest_sampling_dir / 'real_test_invalid_token_events.json'
        self.generated_samples_filepath                 = self.latest_sampling_dir / untokenizer.DEFAULT_INPUT_CSV_NAME
        self.filtered_samples_filepath                  = self.latest_sampling_dir / 'filtered_samples.csv'
        self.sampled_leading_particles_filepath         = self.latest_sampling_dir / 'sampled_leading_particles.csv'
        self.verbose_particles_filepath                 = self.latest_sampling_dir / 'untokenized_samples_verbose.csv'
        self.untokenized_samples_filepath               = self.latest_sampling_dir / untokenizer.DEFAULT_OUTPUT_CSV_NAME
        self.untokenizing_metadata_filepath             = self.latest_sampling_dir / untokenizer.DEFAULT_METADATA_NAME
        self.invalid_tokens_filepath                    = self.latest_sampling_dir / untokenizer.DEFAULT_INVALID_TOKENS_NAME
        self.metrics_results_filepath                   = self.latest_sampling_dir / 'metrics_results.json'
        self.sampling_metadata_filepath                 = self.latest_sampling_dir / 'sampling_metadata.json'
        self.plotted_distributions_dir                  = self.latest_sampling_dir / 'plotted_distributions'

        if self.sampling_metadata_filepath.exists():
            try:
                with open(self.sampling_metadata_filepath, 'r') as f:
                    self.sampling_metadata = json.load(f)
            except Exception as exc:
                raise RuntimeError(f"Failure while trying to load json from {self.sampling_metadata_filepath}! Exception:\n{exc}") from exc

            final_csv_path = self.sampling_metadata.get('output_filepath', None)
            if final_csv_path is not None:
                final_csv_path = paths.PROJECT_DIR / Path(final_csv_path)
                if final_csv_path.exists():
                    self.generated_samples_filepath = final_csv_path
        else:
            self.sampling_metadata = {}

        if not self.generated_samples_filepath.exists():
            raise FileNotFoundError(f'generated_samples_filepath does not exist: {self.generated_samples_filepath}')

        raw_start, raw_end = self.dls_conf.get_raw_tokens_range()
        self.test_split_start_token_idx = raw_start
        self.test_split_end_token_idx = raw_end
        self.dictionary = pUtil.get_dictionary(conf.generic.preparation_config_file)


    def generate_verbose_particle_information(self):
        if not self.untokenized_samples_filepath.exists():
            raise FileNotFoundError(f'untokenized_samples_filepath does not exist: {self.untokenized_samples_filepath}')

        untokenized_data = data_manager.load_geant4_dataset(self.untokenized_samples_filepath, pad_token=np.nan)
        verbose_data = data_manager.convert_to_verbose_particles(untokenized_data)

        with open(self.verbose_particles_filepath, 'w') as out_file:
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
            with open(self.generated_samples_filepath) as gen_samples_file:
                total_starters = sum(1 for event in gen_samples_file if event.strip() != '')
        num_test_sequences = min(int(total_starters), int(self.dls_conf.num_sequences))

        data = np.memmap(self.dls_conf.tmd_conf.tokenized_data_filepath, dtype=self.dls_conf.tmd_conf.dtype, mode='r')
        token_start = self.test_split_start_token_idx
        token_end = token_start + num_test_sequences * self.dls_conf.tmd_conf.sequence_length
        if token_end > self.test_split_end_token_idx:
            raise ValueError(f"Requested real test token_end={token_end}, but test split ends at {self.test_split_end_token_idx}.")

        tokenized_data = data[token_start:token_end].reshape(num_test_sequences, self.dls_conf.tmd_conf.sequence_length)
        with open(self.real_test_tokens_filepath, 'w') as real_test_tokens_file:
            for event in tokenized_data:
                real_test_tokens_file.write(' '.join(str(int(token)) for token in event) + '\n')

        # Untokenize real test data
        tokenizer_class_str = self.dls_conf.tmd_conf.tokenizer_class
        selected_tokenizer_class = None
        match tokenizer_class_str:
            case "EventPerSequenceParticleFeatureTokenizer":
                selected_tokenizer_class = EventPerSequenceParticleFeatureTokenizer
            case "PackedEventStreamParticleFeatureTokenizer":
                selected_tokenizer_class = PackedEventStreamParticleFeatureTokenizer
        
        if selected_tokenizer_class is None:
            raise ValueError(f"Could not determine untokenizer for tokenizer class {tokenizer_class_str}.")
        
        temp_dir=paths.PROJECT_DIR / 'data' / 'tokenized' / self.dictionary.tokenization_name / 'temp'
        tokenizer = selected_tokenizer_class(dictionary=self.dictionary, temp_dir=None)
        tokenizer.decode_dataset(self.real_test_tokens_filepath)
        tokenizer.save_data(self.real_test_untokenized_filepath)

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
        if not self.untokenized_samples_filepath.exists():
            raise FileNotFoundError(f'untokenized_samples_filepath does not exist: {self.untokenized_samples_filepath}')
        if not self.real_test_untokenized_filepath.exists():
            # raise FileNotFoundError(f'real_test_untokenized_filepath does not exist: {self.real_test_untokenized_filepath}')
            self.generate_real_test_data()

        self.plotted_distributions_dir.mkdir(parents=True, exist_ok=True)

        real_data = data_manager.load_geant4_dataset(self.real_test_untokenized_filepath, pad_token=np.nan)
        generated_data = data_manager.load_geant4_dataset(self.untokenized_samples_filepath, pad_token=np.nan)
        real_verbose_data = data_manager.convert_to_verbose_particles(real_data)
        generated_verbose_data = data_manager.convert_to_verbose_particles(generated_data)
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
        untokenized_data = data_manager.load_geant4_dataset(self.real_test_untokenized_filepath, pad_token=0.0)
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

        if not self.untokenized_samples_filepath.exists():
            raise FileNotFoundError(f'untokenized_samples_filepath does not exist: {self.untokenized_samples_filepath}')

        # Since untokenized samples file uses the same format as Geant4
        untokenized_data = data_manager.load_geant4_dataset(self.untokenized_samples_filepath, pad_token=0.0)
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

        with open(self.metrics_results_filepath, "w") as opt_file:
            json.dump(metrics_results_dict, opt_file, indent=4)


def untokenize_generated_data(sampling_metadata_filepath: Path):
    # Load sampling metadata to get tokenized_metadata_file
    if not sampling_metadata_filepath.exists():
        raise FileNotFoundError(f"Sampling metadata file not found at {sampling_metadata_filepath}")
    with sampling_metadata_filepath.open('r') as f:
        sampling_metadata = json.load(f)
    tokenized_metadata_filepath = Path(sampling_metadata['tokenized_metadata_filepath'])
    generated_samples_filepath = Path(sampling_metadata['output_filepath'])
    
    if not generated_samples_filepath.exists():
        raise FileNotFoundError("Generated samples.csv needs to exist before they can be untokenized!")
    
    # Load tokenized metadata file to get the dictionary_file
    if not tokenized_metadata_filepath.exists():
        raise FileNotFoundError(f"Tokenized metadata file not found at {tokenized_metadata_filepath}")
    with tokenized_metadata_filepath.open('r') as f:
        tokenized_metadata = json.load(f)
    tmd_conf = TokenizedMetadataConfig(tokenized_metadata_filepath)
        
    dictionary_filepath = Path(tokenized_metadata["dictionary_filepath"])
    dictionary = Dictionary(dictionary_filepath)
    
    tokenizer_class_str =  tmd_conf.tokenizer_class
    selected_tokenizer_class = None
    match tokenizer_class_str:
        case "EventPerSequenceParticleFeatureTokenizer":
            selected_tokenizer_class = EventPerSequenceParticleFeatureTokenizer
        case "PackedEventStreamParticleFeatureTokenizer":
            selected_tokenizer_class = PackedEventStreamParticleFeatureTokenizer
    
    if selected_tokenizer_class is None:
        raise ValueError(f"Could not determine untokenizer for tokenizer class {tokenizer_class_str}.")
    
    untokenized_samples_filepath = sampling_metadata_filepath.parent / 'untokenized_samples.csv'
    temp_dir = paths.PROJECT_DIR / 'data' / 'tokenized' / dictionary.tokenization_name / 'temp'
    tokenizer = selected_tokenizer_class(dictionary=dictionary, temp_dir=None)
    tokenizer.decode_dataset(generated_samples_filepath)
    tokenizer.save_data(untokenized_samples_filepath)

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
    
    print(f'Analyzing model {conf.generic.model_name}')

    
    # Untokenize data
    if not args.no_untokenize:
        print("Untokenizing data")
        untokenize_generated_data(sampling_metadata_filepath)
        
    dataset_analyzer = Analyzer(conf.generic.model_name)
    
    # Generate verbose data
    print("Generating verbose particle information")
    dataset_analyzer.generate_verbose_particle_information()
    
    # Generate distributions
    if not args.no_distributions:
        print("Generating distributions")
        dataset_analyzer.generate_distributions()
        
    # Calculate metrics
    if not args.no_metrics:
        print("Calculating metrics")
        dataset_analyzer.calculate_metrics()

    print('Analysis finished successfully.')
