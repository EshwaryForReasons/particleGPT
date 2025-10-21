import sys
import json
import pickle
import numpy as np
from pathlib import Path
import time

import configurator as conf
from dictionary import Dictionary
from dictionary import ETokenTypes
import pUtil
import data_manager
import analysis as anal
import untokenizer
import wandb_doer

script_dir = Path(__file__).resolve().parent

class Analyzer:
    def __init__(self, model_name):
        self.model_name = model_name
        
        preparation_dir = pUtil.get_model_preparation_dir(model_name)
        self.latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)

        self.dictionary_filename                  = preparation_dir / 'dictionary.json'
        self.meta_filename                        = preparation_dir / 'meta.pkl'
        self.test_real_bin_filename               = preparation_dir / 'test_real.bin'
        self.real_leading_test_particles_filename = preparation_dir / 'real_leading_test_particles.csv'
        self.generated_samples_filename           = self.latest_sampling_dir / 'generated_samples.csv'
        self.filtered_samples_filename            = self.latest_sampling_dir / 'filtered_samples.csv'
        self.sampled_leading_particles_filename   = self.latest_sampling_dir / 'sampled_leading_particles.csv'
        self.verbose_particles_filename           = self.latest_sampling_dir / 'untokenized_samples_verbose.csv'
        self.untokenized_samples_filename         = self.latest_sampling_dir / 'untokenized_samples.csv'
        self.metrics_results_filename             = self.latest_sampling_dir / 'metrics_results.json'
        
        if not self.meta_filename.exists():
            print("Data has not been prepared! Please prepare data first!")
            exit()
            
        with open(self.meta_filename, 'rb') as f:
            meta = pickle.load(f)
            self.num_particles_per_event = meta['num_particles_per_event']
            
        self.dictionary = Dictionary(self.dictionary_filename)
        
    def generate_verbose_particle_information(self):
        # Output will be num_particles, pdgid, e, px, py, pz, pt, eta, theta, phi

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
        
    def filter_data(self):
        num_features_per_particle = 4
        
        # Load data
        tokenized_data = []
        with open(self.generated_samples_filename) as gen_samples_file:
            for event in gen_samples_file:
                event = [int(x) for x in event.strip().split()]
                tokenized_data.append(event)
        
        # Ensure valid borders
        tokenized_data = [e for e in tokenized_data if e[0] == self.dictionary.event_start_token and e[-1] == self.dictionary.event_end_token]
        
        # Remove special tokens
        tokenized_data = [[x for x in e if x not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] for e in tokenized_data]
            
        # Ensure events are well formed (at least num_features_per_particle features per particle and an number of particles)
        tokenized_data = [e for e in tokenized_data if len(e) > num_features_per_particle and len(e) % num_features_per_particle == 0]
        
        # Util function; works on non-uniform 2D lists.
        def convert_2D_list_to_array(lst, pad_token=np.nan, dtype=np.uint32):
            max_len = max(len(sub) for sub in lst)
            padded = np.full((len(lst), max_len), pad_token, dtype=dtype)
            for i, sub in enumerate(lst):
                padded[i, :len(sub)] = sub
            return padded
        
        def vectorized_token_range_filtration():
            nonlocal tokenized_data
            tokenized_data = convert_2D_list_to_array(tokenized_data, pad_token=0, dtype=np.uint16)
            
            # Precompute (token, token type) mapping.
            token_type_lookup = np.empty(self.dictionary.vocab_size, dtype=np.uint8)
            for token in range(self.dictionary.vocab_size):
                token_type_lookup[token] = self.dictionary.get_token_type(token).value
            
            # Convert token IDs to token types via lookup table.
            token_types = token_type_lookup[tokenized_data]

            # Build expected pattern based on token position in particle (pdgid, pt, eta, phi).
            expected_pattern_arr = []
            for type_str in self.dictionary.tokenization_schema:
                if type_str == 'pdgid':
                    expected_pattern_arr.append(ETokenTypes.PDGID.value)
                elif type_str == 'pt':
                    expected_pattern_arr.append(ETokenTypes.PT.value)
                elif type_str == 'eta':
                    expected_pattern_arr.append(ETokenTypes.ETA.value)
                elif type_str == 'theta':
                    expected_pattern_arr.append(ETokenTypes.THETA.value)
                elif type_str == 'phi':
                    expected_pattern_arr.append(ETokenTypes.PHI.value)
                elif type_str == 'energy':
                    expected_pattern_arr.append(ETokenTypes.ENERGY.value)
                elif type_str == 'px':
                    expected_pattern_arr.append(ETokenTypes.PX.value)
                elif type_str == 'py':
                    expected_pattern_arr.append(ETokenTypes.PY.value)
                elif type_str == 'pz':
                    expected_pattern_arr.append(ETokenTypes.PZ.value)

            expected_pattern = np.array(expected_pattern_arr, dtype=token_types.dtype)
            expected_token_types = np.tile(expected_pattern, tokenized_data.shape[1] // num_features_per_particle)[:tokenized_data.shape[1]]
            # Ensure padding is not counted among the expected token types.
            padding_mask = tokenized_data == self.dictionary.padding_token
            expected_token_types = np.where(~padding_mask, expected_token_types, ETokenTypes.PADDING.value).astype(int)
            
            # Filter rows where all tokens match the expected pattern.
            valid_mask = np.all(token_types == expected_token_types, axis=1)
            filtered_data = tokenized_data[valid_mask]

            return filtered_data
        
        tokenized_data = vectorized_token_range_filtration()
        
        # Output data
        with open(self.filtered_samples_filename, 'w') as filtered_file:
            for event in tokenized_data:
                event = [str(x) for x in event if x != self.dictionary.padding_token]
                event = ' '.join(event)
                filtered_file.write(event + '\n')
     
    def generate_distributions(self):
        plotter = anal.plotting()
        plotter.load_data_by_model_names([self.model_name])
        
        # The full training run probably isn't too useful, but we plot it anyway.
        # plotter.plot_pdgid_distribution_leading([self.model_name], normalized=True, use_log=False, out_file=self.latest_sampling_dir / 'distribution_leading_pdgid.png')
        # plotter.plot_pdgid_distribution_leading([self.model_name], normalized=True, use_log=True, out_file=self.latest_sampling_dir / 'distribution_leading_pdgid_log.png')
        # plotter.plot_pdgid_distribution_all([self.model_name], normalized=True, use_log=False, out_file=self.latest_sampling_dir / 'distribution_all_pdgid.png')
        # plotter.plot_pdgid_distribution_all([self.model_name], normalized=True, use_log=True, out_file=self.latest_sampling_dir / 'distribution_all_pdgid_log.png')
        # plotter.plot_energy_conservation([self.model_name], normalized=True, use_log=True, out_file=self.latest_sampling_dir / 'distribution_energy_conservation_log.png')
        # plotter.plot_energy_conservation([self.model_name], normalized=True, use_log=False, out_file=self.latest_sampling_dir / 'distribution_energy_conservation.png')
        plotter.plot_num_particles([self.model_name], normalized=False, use_log=False, out_file=self.latest_sampling_dir / 'distribution_num_particles.png')
        # for column_name in ['e', 'px', 'py', 'pz', 'pt', 'eta', 'theta', 'phi']:
        #     plotter.plot_distribution_leading([self.model_name], column_name, out_file=self.latest_sampling_dir / f'distribution_leading_{column_name}.png')
        #     plotter.plot_distribution_leading([self.model_name], column_name, out_file=self.latest_sampling_dir / f'distribution_leading_log_{column_name}.png', use_log=True)
        #     plotter.plot_distribution_all([self.model_name], column_name, out_file=self.latest_sampling_dir / f'distribution_all_{column_name}.png')
        #     plotter.plot_distribution_all([self.model_name], column_name, out_file=self.latest_sampling_dir / f'distribution_all_log_{column_name}.png', use_log=True)

    def get_real_jets(self):
        # -------------------------------------------------------------------------------
        # Preparing real Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # -------------------------------------------------------------------------------
        
        test_real_events = np.memmap(self.test_real_bin_filename, dtype=np.float64, mode='r')
        test_real_events = test_real_events.reshape(-1, self.num_particles_per_event, 5)
        angular_real_data = data_manager.convert_data_4vector_to_features(test_real_events, pad_token=0.0)

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
        # -------------------------------------------------------------------------------
        # Preparing generated Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # -------------------------------------------------------------------------------
        
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
        
        # -------------------------------------------------------------------------------
        # Coverage and MMD
        # -------------------------------------------------------------------------------

        cov, mmd = anal.metrics.jetnet_eval_cov_mmd(real_jets, generated_jets)

        # -------------------------------------------------------------------------------
        # FPD and KPD
        # -------------------------------------------------------------------------------

        suggested_real_features = anal.metrics.jetnet_get_suggested_kpd_fpd_features(real_jets)
        suggested_generated_features = anal.metrics.jetnet_get_suggested_kpd_fpd_features(generated_jets)

        suggested_real_features = np.nan_to_num(suggested_real_features, nan=0.0)
        suggested_generated_features = np.nan_to_num(suggested_generated_features, nan=0.0)

        kpd_median, kpd_error = anal.metrics.jetnet_eval_kpd(suggested_real_features, suggested_generated_features, num_threads=0)
        fpd_value, fpd_error = anal.metrics.jetnet_eval_fpd(suggested_real_features, suggested_generated_features)

        # -------------------------------------------------------------------------------
        # Wasserstein Distances 
        # -------------------------------------------------------------------------------

        # I don't know what this means so I am not using this one.
        # Wasserstein distances between Energy Flow Polynomials
        # w1_scores_avg_efp = anal.metrics.jetnet_eval_w1efp(real_jets, generated_jets)

        # Wasserstein distance between masses of jets1 and jets2
        w1_mass_score = anal.metrics.jetnet_eval_w1m(real_jets, generated_jets)
        # Wasserstein distances between particle features of jets1 and jets2
        w1_scores_avg_features = anal.metrics.jetnet_eval_w1p(real_jets, generated_jets)

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
    print(f'Generating distributions and metrics for dataset {conf.generic.preparation_name}.')
    model_to_analyze = conf.generic.model_name
        
    print(f'Analyzing model {model_to_analyze}')
    
    sampling_dir = pUtil.get_latest_sampling_dir(model_to_analyze)
    if not sampling_dir.exists():
        print(f'Analysis for model {model_to_analyze} cannot be performed, because no sampling data is available.')
        sys.exit()
    
    # Run the analysis
    dataset_analyzer = Analyzer(model_to_analyze)
    # dataset_analyzer.filter_data()
    # untokenizer.untokenize_data(dataset_analyzer.filtered_samples_filename, dataset_analyzer.untokenized_samples_filename)
    # dataset_analyzer.generate_verbose_particle_information()
    dataset_analyzer.generate_distributions()
    # dataset_analyzer.calculate_metrics()
    # wandb_doer.send_to_wandb(model_to_analyze)
    
    print('Distributions and metrics generated successfully.')