import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
from collections import Counter

import jetnet
from particle import Particle

import configurator as conf
from dictionary import Dictionary
import pUtil
import pTokenizerModule as pTokenizer
import data_manager
import analysis as anal

script_dir = Path(__file__).resolve().parent

class Analyzer:
    def __init__(self, model_name, preparation_name, scheme):
        self.preparation = preparation_name
        self.model_name = model_name
        
        self.latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)

        self.dictionary_filename                  = script_dir / 'data' / preparation_name / 'dictionary.json'
        self.meta_filename                        = script_dir / 'data' / preparation_name / 'meta.pkl'
        self.test_real_bin_filename               = script_dir / 'data' / preparation_name / 'test_real.bin'
        self.real_leading_test_particles_filename = script_dir / 'data' / preparation_name / 'real_leading_test_particles.csv'
        self.generated_samples_filename           = self.latest_sampling_dir / 'generated_samples.csv'
        self.filtered_samples_filename            = self.latest_sampling_dir / 'filtered_samples.csv'
        self.sampled_leading_particles_filename   = self.latest_sampling_dir / 'sampled_leading_particles.csv'
        self.untokenized_samples_filename         = self.latest_sampling_dir / 'untokenized_samples.csv'
        self.metrics_results_filename             = self.latest_sampling_dir / 'metrics_results.json'
        
        if not self.meta_filename.exists():
            print("Data has not been prepared! Please prepare data first!")
            exit()
            
        with open(self.meta_filename, 'rb') as f:
            meta = pickle.load(f)
            self.num_particles_per_event = meta['num_particles_per_event']
            
        self.dictionary = Dictionary(self.dictionary_filename.as_posix())

        def get_bin_count(type_str):
            step_size = self.dictionary.token_step_size(type_str)
            if step_size == 0:
                return 0
            return int(self.dictionary.token_range(type_str) // step_size)
        # Convenience dictionary definitions
        p_bin_count = int(self.dictionary.token_range('e') // 1000)
        e_bin_count = get_bin_count('e')
        eta_bin_count = get_bin_count('eta')
        theta_bin_count = get_bin_count('theta')
        phi_bin_count = get_bin_count('phi')
        pt_bin_count = get_bin_count('pt')

        self.columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
        self.bin_settings = {
            "num_particles": { "min": 0,                                  "max": 50,                                 "bins": 50 },
            "e":             { "min": self.dictionary.token_min('e'),     "max": self.dictionary.token_max('e'),     "bins": e_bin_count },
            "px":            { "min": self.dictionary.token_min('e'),     "max": self.dictionary.token_max('e'),     "bins": p_bin_count },
            "py":            { "min": self.dictionary.token_min('e'),     "max": self.dictionary.token_max('e'),     "bins": p_bin_count },
            "pz":            { "min": self.dictionary.token_min('e'),     "max": self.dictionary.token_max('e'),     "bins": p_bin_count },
            "eta":           { "min": self.dictionary.token_min('eta'),   "max": self.dictionary.token_max('eta'),   "bins": eta_bin_count },
            "theta":         { "min": self.dictionary.token_min('theta'), "max": self.dictionary.token_max('theta'), "bins": theta_bin_count },
            "phi":           { "min": self.dictionary.token_min('phi'),   "max": self.dictionary.token_max('phi'),   "bins": phi_bin_count },
            "pt":            { "min": self.dictionary.token_min('pt'),    "max": self.dictionary.token_max('pt'),    "bins": pt_bin_count },
        }
    
    def generate_leading_particle_information(self):
        # Output will be num_particles, pdgid, e, px, py, pz, pt, eta, theta, phi

        untokenized_samples_data = []
        with open(self.untokenized_samples_filename, 'r') as in_file:
            for event in in_file:
                particles = event.strip().split(';')
                event_arr = []
                for particle in particles:
                    particle = [float(x) for x in particle.strip().split()]
                    event_arr.extend(particle)
                untokenized_samples_data.append(event_arr)
        
        with open(self.sampled_leading_particles_filename, 'w') as out_file:
            for event in untokenized_samples_data:
                if len(event) == 0:
                    continue
                event = np.array(event)
                num_particles = len(event) // 5
                particles = event.reshape((num_particles, 5))
                secondaries = particles[1:]
                # Find index of particle with the highest energy
                leading_particle_idx = np.argmax(secondaries[:, 1])
                leading_particle = secondaries[leading_particle_idx]
                secondaries = [s for s in secondaries if s[0] != 0]
                
                pdgid = leading_particle[0]
                e = leading_particle[1]
                px = leading_particle[2]
                py = leading_particle[3]
                pz = leading_particle[4]
                
                r = np.sqrt(px * px + py * py + pz * pz)
                pt = np.sqrt(px * px + py * py)
                theta = np.arccos(pz / r)
                phi = np.arctan2(py, px)
                eta = -np.log(np.tan(theta / 2))
                
                out_file.write(f'{len(secondaries)} {int(pdgid)} {e} {px} {py} {pz} {pt} {eta:.5f} {theta:.5f} {phi:.5f}\n')
    
    def filter_data(self):
        # Load data
        tokenized_data = []
        with open(self.generated_samples_filename) as gen_samples_file:
            for event in gen_samples_file:
                event = [int(x) for x in event.strip().split()]
                tokenized_data.append(event)
                
        filtered_data = []
        
        # Ensure valid borders
        tokenized_data = [e for e in tokenized_data if e[0] == 1 and e[-1] == 2]
        
        # Remove special tokens
        tokenized_data = [[x for x in e if x not in [0, 1, 2, 3, 4]] for e in tokenized_data]
            
        # Ensure events are well formed
        tokenized_data = [e for e in tokenized_data if len(e) > 5 and len(e) % 5 == 0]
        
        # Ensure valid token ranges
        pdgid_offset_min = self.dictionary.PDGID_OFFSET
        pdgid_offset_max = self.dictionary.PDGID_OFFSET + len(self.dictionary.pdgids)
        energy_offset_min = self.dictionary.ENERGY_OFFSET
        energy_offset_max = self.dictionary.ENERGY_OFFSET + len(self.dictionary.e_bins)
        eta_offset_min = self.dictionary.ETA_OFFSET
        eta_offset_max = self.dictionary.ETA_OFFSET + len(self.dictionary.eta_bins)
        theta_offset_min = self.dictionary.THETA_OFFSET
        theta_offset_max = self.dictionary.THETA_OFFSET + len(self.dictionary.theta_bins)
        phi_offset_min = self.dictionary.PHI_OFFSET
        phi_offset_max = self.dictionary.PHI_OFFSET + len(self.dictionary.phi_bins)
        
        for event in tokenized_data:
            b_keep_event = True
            for i, token in enumerate(event):
                token_type_id = i % 5
                if token_type_id == 0:
                    if token < pdgid_offset_min or token >= pdgid_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 1:
                    if token < energy_offset_min or token >= energy_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 2:
                    if token < eta_offset_min or token >= eta_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 3:
                    if token < theta_offset_min or token >= theta_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 4:
                    if token < phi_offset_min or token >= phi_offset_max:
                        b_keep_event = False
                        break
            
            if b_keep_event:
                filtered_data.append(event)
        
        # Output data
        with open(self.filtered_samples_filename, 'w') as filtered_file:
            for event in filtered_data:
                event = [str(x) for x in event]
                event = ' '.join(event)
                filtered_file.write(event + '\n')
    
    def generate_distributions(self):
        # if self.dictionary.scheme == 'standard':
        #     self.filter_data()
        # elif self.dictionary.scheme == 'no_eta':
        #     raise RuntimeError("Scheme no_eta is no longer supported!")
        # elif self.dictionary.scheme == 'no_particle_boundaries':
        #     self.filter_data()
        # elif self.dictionary.scheme == 'paddingv2':
        #     self.filter_data()
        
        # pTokenizer.untokenize_data(self.dictionary_filename.as_posix(), self.dictionary.scheme, self.filtered_samples_filename.as_posix(), self.untokenized_samples_filename.as_posix())
        # self.generate_leading_particle_information()

        real_df = pd.read_csv(self.real_leading_test_particles_filename, sep=" ", names=self.columns, engine="c", header=None)
        sampled_df = pd.read_csv(self.sampled_leading_particles_filename, sep=" ", names=self.columns, engine="c", header=None)
        
        # PDGID is a fundamentally different plot, so we do it first here.
        read_pdgids = real_df['pdgid'].astype(str)
        sampled_pdgids = sampled_df['pdgid'].astype(str)
        real_freq = Counter(read_pdgids)
        sampled_freq = Counter(sampled_pdgids)
        
        out_file = self.latest_sampling_dir / f'histogram_pdgid.png'
        anal.plotting.plot_discrete_distribution([real_freq, sampled_freq], ['Input', 'Sampled'], name="Particle IDs", out_file=out_file)

        for column, settings in self.bin_settings.items():
            if settings["bins"] == 0:
                continue
            real_data = real_df[column].to_list()
            sampled_data = sampled_df[column].to_list()
            out_file = self.latest_sampling_dir / f'histogram_{column}.png'
            anal.plotting.plot_continuous_distribution([real_data, sampled_data], ['Input', 'Sampled'], name=column, min=settings["min"], max=settings["max"], n_bins=settings["bins"], normalized=True, out_file=out_file)
        

    def get_real_jets(self):
        # -------------------------------------------------------------------------------
        # Preparing real Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # -------------------------------------------------------------------------------
        
        test_real_events = np.memmap(self.test_real_bin_filename, dtype=np.float64, mode='r')
        test_real_events = test_real_events.reshape(-1, self.num_particles_per_event, 5)
        angular_real_data = data_manager.convert_data_4vector_to_angular(test_real_events, pad_token=0.0)
                
        accumulated_data = []
        for event in angular_real_data:
            single_jet = []
            for particle in event:
                pdgid, e, eta, theta, phi = particle
                part = Particle.from_pdgid(pdgid)
                pt = np.sqrt(e * e + part.mass * part.mass) / np.cosh(eta)
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
        angular_untokenized_data = data_manager.convert_data_4vector_to_angular(untokenized_data, pad_token=0.0)
        
        accumulated_data = []
        for event in angular_untokenized_data:
            single_jet = []
            for particle in event:
                pdgid, e, eta, theta, phi = particle
                part = Particle.from_pdgid(pdgid)
                pt = np.sqrt(e * e + part.mass * part.mass) / np.cosh(eta)
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

        cov, mmd = jetnet.evaluation.cov_mmd(real_jets, generated_jets)

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
            "kpd_error": kpd_error,
            "fpd_value": fpd_value,
            "fpd_error": fpd_error,
            "w1m_score": w1m_score,
            "w1m_score_std": w1m_score_std,
            "w1p_avg_eta": w1p_avg_eta,
            "w1p_avg_phi": w1p_avg_phi,
            "w1p_avg_pt": w1p_avg_pt,
            "w1p_avg_eta_std": w1p_avg_eta_std,
            "w1p_avg_phi_std": w1p_avg_phi_std,
            "w1p_avg_pt_std": w1p_avg_pt_std,
        }

        with open(self.metrics_results_filename, "w") as opt_file:
            json.dump(metrics_results_dict, opt_file, indent=4)

def analyze_dataset_worker(model_name):
    sampling_dir = pUtil.get_latest_sampling_dir(model_name)
    if not sampling_dir.exists():
        return
    
    print(f'Analyzing model {model_name}')

    # Extract dataset name from sampling info
    preparation_name = pUtil.get_model_preparation_name(model_name)
    config_filename = pUtil.get_model_config_filename(model_name)
    
    with open(config_filename, 'r') as config_file:
        config = json.load(config_file)
        scheme = config['training_config'].get('scheme', 'standard')

    # Run the analysis
    dataset_analyzer = Analyzer(model_name, preparation_name, scheme)
    dataset_analyzer.generate_distributions()
    dataset_analyzer.calculate_metrics()

def analyze_dataset():
    # If argument 'all' is provided, generate distributions and metrics for all sampled datasets' latest sampling.
    # JetNet does not enjoy multi-threading (it already uses it internally to speed up calculations).
    if 'all' in sys.argv:
        print('Generating distributions and metrics for all datasets.')
        all_model_names = pUtil.get_all_model_names()
        for model_name in all_model_names:
            analyze_dataset_worker(model_name)
        print('Distributions and metrics generated successfully for all datasets.')
    else:
        print(f'Generating distributions and metrics for dataset {conf.generic.preparation_name}.')
        analyze_dataset_worker(conf.generic.model_name)
        print('Distributions and metrics generated successfully.')

if __name__ == "__main__":
    analyze_dataset()