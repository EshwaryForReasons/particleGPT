import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path

# import jetnet

import configurator
from dictionary import Dictionary
import pUtil
import pTokenizerModule as pTokenizer

script_dir = Path(__file__).resolve().parent
dictionary = Dictionary(script_dir / 'data' / configurator.preparation_name / 'dictionary.json')

class Analyzer:
    def __init__(self, preparation, model_name, scheme):
        self.preparation = preparation
        self.model_name = model_name
        self.scheme = scheme
        
        self.latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)

        self.dictionary_filename                  = script_dir / 'data' / preparation / 'dictionary.json'
        self.meta_filename                        = script_dir / 'data' / preparation / 'meta.pkl'
        self.test_real_bin_filename               = script_dir / 'data' / preparation / 'test_real.bin'
        self.real_leading_test_particles_filename = script_dir / 'data' / preparation / 'real_leading_test_particles.csv'
        self.generated_samples_filename           = self.latest_sampling_dir / 'generated_samples.csv'
        self.filtered_samples_filename            = self.latest_sampling_dir / 'filtered_samples.csv'
        self.sampled_leading_particles_filename   = self.latest_sampling_dir / 'sampled_leading_particles.csv'
        self.untokenized_samples_filename         = self.latest_sampling_dir / 'untokenized_samples.csv'
        self.metrics_results_filename             = self.latest_sampling_dir / 'metrics_results.json'
        
        if not self.meta_filename.exists():
            print("Data has not been prepared! Please prepare data first!")
            exit()
            
        with open(self.dictionary_filename) as dictionary_file:
            dictionary = json.load(dictionary_file)

        # Convenience dictionary definitions
        p_bin_count = (dictionary["e_bin_data"]["max"] - dictionary["e_bin_data"]["min"]) // 1000
        e_bin_count = (dictionary["e_bin_data"]["max"] - dictionary["e_bin_data"]["min"]) // dictionary["e_bin_data"]["step_size"]
        eta_bin_count = int((dictionary["eta_bin_data"]["max"] - dictionary["eta_bin_data"]["min"]) // dictionary["eta_bin_data"]["step_size"])

        self.columns = ["num_particles", "pdgid", "e", "px", "py", "pz", "eta", "theta", "phi"]
        self.bin_settings = {
            "num_particles": { "min": 0,                                 "max": 50,                                "bins": 50 },
            "e":             { "min": dictionary["e_bin_data"]["min"],   "max": dictionary["e_bin_data"]["max"],   "bins": e_bin_count },
            "px":            { "min": dictionary["e_bin_data"]["min"],   "max": dictionary["e_bin_data"]["max"],   "bins": p_bin_count },
            "py":            { "min": dictionary["e_bin_data"]["min"],   "max": dictionary["e_bin_data"]["max"],   "bins": p_bin_count },
            "pz":            { "min": dictionary["e_bin_data"]["min"],   "max": dictionary["e_bin_data"]["max"],   "bins": p_bin_count },
            "eta":           { "min": dictionary["eta_bin_data"]["min"], "max": dictionary["eta_bin_data"]["max"], "bins": eta_bin_count },
            "theta":         { "min": -2 * np.pi,                        "max": 2 * np.pi,                         "bins": int((4 * np.pi) // dictionary["theta_bin_data"]["step_size"]) },
            "phi":           { "min": -2 * np.pi,                        "max": 2 * np.pi,                         "bins": int((4 * np.pi) // dictionary["phi_bin_data"]["step_size"]) },
        }
    
    def generate_leading_particle_information(self):
        # Output will be num_particles, pdgid, e, px, py, pz, eta, theta, phi

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
                theta = np.arccos(pz / r)
                phi = np.arctan2(py, px)
                eta = -np.log(np.tan(theta / 2))
                
                out_file.write(f'{len(secondaries)} {int(pdgid)} {e} {px} {py} {pz} {eta:.5f} {theta:.5f} {phi:.5f}\n')
    
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
        pdgid_offset_min = dictionary.PDGID_OFFSET
        pdgid_offset_max = dictionary.PDGID_OFFSET + len(dictionary.particles_index)
        energy_offset_min = dictionary.ENERGY_OFFSET
        energy_offset_max = dictionary.ENERGY_OFFSET + len(dictionary.e_bins)
        eta_offset_min = dictionary.ETA_OFFSET
        eta_offset_max = dictionary.ETA_OFFSET + len(dictionary.eta_bins)
        theta_offset_min = dictionary.THETA_OFFSET
        theta_offset_max = dictionary.THETA_OFFSET + len(dictionary.theta_bins)
        phi_offset_min = dictionary.PHI_OFFSET
        phi_offset_max = dictionary.PHI_OFFSET + len(dictionary.phi_bins)
        
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
    
    def filter_data_scheme_no_eta(self):
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
        tokenized_data = [e for e in tokenized_data if len(e) > 4 and len(e) % 4 == 0]
        
        # Ensure valid token ranges
        pdgid_offset_min = dictionary.PDGID_OFFSET
        pdgid_offset_max = dictionary.PDGID_OFFSET + len(dictionary.particles_index)
        energy_offset_min = dictionary.ENERGY_OFFSET
        energy_offset_max = dictionary.ENERGY_OFFSET + len(dictionary.e_bins)
        theta_offset_min = dictionary.THETA_OFFSET
        theta_offset_max = dictionary.THETA_OFFSET + len(dictionary.theta_bins)
        phi_offset_min = dictionary.PHI_OFFSET
        phi_offset_max = dictionary.PHI_OFFSET + len(dictionary.phi_bins)
        
        for event in tokenized_data:
            b_keep_event = True
            for i, token in enumerate(event):
                token_type_id = i % 4
                if token_type_id == 0:
                    if token < pdgid_offset_min or token >= pdgid_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 1:
                    if token < energy_offset_min or token >= energy_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 2:
                    if token < theta_offset_min or token >= theta_offset_max:
                        b_keep_event = False
                        break
                elif token_type_id == 3:
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
        if self.scheme == 'standard':
            self.filter_data()
            pTokenizer.untokenize_data(self.dictionary_filename.as_posix(), self.filtered_samples_filename.as_posix(), self.untokenized_samples_filename.as_posix())
        elif self.scheme == 'no_eta':
            self.filter_data_scheme_no_eta()
            pTokenizer.untokenize_data_scheme_no_eta(self.dictionary_filename.as_posix(), self.filtered_samples_filename.as_posix(), self.untokenized_samples_filename.as_posix())
        self.generate_leading_particle_information()

        df1 = pd.read_csv(self.real_leading_test_particles_filename, sep=" ", names=self.columns, engine="c", header=None)
        df2 = pd.read_csv(self.sampled_leading_particles_filename, sep=" ", names=self.columns, engine="c", header=None)

        for column, settings in self.bin_settings.items():
            if self.scheme == 'no_eta' and column == 'eta':
                continue                
            
            min_val = settings["min"]
            max_val = settings["max"]
            bins = settings["bins"]
            
            df1_weights = np.ones_like(df1[column]) / len(df1[column])
            df2_weights = np.ones_like(df2[column]) / len(df2[column])

            plt.figure(figsize=(21, 6))
            plt.subplot(1, 2, 1)
            plt.xlim([min_val, max_val])
            plt.hist(df1[column], bins=bins, weights=df1_weights, range=(min_val, max_val), alpha=0.7, color="blue", label="Input")
            plt.hist(df2[column], bins=bins, weights=df2_weights, range=(min_val, max_val), alpha=0.7, color="orange", label="Sampled")
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.savefig(f"{self.latest_sampling_dir.as_posix()}/histogram_{column}.png", bbox_inches='tight')

    def calculate_metrics(self):
        real_df = pd.read_csv(self.real_leading_test_particles_filename, sep=" ", names=self.columns, engine="c", header=None)
        generated_df = pd.read_csv(self.sampled_leading_particles_filename, sep=" ", names=self.columns, engine="c", header=None)

        for column, settings in self.bin_settings.items():
            # NOTE: These being normalized is VERY IMPORTANT.
            real_histogram = np.histogram(real_df[column], bins=settings['bins'], range=(settings['min'], settings['max']), density=True)
            generated_histogram = np.histogram(generated_df[column], bins=settings['bins'], range=(settings['min'], settings['max']), density=True)
            settings['real_histogram'] = real_histogram
            settings['generated_histogram'] = generated_histogram

        with open(self.meta_filename, 'rb') as f:
            meta = pickle.load(f)
            num_particles_per_event = meta['num_particles_per_event']
            
        # -------------------------------------------------------------------------------
        # Preparing real Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # -------------------------------------------------------------------------------

        test_real_events = np.memmap(self.test_real_bin_filename, dtype=np.double, mode='r')
        test_real_events = test_real_events.reshape(-1, num_particles_per_event * 5)

        real_jets = []
        for event in test_real_events:
            particles = event.reshape(-1, 5)
            
            single_jet = []
            for particle in particles:
                # Calculate transverse momentum, eta, and phi
                pdgid, e, px, py, pz = particle
                
                # Skip padding particles
                if pdgid == -1:
                    single_jet.append([0, 0, 0])
                    continue
                
                p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
                pt = np.sqrt(px ** 2 + py ** 2)
                theta = np.arctan(pz / p)
                eta = -np.log(np.tan(theta / 2))
                phi = np.arctan2(py, px)
                single_jet.append([eta, phi, pt])
            
            real_jets.append(single_jet)

        real_jets = np.array(real_jets)

        # -------------------------------------------------------------------------------
        # Preparing generated Jet data
        # Features for jet in JetNet is (eta, phi, pT), in that order
        # -------------------------------------------------------------------------------

        with open(self.untokenized_samples_filename, 'r') as f:
            untokenized_samples = f.readlines()
            
        generated_jets = []
        for sample in untokenized_samples:
            particles = sample.strip().split(';')
            
            single_jet = []
            for particle in particles:
                particle = particle.strip().split()
                
                if len(particle) == 0:
                    break
                
                pdgid, e, px, py, pz = map(np.double, particle)
                p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
                pt = np.sqrt(px ** 2 + py ** 2)
                theta = np.arctan(pz / p)
                eta = -np.log(np.tan(theta / 2))
                phi = np.arctan2(py, px)
                
                features = [eta, phi, pt]
                single_jet.append(features)
            
            generated_jets.append(single_jet)

        def pad_to_shape(lst, pad_value=0):
            num_sublists = len(lst)
            max_length = num_particles_per_event # max(len(sublist) for sublist in lst)
            padded_array = np.full((num_sublists, max_length, 3), pad_value, dtype=np.float64)
            for i, sublist in enumerate(lst):
                for j, values in enumerate(sublist):
                    if j < max_length:
                        padded_array[i, j, :] = values  
            return padded_array

        generated_jets = pad_to_shape(generated_jets)

        # -------------------------------------------------------------------------------
        # Coverage and MMD
        # -------------------------------------------------------------------------------

        real_jets = np.nan_to_num(real_jets, nan=0)
        generated_jets = np.nan_to_num(generated_jets, nan=0)

        real_jets = real_jets[:len(generated_jets)]
        cov, mmd = jetnet.evaluation.cov_mmd(real_jets, generated_jets)
        # print(cov, mmd)

        # -------------------------------------------------------------------------------
        # FPD and KPD
        # -------------------------------------------------------------------------------

        suggested_real_features = jetnet.evaluation.get_fpd_kpd_jet_features(real_jets)
        suggested_generated_features = jetnet.evaluation.get_fpd_kpd_jet_features(generated_jets)

        suggested_real_features = np.nan_to_num(suggested_real_features, nan=0)
        suggested_generated_features = np.nan_to_num(suggested_generated_features, nan=0)

        kpd_median, kpd_error = jetnet.evaluation.kpd(suggested_real_features, suggested_generated_features, num_threads=0)
        # print(f"KPD median: {kpd_median}, KPD error: {kpd_error}")

        fpd_value, fpd_error = jetnet.evaluation.fpd(suggested_real_features, suggested_generated_features)
        # print(f"FPD valid: {fpd_value}, FPD error: {fpd_error}")

        # -------------------------------------------------------------------------------
        # Wasserstein Distances 
        # -------------------------------------------------------------------------------

        # Wasserstein distances between Energy Flow Polynomials
        avg_w1_scores_efp = jetnet.evaluation.w1efp(real_jets, generated_jets)
        # print(avg_w1_scores_efp)

        # Wasserstein distance between masses of jets1 and jets2
        w1_mass_score = jetnet.evaluation.w1m(real_jets, generated_jets)
        # print(w1_mass_score)

        # Wasserstein distances between particle features of jets1 and jets2
        avg_w1_scores_features = jetnet.evaluation.w1p(real_jets, generated_jets)
        # print(avg_w1_scores_features)

        metrics_results_dict = {
            "coverage": cov,
            "mmd": mmd,
            "kpd_median": kpd_median,
            "kpd_error": kpd_error,
            "fpd_value": fpd_value,
            "fpd_error": fpd_error
        }

        with open(self.metrics_results_filename, "w") as opt_file:
            json.dump(metrics_results_dict, opt_file, indent=4)

# This function is used to extract the numbers from the dataset folder name for sorting so our table is in order.
def extract_numbers(folder_name):
    parts = folder_name.split('_') 
    numbers = [int(part) for part in parts if part.isdigit()]
    return tuple(numbers)

# Need a function for multi-threading
def analyze_dataset(out_data_dir_name):
    # Extract dataset name from sampling info
    latest_sampling_dir = pUtil.get_latest_sampling_dir(out_data_dir_name)
    sampling_info_file = script_dir / 'generated_samples' / out_data_dir_name / latest_sampling_dir / 'sampling_info.json'
    with open(sampling_info_file) as f:
        sampling_info = json.load(f)
        dataset = sampling_info['dataset']

    # Run the analysis
    dataset_analyzer = Analyzer(dataset, out_data_dir_name)
    dataset_analyzer.generate_distributions()
    dataset_analyzer.calculate_metrics()

if __name__ == "__main__":
    # If argument 'all' is provided, generate distributions and metrics for all sampled datasets' latest sampling
    if 'all' in sys.argv:
        print("Generating distributions and metrics for all datasets.")
        
        # Find all datasets
        generated_samples_path = script_dir / 'generated_samples'
        out_data_dir_names = [folder.name for folder in generated_samples_path.iterdir() if folder.is_dir()]
        out_data_dir_names = sorted(out_data_dir_names, key=extract_numbers)
        
        if 'single_threaded' not in sys.argv:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(analyze_dataset, out_data_dir_names)
        else:
            for out_data_dir_name in out_data_dir_names:
                analyze_dataset(out_data_dir_name)
        
        print("Distributions and metrics generated successfully for all datasets.")
    else:
        print(f'Generating distributions and metrics for dataset {configurator.preparation_name}.')
        dataset_analyzer = Analyzer(configurator.preparation_name, configurator.model_name, configurator.scheme)
        dataset_analyzer.generate_distributions()
        # dataset_analyzer.calculate_metrics()
        print("Distributions and metrics generated successfully.")