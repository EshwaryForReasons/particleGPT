import pTokenizerModule as pTokenizer
import pUtil
from dictionary import Dictionary
import numpy as np

model_name = 'model_5_2_1'
dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
tokenized_data_filename = pUtil.get_model_preparation_dir(model_name) / 'tokenized_data.csv'
test_tokenized_bin_filename = pUtil.get_model_preparation_dir(model_name) / 'test_tokenized.bin'
written_test_tokenized_filename = pUtil.get_temp_dir() / 'written_test_tokenized.csv'
written_test_filtered_filename = pUtil.get_temp_dir() / 'written_test_filtered.csv'
written_test_untokenized_filename = pUtil.get_temp_dir() / 'written_test_untokenized.csv'
real_leading_test_particles_filename = pUtil.get_temp_dir() / 'real_leading_test_particles.txt'

def filter_data(model_name):
    dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
    dictionary = Dictionary(dictionary_filename)
    # Load data
    tokenized_data = []
    with open(written_test_tokenized_filename) as gen_samples_file:
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
    with open(written_test_filtered_filename, 'w') as filtered_file:
        for event in filtered_data:
            event = [str(x) for x in event]
            event = ' '.join(event)
            filtered_file.write(event + '\n')

filter_data(model_name)
pTokenizer.untokenize_data(dictionary_filename.as_posix(), written_test_filtered_filename.as_posix(), written_test_untokenized_filename.as_posix())

with open(real_leading_test_particles_filename, 'w') as out_file, open(written_test_untokenized_filename, 'r') as in_file:
    for event in in_file:
        particles = event.strip().split(';')
        particles = [particle.split() for particle in particles]
        particles = particles[:-1]
        particles = [[int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])] for p in particles]
        particles = np.array(particles)
        particles = particles.reshape(-1, 5)
        secondaries = particles[1:]
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