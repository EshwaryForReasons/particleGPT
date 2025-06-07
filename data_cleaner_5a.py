import json
from pathlib import Path

import data_manager
import analysis as anal

script_dir = Path(__file__).resolve().parent

in_dataset_filename = script_dir / "data" / "dataset_2.csv"
out_dataset_filename = script_dir / "data" / "dataset_5a.csv"

loaded_dataset = data_manager.load_geant4_dataset(in_dataset_filename, pad_token=0.0)
freq, occurrences = anal.dataset.get_pdgid_frequency_distribution(loaded_dataset)

out_file = script_dir / f'pdgid_freq_particle_dist_{in_dataset_filename.name}_new.png'
# anal.plotting.plot_bar(freq, 'Dataset', use_log=True, out_file=out_file)

# Turn occurrences into a freq distribution before plotting.
ofreq = {}
for pid, n_events in occurrences.items():
    ofreq[pid] = len(n_events)
out_file = script_dir / f'pdgid_freq_event_dist_{in_dataset_filename.name}_new.png'
# anal.plotting.plot_bar(ofreq, 'Dataset', use_log=True, out_file=out_file)

sorted_frequency = sorted(freq.items(), key=lambda f: f[1], reverse=True)
sorted_rows_occuring = sorted(occurrences.items(), key=lambda f: len(f[1]), reverse=True)

# Map of particle id and how many events it occurs in.
x = {}
for pid, events_occ in sorted_rows_occuring:
    x[pid] = len(events_occ)
print('Particle id and how many events it occurs in.')
print(json.dumps(x, indent=4))

# Map of particle di and how many  times it occurs in the dataset.
y = {}
for pid, fqr in sorted_frequency:
    y[pid] = fqr
print('Particle id and how many times it occurs in the dataset.')
print(json.dumps(y, indent=4))

# Map num of least frequent particles removed to number of events lost.
z = {}
for i in range(1, len(occurrences) + 1):
    events_lost = anal.dataset.find_events_lost_due_to_particle_removal(occurrences, i)
    n_events_lost = len(events_lost)
    z[i] = n_events_lost
print('Number of least frequent particles removed and events affected.')
print(json.dumps(z, indent=4))

# Say we are okay with removing 10,000 events, how many particle can we remove?
n_allowed_event_removals = 10_000
n_most_removable_particles = anal.dataset.calculate_num_removable_particles(occurrences, n_allowed_event_removals)
print(f'We can remove {n_most_removable_particles} particles and still only remove {n_allowed_event_removals} events.')