import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import data_manager

script_dir = Path(__file__).resolve().parent

in_dataset_filename = script_dir / "data" / "dataset_5.csv"
out_dataset_filename = script_dir / "data" / "dataset_5a.csv"

# Returns freq containing the frequency of each PDGID in the dataset and
# occurrences containing the events (as row numbers) in which each PDGID occurs.
def get_pdgid_frequency_distribution(dataset):
    freq = Counter()
    occurrences = defaultdict(list)
    
    for event_idx, event in enumerate(dataset):
        found_ids = set()
        for particle in event:
            pdgid = particle[0]
            if pdgid == 0.0:
                continue
            freq[pdgid] += 1
            found_ids.add(pdgid)
        for pid in found_ids:
            occurrences[pid].append(event_idx)
            
    return freq, occurrences

# If we remove all events containing the n_least_frequent particles, which events (as row numbers) do we lose?
def find_events_lost_due_to_particle_removal(rows_occuring, n_least_frequent):
    sorted_items = sorted(rows_occuring.items(), key=lambda x: len(x[1]), reverse=False)
    sorted_items = sorted_items[:n_least_frequent]
    
    events_lost = set()
    for particle, events in sorted_items:
        events_lost.update(events)
    return events_lost

# Returns number of least frequent particles we can remove and still only remove n_allowed_removals events.
def calculate_num_removable_particles(rows_occuring, n_allowed_event_removals):
    n_most_removable_particles = 0
    for i in range(0, len(rows_occuring)):
        removed_events = find_events_lost_due_to_particle_removal(rows_occuring, i)
        n_removed_events = len(removed_events)
        if n_removed_events <= n_allowed_event_removals:
            n_most_removable_particles = i
    return n_most_removable_particles



# Plots frequency of each PDGID
def plot_pdgid_particle_frequency_distribution(freq, use_log=True):
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_items)

    plt.figure(figsize=(14, 6))
    if use_log:
        plt.yscale("log")
    plt.bar(range(len(labels)), counts, tick_label=labels, color="blue")
    plt.xlabel("PDGID")
    plt.ylabel("Frequency")
    plt.title(f"Frequency Distribution of PDGIDs in {in_dataset_filename.name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(script_dir / f'pdgid_freq_particle_dist_{in_dataset_filename.name}.png', bbox_inches='tight')

# Plots frequency of events containing each PDGID
def plot_pdgid_event_frequency_distribution(rows_occuring, use_log=True):
    freq = {}
    for pid, n_events in rows_occuring.items():
        freq[pid] = len(n_events)
        
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_items)

    plt.figure(figsize=(14, 6))
    if use_log:
        plt.yscale("log")
    plt.bar(range(len(labels)), counts, tick_label=labels, color="blue")
    plt.xlabel("PDGID")
    plt.ylabel("Frequency")
    plt.title(f"Frequency Distribution of PDGIDs in {in_dataset_filename.name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(script_dir / f'pdgid_freq_event_dist_{in_dataset_filename.name}.png', bbox_inches='tight')


loaded_dataset = data_manager.load_geant4_dataset(in_dataset_filename, pad_token=0.0)
freq, occurrences = get_pdgid_frequency_distribution(loaded_dataset)
plot_pdgid_particle_frequency_distribution(freq, use_log=False)
plot_pdgid_event_frequency_distribution(occurrences, use_log=False)

sorted_frequency = sorted(freq.items(), key=lambda f: f[1], reverse=True)
sorted_rows_occuring = sorted(occurrences.items(), key=lambda f: len(f[1]), reverse=True)

x = {}
for pid, events_occ in sorted_rows_occuring:
    x[pid] = len(events_occ)
print('Particle id and how many events it occurs in.')
print(json.dumps(x, indent=4))

y = {}
for pid, fqr in sorted_frequency:
    y[pid] = fqr
print('Particle id and how many times it occurs in the dataset.')
print(json.dumps(y, indent=4))

# Map num of least frequent particles removed to number of events lost
z = {}
for i in range(1, len(occurrences) + 1):
    events_lost = find_events_lost_due_to_particle_removal(occurrences, i)
    n_events_lost = len(events_lost)
    z[i] = n_events_lost
print('Number of least frequent particles removed and events affected.')
print(json.dumps(z, indent=4))

# Say we are okay with removing 10,000 events.
n_allowed_event_removals = 10_000
n_most_removable_particles = calculate_num_removable_particles(occurrences, n_allowed_event_removals)
print(f'We can remove {n_most_removable_particles} particles and still only remove {n_allowed_event_removals} events.')