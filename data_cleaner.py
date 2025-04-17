import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent

def create_dataset_5c1():
    dataset_path = script_dir / 'data' / 'dataset_5.csv'
    out_dataset_path = script_dir / 'data' / 'dataset_5c1.csv'

    dataset_loaded = []
    with open(dataset_path, "r") as f:
        for event in f:
            event = event.strip().split(';')
            event = [
                [int(e.split()[0])] + [float(x) for x in e.split()[1:]]
                for e in event
            ]
            dataset_loaded.append(event)
            
    # Remove all events with particles < 5 or > 35
    dataset_loaded = [e for e in dataset_loaded if (len(e) >= 5 and len(e) <= 35)]

    with open(out_dataset_path, "w") as f:
        for event in dataset_loaded:
            event_str = ';'.join([' '.join(map(str, e)) for e in event])
            f.write(event_str + '\n')

def validate_dataset_5c1():
    out_dataset_path = script_dir / 'data' / 'dataset_5c1.csv'
    
    with open(out_dataset_path, 'r') as f:
        for event in f:
            event = event.strip().split(';')
            num_particles = len(event)
            if num_particles > 35 or num_particles < 5:
                print(f"Error: Event with {num_particles} particles found.")

validate_dataset_5c1()