
from tokenizer import untokenize_data
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_storage_dir = sys.argv[1]
filtered_sample_output_file = os.path.join(dataset_storage_dir, 'filtered_sample.txt')
untokenized_sample_output_file = os.path.join(dataset_storage_dir, 'untokenized_sample.csv')
sorted_sample_output_file = os.path.join(dataset_storage_dir, 'sorted_sample.csv')
leading_particle_sample_output_file = os.path.join(dataset_storage_dir, 'sampled_leading_particle.csv')

input_file = os.path.join(script_dir, 'data.csv')
sorted_input_output_file = os.path.join(dataset_storage_dir, 'sorted_input.csv')
leading_particle_input_output_file = os.path.join(dataset_storage_dir, 'input_leading_particle.csv')

# Untokenize the filtered output
untokenize_data(filtered_sample_output_file, untokenized_sample_output_file)

# Reorder each event to keep input event first and sort the rest by energy
def reorder_particles(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        reader = infile.readlines()

        for line in reader:
            parts = line.strip().split(';')
            num_particles = len(parts) - 1
            first_part = parts[0]
            remaining_parts = parts[1:]
            split_parts = [part.split() for part in remaining_parts]
            split_parts = [part for part in split_parts if len(part) > 1]
            sorted_parts = sorted(split_parts, key=lambda x: float(x[1]), reverse=True)
            sorted_parts = [' '.join(part) for part in sorted_parts]
            result = [str(num_particles)] + [first_part] + sorted_parts
            outfile.write('; '.join(result) + ';\n')

def extract_leading_particle(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        reader = infile.readlines()

        for line in reader:
            parts = line.strip().split(';')
            num_particles = len(parts) - 1
            remaining_parts = parts[1:]
            split_parts = [part.split() for part in remaining_parts]
            split_parts = [part for part in split_parts if len(part) > 1]
            sorted_parts = sorted(split_parts, key=lambda x: float(x[1]), reverse=True)
            largest_second_value_part = sorted_parts[0] if sorted_parts else []

            if largest_second_value_part:
                result = [str(num_particles)] + [' '.join(largest_second_value_part)] 
                outfile.write(' '.join(result) + '\n')

reorder_particles(untokenized_sample_output_file, sorted_sample_output_file)
extract_leading_particle(sorted_sample_output_file, leading_particle_sample_output_file)

reorder_particles(input_file, sorted_input_output_file)
extract_leading_particle(sorted_input_output_file, leading_particle_input_output_file)