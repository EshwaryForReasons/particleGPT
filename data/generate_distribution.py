
import tokenizer
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir_name = sys.argv[1]

dataset_storage_dir = sys.argv[2]
filtered_samples_file = os.path.join(dataset_storage_dir, 'filtered_samples.txt')
untokenized_samples_file = os.path.join(dataset_storage_dir, 'untokenized_samples.csv')
sorted_samples_file = os.path.join(dataset_storage_dir, 'sorted_samples.csv')
sampled_leading_particles_file = os.path.join(dataset_storage_dir, 'sampled_leading_particles.csv')

input_file = os.path.join(script_dir, output_dir_name, 'data.csv')
sorted_inputs_file = os.path.join(dataset_storage_dir, 'sorted_inputs.csv')
input_leading_particles_file = os.path.join(dataset_storage_dir, 'input_leading_particles.csv')

# Untokenize the filtered output
tokenizer.untokenize_data(filtered_samples_file, untokenized_samples_file)

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

reorder_particles(untokenized_samples_file, sorted_samples_file)
extract_leading_particle(sorted_samples_file, sampled_leading_particles_file)

reorder_particles(input_file, sorted_inputs_file)
extract_leading_particle(sorted_inputs_file, input_leading_particles_file)