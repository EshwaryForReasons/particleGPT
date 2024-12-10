
from tokenizer import untokenize_data
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Untokenize the filtered output
untokenize_data(os.path.join(script_dir, 'outputs/filtered_outputs.txt'), os.path.join(script_dir, 'outputs/untokenized_output.csv'))

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

reorder_particles('outputs/untokenized_output.csv', 'outputs/sorted_samples.csv')
extract_leading_particle('outputs/sorted_samples.csv', 'outputs/sampled_leading_particle.csv')

reorder_particles('data.csv', 'outputs/sorted_data.csv')
extract_leading_particle('outputs/sorted_data.csv', 'outputs/input_leading_particle.csv')