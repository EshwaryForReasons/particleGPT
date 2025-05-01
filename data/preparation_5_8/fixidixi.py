import json

# Step 1: Load data from dictionary.json
with open('dictionary.json', 'r') as infile:
    data = json.load(infile)

particles_index = data["particles_index"]
particles_id = data["particles_id"]

# Step 2: Create a name -> PGD ID mapping for reverse lookup
name_to_pgd_id = {name: pgd_id for pgd_id, name in particles_id.items()}

# Step 3: Create the index -> PGD ID mapping
index_to_pgd_id = {
    index: name_to_pgd_id[name]
    for name, index in particles_index.items()
    if name in name_to_pgd_id
}

# Step 4: Fill in missing indices up to 74 with "reserved"
for i in range(75):
    if i not in index_to_pgd_id:
        index_to_pgd_id[str(i)] = "reserved"

# Step 5: Finalize and write to file
data['pdgids'] = index_to_pgd_id
del data['particles_index']
del data['particles_id']

with open('dictionary.json', 'w') as outfile:
    json.dump(data, outfile, indent=2)

print("Index to PGD ID mapping (with reserved entries) saved to new_dictionary.json.")
