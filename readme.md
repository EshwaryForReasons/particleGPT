# ParticleGPT #

GPT based on nanoGPT for generating particle collision data.

```
https://github.com/scikit-hep/particle used to PDGID conversions.
https://hub.docker.com/repository/docker/eshym/docker_img_particlegpt/general docker image with correct dependencies.
```

## Usage ##

The dataset should be in `data/dataset_name/data.csv`. The `*.py` and `dictionary.json` files will always be the same so just copy/paste from a different dataset. The `particles_id` and `particles_index` objects in `dictionary.json` will be updated in the preparation phase when training the model. This is to ensure only relevant particles (those present in the dataset) are included in the vocabulary. The data is only prepared once. Subsequent training will just use the preprepared data.

If the dataset, any of the python files, or the dictionary are changed (for example to change the tokenization) then be sure to delete `outputs/meta.pkl` in the dataset directory to trigger repreparation as otherwise the dataset will not be reprepared and the old tokenized files will be reused.

```
# For first usage we need to ensure pTokenizerModule is built and works
cd pTokenizer
mkdir build
cd build
cmake ..
make
cd ..
python setup.py bdist_wheel
pip install dist/*.whl
```

Using the docker image `eshym/docker_img_particlegpt` is necessary for the correct python dependencies.
```
shifterimg -v pull docker:eshym/docker_img_particlegpt:v2
shifter --image=docker:eshym/docker_img_particlegpt:v2 /bin/bash

# Training:
python train.py config/dataset_to_train.json # Single node, single GPU
torchrun --standalone --nproc_per_node=4 train.py config/dataset_5_1_1.json # Single node, multiple GPU (4 here)

# Sampling:
python sample.py config/dataset_to_sample.json

# Generate distributions:
python generate_distributions.py config/dataset_to_use.json
```

Trained models are stored in `trained_models/dataset_name`. Generated samples are stored in `generated_samples/dataset_name/sampling_index`.

`analysis.ipynb` includes all graphing. By default all cells will use the latest generated sample. Make sure to run `generate_distributions.py` on the relevant dataset first.

## Notes ##

```
# Running interactive job:
srun -C "gpu" -q interactive -N 1 -G 1 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -l
srun -C "gpu" -q interactive -N 1 -G 4 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -l
srun -C "gpu&hbm80g" -q interactive -N 1 -G 4 -c 32 -t 4:00:00 -A m3443 --pty /bin/bash -l

# Running batch job:
python submit_job.py job_scripts/job_config_to_use.json

# Profiling scripts using cProfile:
python -m cProfile -o output_file_name.profile script_to_profile.py

# Visualizing profiling using snakeviz:
# Make sure to use the port that ssh tunnels
snakeviz output_file_name.profile -p 8080 -s
```

dataset_1 has 10,000 events
dataset_2 has 100,000 events
dataset_3 has 10,000 events
dataset_4 had 1,000,000 events
dataset_5 has 10,000,000 events
dataset_6 has 10,000 events

Submitting batch jobs will add it to `current_jobs.md` automatically.

pTokenizer/build/
pTokenizer/dist/
pTokenizer/pTokenizer.egg-info/
pTokenizer/pybind11