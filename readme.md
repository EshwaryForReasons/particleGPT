# ParticleGPT #

GPT based on nanoGPT for generating particle collision data.

## Usage ##

`control_panel.ipynb` includes all actions. Just run the appropriate cell. MAKE SURE TO RUN THE SETUP CELL FIRST!!!!

## Notes ##

- Make sure converting spherical to cartesian coordinates is done correctly

## TODO ##

VERY IMPORTANT: A LOT OF TOKENS ARE HARDCODED, FIX THAT!!!!

For now:
- [ ] Redo distributions and fix them
  - [ ] Increase granularity of bins
  - [ ] Play with hyperparameters
- [ ] Finish logging script
- [ ] Switch to jupiter notebook as it is better for visualizing data, especially in an ssh session
- [ ] Figure out why training takes so long on Perlmutter but not on my laptop
- [ ] Streamline the training and sample generation
  - [x] Create a `control_panel.ipynb` which can handle:
    - [ ] Training (outputs plots and final loss, val)
    - [ ] Sampling (filters immediately after and outputs filtering info)
    - [ ] Distribution generation (histograms for particle count, pdgid, e, px, py, pz for sampled and input data)

For later:
- [ ] Maybe treat bins as gaussian distribution for untokenized data