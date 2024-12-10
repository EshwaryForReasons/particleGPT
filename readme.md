# ParticleGPT #

GPT based on nanoGPT for generating particle collision data.

## Usage ##

For training:

- Just run the training script `python train.py config/config_file.py`. The script will handle preparing the data.

For sampling:

- Just run the sample script `python sample.py config/config_file.py`. The output will be in the file specified in `config_file.py`.
- 
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
  - [ ] Create a `control_panel.ipynb` which can handle:
    - [ ] Training (outputs plots and final loss, val)
    - [ ] Sampling (filters immediately after and outputs filtering info)
    - [ ] Distribution generation (histograms for particle count, pdgid, e, px, py, pz for sampled and input data)

Archive as of December 6, 2024:
For now:
- [x] Print out variable ranges in humandictionary.txt
- [x] Place EVENT_END token at the end of the event (before padding tokens)
- [x] Try reducing learning rate (and messing with other parameters) to find notable differences
  - [x] Causes terribly long training times (ask about this)
- [x] Add stop condition for token generation

Evaluation script:
- [x] Compare distributions from GPT to MCGenerators
    - [x] Particle count distribution
    - [x] pdgid distribution
    - [x] e distribution
    - [x] px distribution
    - [x] py distribution
    - [x] pz distribution
- Will do more complicated ones later; we must first ensure the basic 6 work
  - They do not work, are bad
  - [ ] Increase granularity of bins
  - [ ] Play with hyperparameters to improve distributions

Convenience items:
- [x] Have train.py run prepare.py for the dateset
- [x] Have train.py log run information (for record-keeping purposes and to make graphing easier)
- [ ] Have sample.py run filter_output.py and log the information from it
- [x] Make a proper logging script
  - The idea is any other script can use it for logging (it will name and prefix the log files based on various metadata)
  - e.g "Various metadata": date, running script, and dataset being processed, etc.

For later:
- [ ] Maybe treat bins as gaussian distribution for untokenized data

## Test Notes ##

small_test:
- Run on the 1000 event dataset
- Most of the data is malformed
    - Particles do not have 5 tokens each
- Running for 5000 or 50000 iterations does not make a difference in data malformity

small_test_delineated:
- Run on the 1000 event dataset
- This one includes tokens for the start and end of the particles as well
    - I suspect GPT will be more consistent in its generations with those tokens included
    - small_test leads to many malformed events, hopefully this one will not suffer the same

- It seems it is indeed able to generate tokens more consistently (less malformed events) with the delineation tokens
- This one performs miles better; I will be using the event_start and event_end tokens from now on

- observations:
    - I suspect its just adding '0' tokens regardless of quantity because of the input data
        - This does not happen once we add '3' and '4' tokens

small_test_ending_fix_a:
- Run on the 1000 event dataset
- This test includes delineation tokens and moves the EVENT_END token to the end of the generated particles (before the padding)
- IMPORTANT: This test DOES include delineation tokens for the padding particles as well
    - Padding is not purely a tensor alignment tool, it also provides gpt with a "feel" for how many particles an event should have. An "intuitive" upper limit of sorts.

- Observations:
    - Output looks good!

small_test_ending_fix_b:
- Run on the 1000 event dataset
- This test includes delineation tokens and moves the EVENT_END token to the end of the generated particles (before the padding)
- IMPORTANT: This test DOES NOT include delineation tokens for the padding particles as well
    - Padding is treated purely as a tensor alignment tool

- Observations:
    - This one gave us a mega event...
    - 1 11 69 79 159 287 4 3 7 69 79 159 288 4 3 16 69 79 159 286 4 3 11 69 79 159 287 4 3 11 69 79 159 288 4 3 16 69 79 158 287 4 3 10 69 79 160 287 4 3 7 69 79 159 288 4 3 11 69 78 162 288 4 3 10 69 78 162 288 4 3 11 69 79 158 292 4 3 10 69 79 159 291 4 3 10 69 79 159 288 4 3 11 69 79 158 288 4 3 7 69 79 159 288 4 3 10 69 79 160 289 4 3 7 69 79 157 288 4 3 10 69 80 155 288 4 3 17 69 79 160 290 4 3 7 69 79 159 286 4 3 11 69 78 161 279 4 3 7 69 79 160 278 4 3 7 69 79 160 285 4 3 10 69 78 161 285 4 3 11 69 79 160 288 4 3 10 69 80 155 288 4 3 10 69 79 160 286 4 3 7 69 79 159 287 4 3 7 69 78 163 287 4 3 7 69 79 160 286 4 3 10 69 79 159 288 4 3 17 69 79 160 281 4 3 7 69 79 160 288 4 3 11 69 78 162 287 4 3 7 69 79 160 286 4 3 7 69 78 161 286 4 3 10 69 78 161 286 4 3 16 69 78 161 246 4 3 12 69 79 159 287 4 3 7 69 78 161 287 4 3 10 69 78 163 282 4 3 7 69 78 161 286 4 3 7 69 80 156 273 4 3 17 69 78 164 287 4 3 10 69 78 165 292 4 3 11 69 78 161 287 4 3 10 69 78 162 290 4 3 10 69 78 161 291 4 2
    - Very interesting!! This one generally has larger events!
      - I suspect it just generates until it feels like, whereas for small_test_ending_fix_a it has a "feel" for now many particles an event should have since they are delineated
      - I will likely stick to the small_test_ending_fix_a version because of this discovery