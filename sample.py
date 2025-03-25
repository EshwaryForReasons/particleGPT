"""
Sample from a trained model
"""
from pathlib import Path
import torch
import numpy as np
import pickle
from contextlib import nullcontext
from timeit import default_timer as timer

import pLogging
import pUtil
import configurator
import model
from model import GPTConfig, GPT

script_dir = Path(__file__).resolve().parent

# sampling_ids are consecutive so we need to determine the next one (for us)
sampling_id = pUtil.get_latest_sampling_id(configurator.output_dir_name) + 1

model_filename = Path(pUtil.get_training_dir(configurator.output_dir_name), 'ckpt.pt')
test_bin_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'test.bin')
meta_filename = Path(script_dir, 'data', configurator.dataset, 'outputs', 'meta.pkl')
samples_output_filename = Path(pUtil.get_sampling_dir(configurator.output_dir_name), f'sampling_{sampling_id}', 'generated_samples.csv')

# Ensure the output directory of the samples exists. This needs to be done before the logger is created.
Path(samples_output_filename).parent.mkdir(parents=True, exist_ok=True)

logger_idx = pLogging.create_sampling_logger(configurator.output_dir_name, sampling_id)
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Sample generation started.')

if not meta_filename.exists():
    pLogging.info(logger_idx, "Data not prepared!")
    exit()
if not model_filename.exists():
    pLogging.info(logger_idx, "Could not find trained model!")
    exit()
    
pLogging.info(logger_idx, "Sampling info", {
    "dataset": configurator.dataset,
    "model_path": model_filename,
    "samples_output_filename": samples_output_filename,
    "max_new_tokens": configurator.max_new_tokens,
    "temperature": configurator.temperature,
    "top_k": configurator.top_k,
    "seed": configurator.seed,
    "device": configurator.device,
    "dtype": configurator.dtype,
    "compile": configurator.compile
})

torch.manual_seed(configurator.seed)
torch.cuda.manual_seed(configurator.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in configurator.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[configurator.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
checkpoint = torch.load(model_filename, map_location=configurator.device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(configurator.device)
if compile:
    model = torch.compile(model)

# Generate samples
with torch.no_grad(), ctx, open(samples_output_filename, 'a') as out_file, open(meta_filename, 'rb') as meta_file:
    meta = pickle.load(meta_file)
    max_sequence_len = meta['max_sequence_length']
    
    test_data = np.memmap(test_bin_filename, dtype=np.uint16, mode='r')
    test_data = test_data.reshape(-1, max_sequence_len)
    sampling_starters = [event[:8] for event in test_data]
    
    pLogging.info(logger_idx, 'Generating samples.')
    
    start = timer()
    for starting_tokens in sampling_starters:
        x = (torch.tensor(starting_tokens.tolist(), dtype=torch.long, device=configurator.device)[None, ...])
        y = model.generate(x, max_sequence_len, temperature=configurator.temperature, top_k=configurator.top_k)
        out_file.write(" ".join(map(str, y[0].tolist())) + "\n")
    end = timer()
    
    pLogging.info(logger_idx, f'Sample generation finished in {end - start} seconds.')