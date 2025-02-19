"""
Sample from a trained model
"""
import os
import torch
import subprocess
import sys
import pUtil
from contextlib import nullcontext
from timeit import default_timer as timer

import pLogging
import configurator
import model
from model import GPTConfig, GPT
import data.prepare as prepare

script_dir = os.path.dirname(os.path.abspath(__file__))

# exec(open('configurator.py').read())

# sampling_ids are consecutive so we need to determine the next one (for us)
sampling_id = pUtil.get_latest_sampling_id(configurator.output_dir_name) + 1
model_path = os.path.join(pUtil.get_training_dir(configurator.output_dir_name), 'ckpt.pt')
samples_leading_input_file = os.path.join(pUtil.get_dataset_dir(configurator.dataset), 'outputs', 'temp_sampling_lead.csv')
samples_output_file = os.path.join(pUtil.get_sampling_dir(configurator.output_dir_name), f'sampling_{sampling_id}', 'generated_samples.txt')

# Prepare the data before sampling it
prepare.prepare_sampling

# Ensure the output directory of the samples exists. This needs to be done before the logger is created.
os.makedirs(os.path.dirname(os.path.join(script_dir, samples_output_file)), exist_ok=True)

logger_idx = pLogging.create_sampling_logger(configurator.output_dir_name, sampling_id)
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Sample generation started.')

if not os.path.exists(model_path):
    pLogging.info(logger_idx, "Could not find trained model!")
    exit()
if not os.path.exists(samples_leading_input_file):
    pLogging.info(logger_idx, "Could not find leading tokens!")
    exit()
    
pLogging.info(logger_idx, "Sampling info", {
    "dataset": configurator.dataset,
    "model_path": model_path,
    "samples_output_filename": samples_output_file,
    "start": configurator.start,
    "num_samples": configurator.num_samples,
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
checkpoint = torch.load(model_path, map_location=configurator.device)
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
with torch.no_grad(), ctx, open(samples_output_file, 'a') as out_file, open(samples_leading_input_file, 'r') as in_file:
    pLogging.info(logger_idx, 'Generating samples.')
    
    start = timer()
    for sample_lead in in_file:
        # encode the beginning of the prompt
        start_ids = [int(x) for x in sample_lead.split()]
        x = (torch.tensor(start_ids, dtype=torch.long, device=configurator.device)[None, ...])
        y = model.generate(x, configurator.max_new_tokens, temperature=configurator.temperature, top_k=configurator.top_k)
        out_file.write(str(y[0].tolist()).strip('[]').replace(',', '') + '\n')
    end = timer()
    
    pLogging.info(logger_idx, f'Sample generation finished in {end - start} seconds.')