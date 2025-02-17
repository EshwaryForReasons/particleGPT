"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
from timeit import default_timer as timer
import datetime as dt
import subprocess
import sys

import model
from model import GPTConfig, GPT
import pLogging

script_dir = os.path.dirname(os.path.abspath(__file__))

logger_idx = pLogging.create_logger('sample')
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Sample generation started.')

# -----------------------------------------------------------------------------
dataset = ""
output_dir_name = ""
start = ""
num_samples = 40
max_new_tokens = 500
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

now = dt.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
model_path = os.path.join(script_dir, "trained_models", output_dir_name, 'ckpt.pt')
samples_leading_input_filename = os.path.join('data', dataset, 'outputs', 'temp_sampling_lead.csv')
samples_output_filename = os.path.join(script_dir, "generated_samples", output_dir_name, now, 'generated_samples.txt')

# Prepare the data before sampling it
res = subprocess.run([sys.executable, os.path.join('data', 'prepare_sampling.py'), dataset], capture_output=True, text=True)
print(res.stderr, res.stdout)

if not os.path.exists(model_path):
    pLogging.info(logger_idx, "Cound not find trained model!")
    exit()
if not os.path.exists(samples_leading_input_filename):
    pLogging.info(logger_idx, "Cound not find leading tokens!")
    exit()

# Ensure the output directory of the samples exists
os.makedirs(os.path.dirname(os.path.join(script_dir, samples_output_filename)), exist_ok=True)

pLogging.info(logger_idx, "Sampling info", {
    "dataset": dataset,
    "model_path": model_path,
    "samples_output_filename": samples_output_filename,
    "start": start,
    "num_samples": num_samples,
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_k": top_k,
    "seed": seed,
    "device": device,
    "dtype": dtype,
    "compile": compile
})

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
checkpoint = torch.load(model_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# Generate samples
with torch.no_grad(), ctx, open(samples_output_filename, 'a') as out_file, open(samples_leading_input_filename, 'r') as in_file:
    pLogging.info(logger_idx, 'Generating samples.')
    
    start = timer()
    for sample_lead in in_file:
        # encode the beginning of the prompt
        start_ids = [int(x) for x in sample_lead.split()]
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        out_file.write(str(y[0].tolist()).strip('[]').replace(',', '') + '\n')
    end = timer()
    
    pLogging.info(logger_idx, f'Sample generation finished in {end - start} seconds.')