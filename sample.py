"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
from timeit import default_timer as timer

import model
from model import GPTConfig, GPT
import pLogging

script_dir = os.path.dirname(os.path.abspath(__file__))

logger_idx = pLogging.create_logger('sample')
model.set_logger(logger_idx)
pLogging.info(logger_idx, 'Sample generation started.')

# -----------------------------------------------------------------------------
dataset = ""
start = ""
num_samples = 40
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

# Ensure the output directory exists
out_dir = os.path.join(script_dir, "trained_models", dataset)
os.makedirs(out_dir, exist_ok=True)

pLogging.info(logger_idx, "Sampling info", {
    "out_dir": out_dir,
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
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
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

# encode the beginning of the prompt
start_ids = [int(x) for x in start.split()]
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Generate samples
with torch.no_grad(), ctx, open('output.txt', 'a') as f:
    pLogging.info(logger_idx, 'Generating samples.')
    
    # For curiosity, I am timing the sample generation
    start = timer()
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        f.write(str(y[0].tolist()).strip('[]').replace(',', '') + '\n')
    end = timer()
    pLogging.info(logger_idx, f'Sample generation finished in {end - start} seconds.')