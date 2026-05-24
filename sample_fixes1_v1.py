"""
Sample from a trained model
"""
import json

import torch
import pickle
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from timeit import default_timer as timer

import pLogging
import pUtil
import configurator as conf
import model_fixes1_v1 as model_module
from model_fixes1_v1 import GPTConfig, GPT

from dictionary import Dictionary

script_dir = Path(__file__).resolve().parent

# sampling_ids are consecutive so we need to determine the next one (for us)
sampling_id = pUtil.get_latest_sampling_id(conf.generic.model_name) + 1

prep_info_filepath = script_dir / 'preparations' / conf.generic.preparation_name / 'preparation_info.json'
prep_data_filepath = script_dir / 'preparations' / conf.generic.preparation_name / 'preparation.json'

test_bin_filename       = script_dir / 'preparations' / conf.generic.preparation_name / 'test.bin'
model_filename          = pUtil.get_training_dir(conf.generic.model_name) / 'ckpt.pt'
samples_output_filename = pUtil.get_sampling_dir(conf.generic.model_name) / f'sampling_{sampling_id}' / 'generated_samples.csv'

logger_idx = -1
dictionary = None
model = None
ctx = None

# Sets up torch environment and creates and returns a compiled model.
def initialize_model_and_ctx():
    global model
    global ctx
    
    torch.manual_seed(conf.sampling.seed)
    torch.cuda.manual_seed(conf.sampling.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in conf.sampling.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[conf.sampling.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    # init from a model saved in a specific directory
    checkpoint = torch.load(model_filename, map_location=conf.sampling.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(conf.sampling.device)
    if conf.sampling.compile:
        model = torch.compile(model)

# # Exists in case we need to sample on CPU or something.
# @torch.no_grad()
# def sample_standard(sampling_starters, max_new_tokens):
#     with open(samples_output_filename, 'a') as out_file:
#         for starting_tokens in sampling_starters:
#             x = (torch.tensor(starting_tokens.tolist(), dtype=torch.long, device=conf.sampling.device)[None, ...])
#             y = model.generate(x, max_new_tokens, temperature=conf.sampling.temperature, top_k=conf.sampling.top_k)
#             out_file.write(" ".join(map(str, y[0].tolist())) + "\n")
            
@torch.no_grad()
def sample_standard(sampling_starters, max_new_tokens):
    grammar_mask = getattr(conf.sampling, "grammar_mask", False)

    with open(samples_output_filename, "a") as out_file:
        for starting_tokens in sampling_starters:
            x = torch.tensor(
                starting_tokens.tolist(),
                dtype=torch.long,
                device=conf.sampling.device,
            )[None, ...]

            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=conf.sampling.temperature,
                top_k=conf.sampling.top_k,
                grammar_mask=grammar_mask,
            )

            out_file.write(" ".join(map(str, y[0].tolist())) + "\n")

# @torch.no_grad()
# def sample_batched(sampling_starters, max_new_tokens):
#     # Batched sampling only makes sense on GPUs supporting CUDA.
#     if not torch.cuda.is_available():
#         raise RuntimeError('Batched Multi-GPU sampling is only supported on CUDA devices.')
            
#     # Prepare starters. Do not assign device. generate_batched_multiGPU will assign
#     # appropriate split to appropriate device.
#     all_starters = torch.tensor(np.array(sampling_starters), dtype=torch.long)
#     generated = model.generate_batched_multiGPU(
#         starters=all_starters,
#         max_new_tokens=max_new_tokens,
#         temperature=conf.sampling.temperature,
#         top_k=conf.sampling.top_k,
#         batch_size=conf.sampling.batch_size
#     )

#     # Batch sampling outputs into padded tensor. This removes padding and outputs data.
#     with open(samples_output_filename, 'a') as out_file:
#         for seq in generated:
#             seq_list = seq.tolist()
#             if dictionary.event_end_token in seq_list:
#                 seq_list = seq_list[:seq_list.index(dictionary.event_end_token) + 1]
#             out_file.write(" ".join(map(str, seq_list)) + "\n")

@torch.no_grad()
def sample_batched(sampling_starters, max_new_tokens):
    if not torch.cuda.is_available():
        raise RuntimeError("Batched sampling is only supported on CUDA devices.")

    grammar_mask = getattr(conf.sampling, "grammar_mask", False)

    all_starters = torch.tensor(np.array(sampling_starters), dtype=torch.long)

    generated = model.generate_batched_multiGPU(
        starters=all_starters,
        max_new_tokens=max_new_tokens,
        temperature=conf.sampling.temperature,
        top_k=conf.sampling.top_k,
        batch_size=conf.sampling.batch_size,
        grammar_mask=grammar_mask,
    )

    with open(samples_output_filename, "a") as out_file:
        for seq in generated:
            seq_list = seq.tolist()

            if dictionary.event_end_token in seq_list:
                seq_list = seq_list[: seq_list.index(dictionary.event_end_token) + 1]

            out_file.write(" ".join(map(str, seq_list)) + "\n")

# Model and context need to be initialized for master and child processes.
# Since this does not need anything except the filenames vars defined atop, we can run this
# before initializing the logger and whatnot.
initialize_model_and_ctx()

# Everything in this block so it is not run in child processes.
if __name__ == "__main__":
    # Ensure the output directory of the samples exists. This needs to be done before the logger is created.
    Path(samples_output_filename).parent.mkdir(parents=True, exist_ok=True)

    logger_idx = pLogging.create_sampling_logger(conf.generic.model_name, sampling_id)
    model_module.set_logger(logger_idx)
    
    if not prep_data_filepath.exists():
        pLogging.info(logger_idx, "Preparation data not found!")
        exit()
    if not prep_info_filepath.exists():
        pLogging.info(logger_idx, "Preparation info not found!")
        exit()
    if not model_filename.exists():
        pLogging.info(logger_idx, "Could not find trained model!")
        exit()
        
    try:
        with open(prep_data_filepath, 'r') as f:
            prep_data = json.load(f)
        prop_tokenized_dataset_name = prep_data['train_bin']['tokenized_dataset']
        
        dictionary_filepath = script_dir / 'data' / 'tokenized' / prop_tokenized_dataset_name / 'dictionary.json'
        dictionary = Dictionary(dictionary_filepath)
    except Exception as e:
        pLogging.info(logger_idx, f"Error occurred while trying to load dictionary: {e}")
        raise Exception("Error occurred while trying to load dictionary!")

    pLogging.info(logger_idx, 'Sample generation started.')
    pLogging.info(logger_idx, "Sampling info", {
        "preparation": conf.generic.preparation_name,
        "model_path": model_filename,
        "samples_output_filename": samples_output_filename,
        "max_new_tokens": conf.sampling.max_new_tokens,
        "temperature": conf.sampling.temperature,
        "top_k": conf.sampling.top_k,
        "seed": conf.sampling.seed,
        "device": conf.sampling.device,
        "dtype": conf.sampling.dtype,
        "compile": conf.sampling.compile
    })

    with open(prep_info_filepath, 'r') as pif:
        pinf = json.load(pif)
        max_sequence_len = pinf['max_sequence_length']
        
    test_data = np.memmap(test_bin_filename, dtype=np.uint16, mode="r")
    test_data = test_data.reshape(-1, max_sequence_len)

    # 5 because EVENT_START PDGID PT ETA PHI (incident particle as prompt)
    sampling_starters = [event[:5] for event in test_data]

    starter_len = len(sampling_starters[0])
    max_new_tokens = max_sequence_len - starter_len

    pLogging.info(logger_idx, "Generating samples.")
    pLogging.info(logger_idx, "Sampling derived info", {
        "starter_len": starter_len,
        "effective_max_new_tokens": max_new_tokens,
        "grammar_mask": getattr(conf.sampling, "grammar_mask", False),
    })
    
    start = timer()
    with ctx:
        if conf.sampling.device == "cpu":
            sample_standard(sampling_starters, max_new_tokens=max_new_tokens)
        else:
            sample_batched(sampling_starters, max_new_tokens=max_new_tokens)
    end = timer()
    
    pLogging.info(logger_idx, f'Sample generation finished in {end - start} seconds.', {'time_seconds': end - start})
        
    # test_data = np.memmap(test_bin_filename, dtype=np.uint16, mode='r')
    # test_data = test_data.reshape(-1, max_sequence_len)
    # sampling_starters = [event[:5] for event in test_data]
        
    # pLogging.info(logger_idx, 'Generating samples.')
    
    # start = timer()
    # with ctx:
    #     if conf.sampling.device == 'cpu':
    #         sample_standard(sampling_starters, max_new_tokens=max_sequence_len)
    #     else:
    #         sample_batched(sampling_starters, max_new_tokens=max_sequence_len)
    # end = timer()
        
    # pLogging.info(logger_idx, f'Sample generation finished in {end - start} seconds.')