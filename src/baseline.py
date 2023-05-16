"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
SPDX-License-Identifier: CC-BY-NC-4.0
"""

import numpy as np
import transformers
import torch
import argparse
import uuid
from tqdm import tqdm
import my_utils as ut
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from accelerate.utils import broadcast
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='Generate sequences with greedy sampling given context from Google challenge')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_beams', type=int, default=1, help='beam size for beam decoding')
    parser.add_argument('--prefix_size', type=int, default=50, help='size of prefix we provide to the model')
    parser.add_argument('--suffix_size', type=int, default=50, help='size of suffix we generate from the model')
    parser.add_argument('--aligned', type=int, default=1, help='compute loss only over suffix if set')
    parser.add_argument('--test_set_size', type=int, default=1000, help='test set size')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large', 'gpt2', 'gpt2XL'], \
        help='indicate which of GPT-NEO 125M-1.3B-2.7B (small-medium-large) or GPT2 models to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU ID')
    parser.add_argument('--train_preprefix', type=str, default='../datasets/train_preprefix.npy', help="path to binary train_preprefix file")
    parser.add_argument('--train_prefix', type=str, default='../datasets/train_prefix.npy', help="path to binary train_prefix file")
    parser.add_argument('--train_suffix', type=str, default='../datasets/train_suffix.npy', help="path to binary train_suffix file")
    parser.add_argument('--test_prefix', type=str, default='../datasets/pile_test_ppl.npy', help="path to binary test_prefix file")
    args = parser.parse_args()
    accelerator = Accelerator(mixed_precision='fp16')


    # load datasets
    DATASET_PATH = '../datasets'
    prefixes = np.concatenate((ut.load_prompts(args.train_preprefix), \
        ut.load_prompts(args.train_prefix)), axis=1)[:, -args.prefix_size:]
    suffixes = ut.load_prompts(args.train_suffix)[:, :args.suffix_size]


    # sample a random test set
    _, prefix_test, _, suffix_test = train_test_split(prefixes, suffixes, test_size=args.test_set_size)
    # or use last 1k samples for deterministic evaluation
    # prefix_test, suffix_test = prefixes[-args.test_set_size:], suffixes[-args.test_set_size:]
    
    # create dataloader
    test_ds = torch.cat([torch.tensor(prefix_test, dtype=torch.int64), torch.tensor(suffix_test, dtype=torch.int64)], dim=1)
    # make sure all GPUs see the same split, which is what main process (GPU ID 0) has sampled
    test_ds = broadcast(test_ds.cuda(), from_process=0) 
    test_loader = DataLoader(test_ds, batch_size=args.bs)

    # samples coming from the test set of the Pile, this is to measure ppl for defense experiments
    ppl_ds = ut.load_prompts(args.test_prefix)
    np.random.shuffle(ppl_ds)
    ppl_ds = torch.tensor(ppl_ds[:args.test_set_size], dtype=torch.int64)
    ppl_ds = broadcast(ppl_ds.cuda(), from_process=0) 
    ppl_loader = DataLoader(ppl_ds, batch_size=args.bs)

    # load model
    if args.model_size == 'small':
        MODEL_PATH = 'EleutherAI/gpt-neo-125M'
    elif args.model_size == 'medium':
        MODEL_PATH = 'EleutherAI/gpt-neo-1.3B'
    elif args.model_size == 'large':
        MODEL_PATH = 'EleutherAI/gpt-neo-2.7B'
    elif args.model_size == 'gpt2':
        MODEL_PATH = 'gpt2'
    else:
        MODEL_PATH = 'gpt2-xl'
    
    accelerator.print('Loading model..')
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    # optimizer is just a placeholder (accelerator/deepspeed requires it for some reason)
    # we don't do any training in baseline attack
    optimizer = torch.optim.AdamW(params=model.parameters())
    model, optimizer, test_loader, ppl_loader = accelerator.prepare(model, optimizer, test_loader, ppl_loader)
    
    accelerator.print('Generating suffixes..')
    generations_test = ut.generate_suffixes_distributed(model, test_loader, args, accelerator)
    generations_test = np.stack(generations_test, axis=0)

    test_loss = ut.evaluate_distributed(model, ppl_loader, args, accelerator)
    # use this if you want to compute ppl wrt to prompt test data
    # test_loss = ut.evaluate_distributed(model, test_loader, args, accelerator)
    

    if accelerator.is_main_process:
        # measure  fractional and exact match rates
        fract_rate, exact_rate = ut.compute_reconstruct_rate(generations_test, suffix_test, args)
        accelerator.print(f'Exact/Fract extract rate:{exact_rate:.3f}/{fract_rate:.3f}')
        test_plp = np.exp(test_loss)
        accelerator.print(f'Test Loss/PLP:{test_loss:.3f}/{test_plp:.3f}')
        #log the results
        file_name = f"""baseline_id:{uuid.uuid1().hex}_modelSize:{args.model_size}_prefixSize:{args.prefix_size}"""\
                    + f"""_suffixSize:{args.suffix_size}_numBeams:{args.num_beams}"""
        writer = SummaryWriter('../logs/' + file_name)
        writer.add_scalar('Memorization/Fract_Rate', fract_rate, 0)
        writer.add_scalar('Memorization/Exact_Rate', exact_rate, 0)
        writer.add_scalar('Test_Final/Loss', test_loss, 0)
        writer.add_scalar('Test_Final/PLP', np.exp(test_loss), 0)
        writer.flush()
        writer.close()
        


if __name__ == "__main__":
    main()
