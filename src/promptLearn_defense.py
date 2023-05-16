"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
SPDX-License-Identifier: CC-BY-NC-4.0
"""

import torch
import uuid
import numpy as np
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import argparse
from torch.utils.tensorboard import SummaryWriter
import my_utils as ut
from sklearn.model_selection import train_test_split
from accelerate.utils import broadcast
import logging
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description='Tune a soft-prompt, generate sequences by appending it to given prompts')
    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs to train the soft-prompt')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_beams', type=int, default=1, help='beam size for beam decoding')
    parser.add_argument('--len_prompt', type=int, default=5, help='size of the soft prompt in number of tokens')
    parser.add_argument('--aligned', type=int, default=1, help='compute loss only over suffix if set')
    parser.add_argument('--theta', type=float, default=2, help='training loss threshold')
    parser.add_argument('--prefix_size', type=int, default=50, help='size of prefix we provide to the model')
    parser.add_argument('--suffix_size', type=int, default=50, help='size of suffix we generate from the model')
    parser.add_argument('--test_set_size', type=int, default=1000, help='size of the evaluation dataset')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'], \
        help='indicate which of 125M-1.3B-2.7B (small-medium-large) models to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU ID')
    args = parser.parse_args()
    accelerator = Accelerator(mixed_precision='fp16')


    # prepare datasets & dataloaders
    DATASET_PATH = '../datasets'
    prefixes =  np.concatenate((ut.load_prompts(f'{DATASET_PATH}/train_preprefix.npy'),\
        ut.load_prompts(f'{DATASET_PATH}/train_prefix.npy')), axis=1)[:, -args.prefix_size:]
    suffixes = ut.load_prompts(f'{DATASET_PATH}/train_suffix.npy')[:, :args.suffix_size]


    # sample a random training/test set
    prefix_tr, prefix_test, suffix_tr, suffix_test = train_test_split(prefixes, suffixes, test_size=args.test_set_size)
    # or use last 1k samples for deterministic evaluation
    # prefix_tr, suffix_tr = prefixes[:-args.test_set_size], suffixes[:-args.test_set_size]
    # prefix_test, suffix_test = prefixes[-args.test_set_size:], suffixes[-args.test_set_size:]

    # prepending 50256 (eos token) to make multi-token soft-prompt learning work
    train_ds = torch.cat([torch.full((len(prefix_tr), args.len_prompt), 50256),\
        torch.tensor(prefix_tr, dtype=torch.int64), torch.tensor(suffix_tr, dtype=torch.int64)], dim=1)
    test_ds = torch.cat([torch.full((len(prefix_test), args.len_prompt), 50256),\
        torch.tensor(prefix_test, dtype=torch.int64), torch.tensor(suffix_test, dtype=torch.int64)], dim=1)
    # make sure all GPUs see the same split, which is what main process (GPU ID 0) has sampled
    train_ds = broadcast(train_ds.cuda(), from_process=0) 
    test_ds = broadcast(test_ds.cuda(), from_process=0) 
    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs)

    # this is to measure ppl for defense experiments.. samples coming from the test set of the Pile
    ppl_ds = ut.load_prompts('../pile_test/pile_test_ppl.npy')
    np.random.shuffle(ppl_ds)
    ppl_ds = torch.tensor(ppl_ds[:args.test_set_size], dtype=torch.int64)
    ppl_ds = broadcast(ppl_ds.cuda(), from_process=0) 
    ppl_loader = DataLoader(ppl_ds, batch_size=args.bs)

    # model & tokenizer
    if args.model_size == 'small':
        MODEL_PATH = 'EleutherAI/gpt-neo-125M'
    elif args.model_size == 'medium':
        MODEL_PATH = 'EleutherAI/gpt-neo-1.3B'
    else:
        MODEL_PATH = 'EleutherAI/gpt-neo-2.7B'
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    
    # freeze model params and add soft-prompting "layer"
    for p in model.parameters():
        p.requires_grad=False
    soft_prompt = ut.SoftEmbedding(model.get_input_embeddings(), n_tokens=args.len_prompt, initialize_from_vocab=True)
    model.set_input_embeddings(soft_prompt)

    optimizer = torch.optim.AdamW(params=[soft_prompt.learned_embedding], lr=5e-4, weight_decay=0)
    # accelerator version of things
    model, optimizer, train_loader, test_loader, ppl_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader, ppl_loader
    )

    # creating tensorboard logger
    if accelerator.is_main_process:
        file_name = f"""promptLearnAttack_id:{uuid.uuid1().hex}_lenPrompt:{args.len_prompt}_nEpochs:{args.num_epochs}_aligned:{args.aligned}"""\
            + f"""_theta:{args.theta}_prefixSize:{args.prefix_size}_suffixSize:{args.suffix_size}_modelSize:{args.model_size}_numBeams:{args.num_beams}"""
        writer = SummaryWriter('../logs/' + file_name)

    # training the prompt
    for ep in range(args.num_epochs):
        model.train()
        tr_loss = []
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                if args.aligned:
                    labels = torch.clone(batch)
                    # predicting only the last args.suffix_size tokens
                    # so ignore everything else in loss calculation
                    labels[:, :labels.shape[1]-args.suffix_size] = -100
                else:
                    labels=batch
            outputs = model(input_ids=batch, labels=labels)
            # do grad ascent by negating loss if we're below the threshold
            if outputs.loss <= args.theta:
                accelerator.backward(-outputs.loss)
            else:
                accelerator.backward(outputs.loss)
            optimizer.step()
            tr_loss.append(accelerator.gather(outputs.loss*len(batch)).cpu())
        with torch.inference_mode():
            tr_loss = tr_loss[:len(train_loader.dataset)]
            tr_loss = (torch.sum(torch.cat(tr_loss)) / len(train_loader.dataset)).item()
            tr_plp = np.exp(tr_loss)
            test_loss = ut.evaluate_distributed(model, test_loader, args, accelerator)
            test_plp = np.exp(test_loss)
            if accelerator.is_main_process:
                writer.add_scalar('Train/Loss', tr_loss, ep)
                writer.add_scalar('Train/PLP', tr_plp, ep)
                writer.add_scalar('Test/Loss', test_loss, ep)
                writer.add_scalar('Test/PLP', test_plp, ep)
                accelerator.print(f'EP:{ep+1} Tr. Loss/PLP:{tr_loss:.3f}/{tr_plp:.3f}', end=' --- ')
                accelerator.print(f'Test Loss/PLP:{test_loss:.3f}/{test_plp:.3f}', end='\r')
            # if training loss has stablized around theta, finish training
            if tr_loss >= args.theta:
                break
            
    # generate suffixes
    generations_test = ut.generate_suffixes_distributed(model, test_loader, args, accelerator, use_cache=False)
    generations_test = np.stack(generations_test, axis=0)
    # always measure the final loss over suffix tokens
    args.aligned = True
    test_loss = ut.evaluate_distributed(model, ppl_loader, args, accelerator)

    # log results
    if accelerator.is_main_process:
        # measure  fractional and exact match rates
        fract_rate, exact_rate = ut.compute_reconstruct_rate(generations_test, suffix_test, args)
        accelerator.print(f'Exact/Fract extract rate:{exact_rate:.3f}/{fract_rate:.3f}')
        test_plp = np.exp(test_loss)
        accelerator.print(f'Test Loss/PLP:{test_loss:.3f}/{test_plp:.3f}')
        # log the results
        writer.add_scalar('Memorization/Fract_Rate', fract_rate, 0)
        writer.add_scalar('Memorization/Exact_Rate', exact_rate, 0)
        writer.add_scalar('Test_Final/Loss', test_loss, 0)
        writer.add_scalar('Test_Final/PLP', np.exp(test_loss), 0)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
