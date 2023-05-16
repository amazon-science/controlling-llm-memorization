"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
SPDX-License-Identifier: CC-BY-NC-4.0
"""

import os
import csv
import pickle
import multiprocessing as mp
import numpy as np
import json
from transformers import GPT2Tokenizer
import sys
import hashlib

# Example command: python sample_pileTest_data.py ../datasets/pile_test/00.jsonl ../datasets/pile_test_ppl.npy

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
infile=sys.argv[1]
outfile=sys.argv[2]

data = []
ctr = 0
with open(infile) as f:
    for line in f:
        tokens = tokenizer.encode(json.loads(line)['text'])
        if len(tokens) >= 100:
            data.append(tokens[:100])
            ctr += 1
            if ctr >= 5000:
                break

np.save(outfile, data)
