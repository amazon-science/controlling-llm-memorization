"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import glob
import os
import numpy as np
import csv

event_files = defaultdict(list)
directory = '../logs'
log_folders = [x[0] for x in os.walk(directory)][1:]

for folder in log_folders:
    param_conf = tuple(folder.split('/')[-1].split('_')[2:])
    event_file = glob.glob(folder)[0]
    event_files[param_conf].append(event_file)

for k, v in event_files.items():
    metrics = defaultdict(list)
    for log in v:
        ea = event_accumulator.EventAccumulator(log, size_guidance={ event_accumulator.SCALARS: 0,})
        ea.Reload()
        for tag in tags:
            try:
                v = ea.Scalars(tag)
                value = v[-1].value
                metrics[tag].append(value)
            except:
                #print('err')
                print(log)
                #continue
            
    for k2, v2 in metrics.items():
        metrics[k2] = [round(np.mean(v2), 3), round(np.std(v2), 3)]
    filename = f"{directory}/processsed_logs/{k}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(str(metrics).replace(',', ' ±'))
