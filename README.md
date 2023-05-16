
# Controlling LLM memorization

## Setting up environment
First setup your conda environment by running `conda env create -f environment.yml`.
You also need to configure [HF accelerate](https://huggingface.co/docs/accelerate/index).
The configuration used in final experiments is in `accelerate_config.yaml`. You can simply
put this file under the cache folder for Accelerate (`~/.cache/huggingface/accelerate`).
This configuration specifies distributed training over 8 GPUs with fp16 mixed-precision and [DeepSpeed](https://github.com/microsoft/DeepSpeed).
You can also configure Accelerate differently for your needs (e.g., increase/decrease # of GPUs)
with `accelerate config` command and answering the prompted questions in terminal.


## Dataset
The code reads train and test data in numpy binary format from a folder `datasets`, which needs to be created. In this section, we describe how to obtain the data used in the paper.

### Train split
We used the train split from the Pile dataset for training and evaluating our method.
You can use the script `load_dataset.py` from the [LM-extraction-benchmark](https://github.com/google-research/lm-extraction-benchmark) to extract the 15000 examples used in the paper. The script converts the train split of the Pile data (also available within the repo above) into numpy binary files, then used by our code. The resulting files should be saved in a folder named `datasets` within the top-level of the Controlling-LLM-memorization repo.

### Test split
The test split of the [Pile dataset](https://pile.eleuther.ai/) is used to compute perplexity of the models in order to assess performance degradation of the model. After downloading the data, you can use our `convert_test_pile.py` script to convert the test split into numpy binary file. This should also be saved inside the `datasets` folder.

## Models
HF will pull and cache the models on your first run.

## Replicating Results
Simply run `src/runner.sh` after uncommenting the desired experiment. For example, at the top of the script, we see commands for running
the baselines.
```bash
for i in {1..5}
do
    accelerate launch baseline.py --model_size=small 
    accelerate launch baseline.py --model_size=medium
done
```
Running this will create tensorboard files under a folder named `logs`. You can then process
these logs, to obtain mean and standard deviation values by running `scripts/tensorboard_log_read.py`.
This is going to create result files under `logs/processed_logs` where file names specify the experiment setting 
e.g., `modelSize:medium', '(prefixSize:50', 'suffixSize:50', 'numBeams:1').txt` contains results for the 1.3B model.


## Running extra experiments
Simply extend the runner script as desired. For example, if you want to run an aligned CLM attack
with `prompt length=200`, change the first line of the section below in `src/runner.sh` to `for len_prompt in 200`.
```bash
for len_prompt in 1 5 20 100 150
do
    for i in {1..5}
    do
        accelerate launch promptLearn_attack.py --model_size=small  --len_prompt=$len_prompt
        accelerate launch promptLearn_attack.py --model_size=medium  --len_prompt=$len_prompt
    done
done
```
You can pass prompt length, prefix size etc. as arguments. Run `-h` command on python scripts to get comprehensive list
of arguments (e.g., `python promptLearn_attack.py -h`) and see `src/runner.sh` for more usage.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.

## How to cite
The paper introducing the experiments conducted with thsi code is forthcoming, in the interim you can cite it as:
```
@article{ozdayi2023controlling,
  title={Controlling the Extraction of Memorized Data from Large Language Models via Prompt-Tuning},
  author={Ozdayi, Mustafa Safa and Peris, Charith and FitzGerald, Jack and Dupuy, Christophe and Majmudar, Jimit and Khan, Haidar  and Parikh, Rahil and Gupta, Rahul},
  journal={},
  year={2023}
}
```
