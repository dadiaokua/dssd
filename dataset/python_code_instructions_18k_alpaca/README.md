---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: prompt
    dtype: string
  splits:
  - name: train
    num_bytes: 25180782
    num_examples: 18612
  download_size: 11357076
  dataset_size: 25180782
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- question-answering
- text2text-generation
- text-generation
tags:
- code
size_categories:
- 10K<n<100K
---

# Dataset Card for python_code_instructions_18k_alpaca

The dataset contains problem descriptions and code in python language.
This dataset is taken from [sahil2801/code_instructions_120k](https://huggingface.co/datasets/sahil2801/code_instructions_120k), which adds a prompt column in alpaca style. Refer to the source [here](https://huggingface.co/datasets/sahil2801/code_instructions_120k).