# BERT-RoBERTa Training Time, Score and Cost Comparison

## Project Description: 
This project compares the performance of transformer models, namely BERT and RoBERTa on GLUE (COLA) and SQuAD dataset and to compare cost on different GPUs.
This is done by running the code on different GPUs namely A100 and V100.

## Repo Description:
glue.py has the code to finetune BERT and RoBERTa on GLUE task (COLA)
squad.py has the code to finetune BERT and RoBERTa on SQuAD
plots.py has the code to plot graphs using the data brought by glue and squad
COLA Plots/ has all COLA related plots
SQuAD Plots/ has all SQuAD related plots

## Usage:
### For COLA Dataset:
mkdir GLUE
python glue.py
(NOTE: The code automatically downloads and unzips the dataset. The code is running bert and roberta.)

### For SQuAD Dataset:
To download dataset 
curl "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json" >> train-v1.1.json
curl "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" >> dev-v1.1.json

To run BERT:
python squad.py --model_type=bert --model_name_or_path=bert-base-uncased --output_dir='./output' --train_file='./train-v1.1.json' --predict_file='./dev-v1.1.json'  --do_train --do_eval --do_lower_case --logging_steps=1000 --save_steps=1000 --verbose_logging --evaluate_during_training --overwrite_output_dir --threads=12

To run RoBERTa:
python squad.py --model_type=roberta --model_name_or_path=roberta-base --output_dir='./output_roberta' --train_file='./train-v1.1.json' --predict_file='./dev-v1.1.json'  --do_train --do_eval --do_lower_case --logging_steps=1000 --save_steps=1000 --verbose_logging --evaluate_during_training --overwrite_output_dir --threads=12

The information is stored in output/runs/ and we can use 
from tensorflow.python.summary.summary_iterator import summary_iterator

to get the info and process it using plots.py

## Results:
The plots are in COLA Plots/ and SquAD Plots/
We observe that it is more economical to fine-tune on A100 GPU as compared to V100 GPU.
- COLA Train-time Comparison
![](https://github.com/patodiayogesh/Bert-RoBerta-Comparison/blob/main/COLA%20Plots/v100_plots/bert_roberta_train_time.png)

- SQUAD Train-time Comparison
![](https://github.com/patodiayogesh/Bert-RoBerta-Comparison/blob/main/SquAD%20Plots/bert_roberta_train_time.png)

