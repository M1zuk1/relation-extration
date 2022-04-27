# R-Bert relation extraction

## Install The Requirements
`pip install -r requirements.txt`

## Prepare Pre-Train Model And Data
Step1：Download pytorch_model.bin from https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
and put it into ./resource/bert-base-uncased

Step2：Go to ./resource/data and run `python process.py`. The 
processed data are stored in ./resource/data 
## Train Model
`python run.py | tee -a Train.txt` 
Train.txt stored the train info

## Evaluate Model
Go to ./eval `./semeval2010_task8_scorer-v1.2.pl proposed_answer.txt predicted_result.txt`
