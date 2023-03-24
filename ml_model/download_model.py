from pathlib import Path
import torch
import gdown
from transformers import AutoTokenizer, BertForSequenceClassification


def download_model():
    url = 'https://drive.google.com/drive/folders/1nSTLrmkjeNzPNqMQpdzcrAM_avWimfLJ?usp=share_link'
    folder_name = "ml_model/bertforsequnceclassification/"
    model_name = "bert-base-uncased"

    if not Path(folder_name).exists():
        gdown.download_folder(url, output=folder_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(folder_name)

    return tokenizer, model
