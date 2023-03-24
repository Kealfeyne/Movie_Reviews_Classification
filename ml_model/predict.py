import torch
from .download_model import download_model

ml_model_name = "bert-base-uncased"

tokenizer, model = download_model()

# label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
label_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10}


def predict(content):
    inputs = tokenizer(content, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = label_mapping[logits.argmax().item()]
    predicted_binary_class = 'negative' if predicted_class < 5 else 'positive'

    return str(predicted_class), str(predicted_binary_class)
