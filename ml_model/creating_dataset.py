import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_folder(path_to_folder: str) -> dict:
    dictionary = {'id': [], 'content': [], 'label': []}

    for file in tqdm(list(os.listdir(path_to_folder))):
        try:
            with open(path_to_folder + file) as f:
                content = f.read()

            dictionary['id'].append(file.split('_')[0])
            dictionary['content'].append(content)
            dictionary['label'].append(file.split('_')[1].split('.')[0])
        except:
            pass

    return dictionary


def create_dataset(path_to_data: str = "../data/") -> (pd.DataFrame, pd.DataFrame):
    train = pd.concat((pd.DataFrame(parse_folder(path_to_data + "train/neg/")),
                           pd.DataFrame(parse_folder(path_to_data + "train/pos/")))).reset_index(drop=True)

    train.to_csv(path_to_data + "train.csv", index=True)

    test = pd.concat((pd.DataFrame(parse_folder(path_to_data + "test/neg/")),
                           pd.DataFrame(parse_folder(path_to_data + "test/pos/")))).reset_index(drop=True)

    test.to_csv(path_to_data + "test.csv", index=True)

    return train, test


create_dataset()
