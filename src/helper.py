import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.language import Language, EOS_token

def get_data(lang: str, type: str) -> list[list[str]]:
    """
    Returns: 'pairs': list of [input_word, target_word] pairs
    """
    path = "./aksharantar_sampled/{}/{}_{}.csv".format(lang, lang, type)
    df = pd.read_csv(path, header=None)
    pairs = df.values.tolist()
    return pairs

def get_languages(lang: str):
    """
    Returns 
    1. input_lang: input language - English
    2. output_lang: output language - Given language
    3. pairs: list of [input_word, target_word] pairs
    """
    input_lang = Language('eng')
    output_lang = Language(lang)
    pairs = get_data(lang, "train")
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    return input_lang, output_lang, pairs

def get_cell(cell_type: str):
    if cell_type == "LSTM":
        return nn.LSTM
    elif cell_type == "GRU":
        return nn.GRU
    elif cell_type == "RNN":
        return nn.RNN
    else:
        raise Exception("Invalid cell type")
    
def get_optimizer(optimizer: str):
    if optimizer == "SGD":
        return optim.SGD
    elif optimizer == "ADAM":
        return optim.Adam
    else:
        raise Exception("Invalid optimizer")
    
def indexesFromWord(lang:Language, word:str):
    return [lang.word2index[char] for char in word]

def tensorFromWord(lang:Language, word:str, device:str):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang:Language, output_lang:Language, pair:list[str], device:str):
    input_tensor = tensorFromWord(input_lang, pair[0], device)
    target_tensor = tensorFromWord(output_lang, pair[1], device)
    return (input_tensor, target_tensor)