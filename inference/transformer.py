import torch
import pickle
import sys
import os
from inference.language import Language
from inference.utility import Encoder, Decoder, encoderBlock, decoderBlock, MultiHeadAttention, Head, FeedForward
import warnings
from typing import List
warnings.filterwarnings("ignore", category=FutureWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(os.path.dirname(__file__), 'input_lang.pkl'), "rb") as file:
    input_lang = pickle.load(file)

with open(os.path.join(os.path.dirname(__file__), 'output_lang.pkl'), "rb") as file:
    output_lang = pickle.load(file)

encoder = torch.load(os.path.join(os.path.dirname(__file__), 'encoder (1).pth'), map_location=device)
decoder = torch.load(os.path.join(os.path.dirname(__file__), 'decoder (1).pth'), map_location=device)

input_vocab_size = input_lang.vocab_size
output_vocab_size = output_lang.vocab_size

def encode(s):
    return [input_lang.char2index.get(ch, input_lang.char2index['$']) for ch in s]

def generate(input: List[str]) -> List[str]:
    # pre-process the input: same length and max_length = 33
    for i, inp in enumerate(input):
        input[i] = input[i][:33] if len(input[i]) > 33 else input[i].ljust(33, '#')

    input = torch.tensor([encode(i) for i in input], device=device, dtype=torch.long)
    B, T = input.shape

    encoder_output = encoder(input)
    idx = torch.full((B, 1), 2, dtype=torch.long, device=device) # (B,1)

    # idx is (B, T) array of indices in the current context
    for _ in range(30):
        # get the predictions
        logits, loss = decoder(idx, encoder_output) # logits (B, T, vocab_size)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    ans = []
    for id in idx:
        ans.append(output_lang.decode(id.tolist()[1:]).split('#', 1)[0])
    return ans