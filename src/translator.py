import torch
import torch.nn as nn
import torch.nn.functional as F
from src.helper import get_optimizer, tensorsFromPair, get_languages, tensorFromWord, get_data
from src.language import SOS_token, EOS_token
from src.encoder import Encoder
from src.decoder import Decoder
import random
import time
import numpy as np

PRINT_EVERY = 5000
PLOT_EVERY = 100

class Translator:
    def __init__(self, lang: str, params: dict, device: str):
        self.lang = lang
        self.input_lang, self.output_lang, self.pairs = get_languages(self.lang)
        self.input_size = self.input_lang.n_chars
        self.output_size = self.output_lang.n_chars
        self.device = device

        self.training_pairs = [tensorsFromPair(self.input_lang, self.output_lang, pair, self.device) for pair in self.pairs]

        self.encoder = Encoder(in_sz = self.input_size,
                             embed_sz = params["embed_size"],
                             hidden_sz = params["hidden_size"],
                             cell_type = params["cell_type"],
                             n_layers = params["num_layers"],
                             dropout = params["dropout"],
                             device=self.device).to(self.device)
        
        self.decoder = Decoder(out_sz = self.output_size,
                             embed_sz = params["embed_size"],
                             hidden_sz = params["hidden_size"],
                             cell_type = params["cell_type"],
                             n_layers = params["num_layers"],
                             dropout = params["dropout"],
                             device=self.device).to(self.device)

        self.encoder_optimizer = get_optimizer(params["optimizer"])(self.encoder.parameters(), lr=params["learning_rate"])
        self.decoder_optimizer = get_optimizer(params["optimizer"])(self.decoder.parameters(), lr=params["learning_rate"])
        
        self.criterion = nn.NLLLoss()

        self.teacher_forcing_ratio = params["teacher_forcing_ratio"]
        self.max_length = params["max_length"]

    def train_single(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_cell = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_sz, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_cell = self.encoder(input_tensor[ei], encoder_hidden, encoder_cell)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=self.device)
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                loss += self.criterion(decoder_output, target_tensor[di])

                decoder_input = target_tensor[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                loss += self.criterion(decoder_output, target_tensor[di])

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length
    
    def train(self, iters=-1):
        start_time = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        random.shuffle(self.training_pairs)
        iters = len(self.training_pairs) if iters == -1 else iters

        for iter in range(1, iters+1):
            training_pair = self.training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train_single(input_tensor, target_tensor)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % PRINT_EVERY == 0:
                print_loss_avg = print_loss_total / PRINT_EVERY
                print_loss_total = 0
                current_time = time.time()
                print("Loss: {:.4f} | Iterations: {} | Time: {:.3f}".format(print_loss_avg, iter, current_time - start_time))

            if iter % PLOT_EVERY == 0:
                plot_loss_avg = plot_loss_total / PLOT_EVERY
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            
        return plot_losses
    
    def evaluate(self, word):
        with torch.no_grad():
            input_tensor = tensorFromWord(self.input_lang, word, self.device)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()
            encoder_cell = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_sz, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden, encoder_cell = self.encoder(input_tensor[ei], encoder_hidden, encoder_cell)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)
            decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

            decoded_chars = ""

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                topv, topi = decoder_output.topk(1)
                
                if topi.item() == EOS_token:
                    break
                else:
                    decoded_chars += self.output_lang.index2word[topi.item()]

                decoder_input = topi.squeeze().detach()

            return decoded_chars
        
    def test_validate(self, type:str):
        pairs = get_data(self.lang, type)
        accuracy = np.sum([self.evaluate(pair[0]) == pair[1] for pair in pairs])
        return accuracy / len(pairs)