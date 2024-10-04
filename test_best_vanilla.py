from src.translator import Translator
import torch
import random
from src.helper import get_data

random.seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "embed_size": 16,
    "hidden_size": 512,
    "cell_type": "LSTM",
    "num_layers": 2,
    "dropout": 0.1,
    "learning_rate": 0.005,
    "optimizer": "SGD",
    "teacher_forcing_ratio": 0.5,
    "max_length": 50
}

model = Translator("tam", params, device)

model.encoder.load_state_dict(torch.load("./best_model_vanilla/encoder.pt"))
model.decoder.load_state_dict(torch.load("./best_model_vanilla/decoder.pt"))

with open("test_gen.txt", "w") as f:
    test_data = get_data("tam", "test")
    f.write("Input, Target, Output\n")
    accuracy = 0
    for i in range(len(test_data)):
        f.write(test_data[i][0] + ", " + test_data[i][1] + ", " + model.evaluate(test_data[i][0]) + "\n")
        if test_data[i][1] == model.evaluate(test_data[i][0]):
            accuracy += 1

    print("Test Accuracy: " + str(accuracy/len(test_data) * 100) + "%")