from src.translator import Translator
import torch
import random
import argparse

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

language = "tam"

# Argument Parser
parser = argparse.ArgumentParser(description="Transliteration Model")
parser.add_argument("-es", "--embed_size", type=int, default=16, help="Embedding Size, good_choices = [8, 16, 32]")
parser.add_argument("-hs", "--hidden_size", type=int, default=512, help="Hidden Size, good_choices = [128, 256, 512]")
parser.add_argument("-ct", "--cell_type", type=str, default="LSTM", help="Cell Type, choices: [LSTM, GRU, RNN]")
parser.add_argument("-nl", "--num_layers", type=int, default=2, help="Number of Layers, choices: [1, 2, 3]")
parser.add_argument("-d", "--dropout", type=float, default=0.1, help="Dropout, good_choices: [0, 0.1, 0.2]")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.005, help="Learning Rate, good_choices: [0.0005, 0.001, 0.005]")
parser.add_argument("-o", "--optimizer", type=str, default="SGD", help="Optimizer, choices: [SGD, ADAM]")
parser.add_argument("-l", "--language", type=str, default="tam", help="Language")
args = parser.parse_args()

params["embed_size"] = args.embed_size
params["hidden_size"] = args.hidden_size
params["cell_type"] = args.cell_type
params["num_layers"] = args.num_layers
params["dropout"] = args.dropout
params["learning_rate"] = args.learning_rate
params["optimizer"] = args.optimizer
language = args.language

model = Translator(language, params, device)

print("Training Model")
print("Language: {}".format(language))
print("Embedding Size: {}".format(params["embed_size"]))
print("Hidden Size: {}".format(params["hidden_size"]))
print("Cell Type: {}".format(params["cell_type"]))
print("Number of Layers: {}".format(params["num_layers"]))
print("Dropout: {}".format(params["dropout"]))
print("Learning Rate: {}".format(params["learning_rate"]))
print("Optimizer: {}".format(params["optimizer"]))
print("Teacher Forcing Ratio: {}".format(params["teacher_forcing_ratio"]))
print("Max Length: {}\n".format(params["max_length"]))

epochs = 10
old_validation_accuracy = 0

for epoch in range(epochs):
    print("Epoch: {}".format(epoch + 1))
    plot_losses = model.train()

    # take average of plot losses as training loss
    training_loss = sum(plot_losses) / len(plot_losses)

    print("Training Loss: {:.4f}".format(training_loss))
    
    training_accuracy = model.test_validate('train')
    print("Training Accuracy: {:.4f}".format(training_accuracy))

    validation_accuracy = model.test_validate('valid')
    print("Validation Accuracy: {:.4f}".format(validation_accuracy))

    if epoch > 0:
        if validation_accuracy < 0.0001:
            print("Validation Accuracy is too low. Stopping training.")
            break

        if validation_accuracy < 0.95 * old_validation_accuracy:
            print("Validation Accuracy is decreasing. Stopping training.")
            break

    old_validation_accuracy = validation_accuracy
print("Training Complete")

print("Testing Model")
test_accuracy = model.test_validate('test')
print("Test Accuracy: {:.4f}".format(test_accuracy))
print("Testing Complete")