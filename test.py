'''
    CSE583 PRML Term Project
    Code to test the model
    
    Author: Anurag Pendyala
    PSU Email: app5997@psu.edu
    Description: File containing code to test a particular model.
    Run the programming in the following way:
    python test.py --model path/to/model --test path/to/test1.npy path/to/test2.npy ...
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import argparse
from utils import *
from models import ConvNet

# Parsing the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help="Model Name")
parser.add_argument('--test', nargs='+', type=str, required=True, help="File path for test .npy file")

args = parser.parse_args()

# Loading the test data and model
test_npy_locations = args.test
model_path = args.model
test_dict = combine_npy_dicts(test_npy_locations)

# Formating test data for the model to handle
test_X, test_y = test_dict['features'], text_to_class(test_dict['labels'])
num_classes = max(test_y) + 1
input_shape = test_X[0].shape

batch_size = 8

# Creating model and loading it
model = ConvNet(input_shape, num_classes)
model.load_state_dict(torch.load('models/' + model_path))

# Creating data loader
test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_y).long())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Defining loss function
criterion = nn.CrossEntropyLoss()
model.eval()

# initializing losses and variables for accuracy
test_loss = 0.0
test_correct = 0
cwise_correct = {}
cwise_total = {}
for i in range(num_classes):
    cwise_correct[i] = 0
    cwise_total[i] = 0

# testing the loaded model
with torch.no_grad():

    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Calculate loss and accuracy
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == targets).sum().item()

        for i in range(num_classes):
            cwise_correct[i] += (predicted == targets)[targets == i].sum().item()
            cwise_total[i] += (targets == i).sum().item()

    # Calculate validation loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)


# Print epoch statistics
print(f'Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.6f}')
print(f'Class-Wise Accuracies:')
classes = class_to_text(range(num_classes))
for i in range(num_classes):
    print(classes[i], ':', cwise_correct[i]/cwise_total[i])

