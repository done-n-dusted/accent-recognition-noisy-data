'''
    CSE583 PRML Term Project
    Code to train the model
    
    Author: Anurag Pendyala
    PSU Email: app5997@psu.edu
    Description: File containing code to train a particular model.
    Run the programming in the following way:
    python train.py --train path/to/train1.npy path/to/train2.npy --test path/to/test1.npy path/to/test2.npy ... --num_epochs 10 --model model_name
    
    the model is saved in models/<model_name>.pth and history is saved in history/<model_name>.npy
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

parser.add_argument('--train', nargs='+', type=str, required=True, help='List of file paths to train .npy files')
parser.add_argument('--test', nargs='+', type=str, required=True, help='File path for test .npy file')
parser.add_argument('--num_epochs', type=int, required=False, default=10, help='Number of epochs')
parser.add_argument('--model', type=str, required=True, help='Model Name')

args = parser.parse_args()

# Loading the train and test data.
train_npy_locations = args.train
test_npy_locations = args.test
num_epochs = args.num_epochs

train_dict = combine_npy_dicts(train_npy_locations)
test_dict = combine_npy_dicts(test_npy_locations)

# Choosing appropriate device to train
print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Formatting the data for the model
train_X, test_X = train_dict['features'], test_dict['features']
train_y, test_y = train_dict['labels'], test_dict['labels']

train_y, test_y = text_to_class(train_y), text_to_class(test_y)
num_classes = max(train_y) + 1
print(len(train_X), train_X[0].shape)
input_shape = train_X[0].shape

batch_size=8

# Declaring data loaders for train and test set
train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_y).long())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_y).long())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Creating the model
model = ConvNet(input_shape, num_classes)

# Defining loss function 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Saving history
history = {
    'loss' : [],
    'train_acc' : [],
    'test_acc' : []
}

# Begin training and testing the model after each epoch
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}: ')
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == targets).sum().item()

    # Calculate training loss and accuracy
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    history['loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    # Evaluate
    model.eval()
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == targets).sum().item()

    # Calculate validation loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    history['test_acc'].append(test_acc)

    # Print epoch statistics
    
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}, '
          f'Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_acc:.2f}')


np.save('history/' + args.model, history, allow_pickle=True)
torch.save(model.state_dict(), 'models/' + args.model + '.pth')

# print(history)