import torch
from torch.utils import data
from classes import DatasetTracks, NeuralNetModel, accuracy_step, signal_entries
import matplotlib.pyplot as plt
import sys
import argparse
from random import seed
from random import randint
import glob
import os

import json

parser = argparse.ArgumentParser()
parser.add_argument('in_dir', help='Path to the directory containing data frames')
parser.add_argument('out_dir', help='Path to the directory in which the models are saved')
parser.add_argument('batch_size', type=int, help='batch size for training and validation datasets')
parser.add_argument('dataset_size', type=int, help='total size of the dataset, to be splitted into training and validation')
parser.add_argument('l_rate', type=float, help='learning rate')
parser.add_argument('epochs', type=int, help='number of epochs for training loop')
arguments = parser.parse_args()

# Set the run number and create the corresponding directory
seed(1)
run_nbr = randint(0, 1000)
while glob.glob('{}/RUN_{}'.format(arguments.out_dir, run_nbr)):
    run_nbr = randint(0, 1000)
path_rundir = '{}/RUN_{}'.format(arguments.out_dir, run_nbr)
os.mkdir(path_rundir)
print('Run number {}'.format(run_nbr))

# Parameters for DataLoader
params = {'batch_size': arguments.batch_size,  # from 8 to 64
          'shuffle': False,
          'num_workers': 6}

# Generate training and validation datasets
length = arguments.dataset_size
Dataset = DatasetTracks(arguments.in_dir, length)
training_set, validation_set = data.dataset.random_split(Dataset, [int(length/2), int(length/2)])
print(len(training_set))
print(len(validation_set))
training_generator = data.DataLoader(dataset=training_set, **params)
print('Training generator is ready')
validation_generator = data.DataLoader(dataset=validation_set, **params)
print('Test generator is ready')

# Define model, optimizer and loss function
model = NeuralNetModel()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=arguments.l_rate)
model = model.float()
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

# Create config file with model hyperparameters
config_filename = '{}/model_config.json'.format(path_rundir)
with open(config_filename, 'w+') as json_file:
  json.dump(params, json_file)

# sys.exit(0)

# Transfer model to GPU
use_cuda = True
if use_cuda and torch.cuda.is_available():
    model.cuda()

# Create csv file to save loss values
trloss_filename = '{}/train_losses.csv'.format(path_rundir)
f_loss = open(trloss_filename, 'w+')

max_epochs = arguments.epochs
print('Starting the training...')
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [datapoints, targets]
        local_datapoint, local_target = data

        # Transfer data to GPU
        if use_cuda and torch.cuda.is_available():
            local_datapoint = local_datapoint.cuda()
            local_target = local_target.cuda()

        # Model computations
        prediction = model(local_datapoint.float())
        loss = loss_fn(prediction.float(), local_target.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Write loss to file
        f_loss.write('{},'.format(loss.item()))

        # Print statistics
        running_loss = running_loss + loss.item()
        if i % 4 == 0:  # print every 4 mini-batches
            print('[%d, %5d] loss: %.9f' %
                  (epoch, i, running_loss / 4))
            running_loss = 0.0

    # Save the trained model at each epoch
    PATH = '{0}/CNN_epoch{1}_loss{2:.6}.pth'.format(path_rundir, epoch, loss.item())
    print(PATH)
    torch.save(model.state_dict(), PATH)
f_loss.close()

#Validation
vloss_filename = '{}/val_losses.csv'.format(path_rundir)
f_loss = open(vloss_filename, 'w+')
val_running_loss = 0.0
corr_overall = 0
tot_overall = 0
with torch.no_grad():
    for j, val_data in enumerate(validation_generator, 0):
        val_local_datapoint, val_local_target = val_data
        if use_cuda and torch.cuda.is_available():
            val_local_datapoint = val_local_datapoint.cuda()
            val_local_target = val_local_target.cuda()
        model.eval()

        val_prediction = model(val_local_datapoint.float())
        # print(val_prediction.size())
        # print(val_local_target.size())
        # Calculate the loss and print it to file
        val_loss = loss_fn(val_prediction.float(), val_local_target.float())
        f_loss.write('{},'.format(val_loss.item()))
        # print(val_loss.item())

        # Calculate accuracy of the model
        for layer in range(6):
            for sample in range(tuple(val_prediction.size())[0]):
                input = (val_local_datapoint[sample, 0, :, :, layer], val_local_datapoint[sample, 1, :, :, layer])
                pred = signal_entries(input, val_prediction[sample, 0, :, :, layer])
                targ = signal_entries(input, val_local_target[sample, 0, :, :, layer])
                if pred.nelement() > 1:
                    corr, tot = accuracy_step(pred, targ, 0.5)
                    corr_overall += corr
                    tot_overall += tot

accuracy = corr_overall/tot_overall
print('Accuracy of the model:', accuracy)

f_loss.close()


