import torch
from torch.utils import data
from neural_net import NeuralNetModel
from dataset import DatasetCreator
from model_functions import Validation, mask
import matplotlib.pyplot as plt
import sys
import argparse
from random import seed
from random import randint
import glob
import os
import numpy as np

import json

parser = argparse.ArgumentParser()
parser.add_argument('device', help='Device {cpu,cuda:0}')
parser.add_argument('in_dir', help='Path to the directory containing data frames')
parser.add_argument('out_dir', help='Path to the directory in which the models are saved')
parser.add_argument('batch_size', type=int, help='batch size for training and validation datasets')
parser.add_argument('dataset_size', type=int, help='total size of the dataset, to be splitted into training and validation')
parser.add_argument('l_rate', type=float, help='learning rate')
parser.add_argument('epochs', type=int, help='number of epochs for training loop')
parser.add_argument('grid_size', type=int, help='imaged size (if grid_size=512, then images are not cropped)')
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
          'shuffle': True,
          'num_workers': 4}

# Generate training and validation datasets
torch.manual_seed(0)
np.random.seed(0)
length = arguments.dataset_size
Dataset = DatasetCreator(arguments.in_dir, length, arguments.grid_size)
training_set, validation_set = data.dataset.random_split(Dataset, [int(length/2), int(length/2)])
print(len(training_set))
print(len(validation_set))
training_generator = data.DataLoader(dataset=training_set, **params)
print('Training generator is ready')
validation_generator = data.DataLoader(dataset=validation_set, **params)
print('Test generator is ready')

# U-net model
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=26, out_channels=6, init_features=32, pretrained=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=arguments.l_rate)
model = model.float()


# Create config file with model hyperparameters
params.update({'dataset_size': arguments.dataset_size, 'learning_rate': arguments.l_rate,
               'epochs_number': arguments.epochs, 'grid_size': arguments.grid_size})
config_filename = '{}/model_config.json'.format(path_rundir)
with open(config_filename, 'w+') as json_file:
  json.dump(params, json_file)

# Transfer model to GPU
model.to(arguments.device)

# Validation before training
print('Results before training:' )
val_object = Validation(device=arguments.device)
val_object.val_loop(model, validation_generator)
acc = val_object.get_accuracy()
rec = val_object.get_recall()
prec = val_object.get_precision()
f1 = val_object.get_f1()

# Write results to file
results = {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1}
results_filename = '{}/epoch{}_results.json'.format(path_rundir, -1)
with open(results_filename, 'w+') as json_file:
    json.dump(results, json_file)

# Create csv file to save loss values
trloss_filename = '{}/train_losses.csv'.format(path_rundir)
f_loss = open(trloss_filename, 'w+')

val_object = Validation(device=arguments.device)

max_epochs = arguments.epochs
print('Starting the training...')
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    for i, sample_data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [datapoints, targets]
        local_datapoint, local_target = sample_data

        # Transfer data to device
        local_datapoint = local_datapoint.to(arguments.device)
        local_target = local_target.to(arguments.device)

        # Model computations
        prediction = model(local_datapoint.float())
        mask_tensor = mask(local_datapoint.float(), arguments.grid_size, device=arguments.device)
        loss_fn = torch.nn.BCELoss(reduction='mean', weight=mask_tensor)
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
    if epoch % 1 == 0:
        PATH = '{0}/CNN_epoch{1}_loss{2:.6}.pth'.format(path_rundir, epoch, loss.item())
        print(PATH)
        torch.save(model.state_dict(), PATH)

    # Validation
    val_object.val_loop(model, validation_generator)
    acc = val_object.get_accuracy()
    rec = val_object.get_recall()
    prec = val_object.get_precision()
    f1 = val_object.get_f1()

    # Write results to file
    if epoch % 1 == 0:
        results = {'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1}
        results_filename = '{}/epoch{}_results.json'.format(path_rundir, epoch)
        with open(results_filename, 'w+') as json_file:
            json.dump(results, json_file)

f_loss.close()

