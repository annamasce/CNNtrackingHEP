import torch
from torch.utils import data
from classes import Dataset, NeuralNetModel
import matplotlib.pyplot as plt

model = NeuralNetModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# Parameters
params = {'batch_size': 64, #from 8 to 64
          'shuffle': True,
          'num_workers': 6}

max_epochs = 100

training_set = Dataset('../../DataFrames/', 1000)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset('../../DataFrames/', 1000)
validation_generator = data.DataLoader(validation_set, **params)

for epoch in range(max_epochs):
    # Training
    for local_datapoint, local_target in training_generator:
        ##
        ## Transfer to GPU
        # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        ##

        # Model computations
        prediction = model(local_datapoint)
        loss = torch.nn.functional.mse_loss(prediction, local_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


