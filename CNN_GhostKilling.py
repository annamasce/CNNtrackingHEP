import torch
from torch.utils import data
from classes import DatasetTracks, NeuralNetModel
import matplotlib.pyplot as plt

model = NeuralNetModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# Parameters
params = {'batch_size': 4,  # from 8 to 64
          'shuffle': False,
          'num_workers': 6}

max_epochs = 2

training_set = DatasetTracks('./DataFrames/', 200)
training_generator = data.DataLoader(training_set, **params)

print('Training generator is ready')

validation_set = DatasetTracks('./DataFrames/', 200)
validation_generator = data.DataLoader(validation_set, **params)

print('Test generator is ready')

model = NeuralNetModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
model = model.float()

plt.figure()
plt.title('Loss function')
ax = plt.subplot()
ax.plot([], [])

print('Starting the training...')

for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    # for i in range(len(training_set)):
    #     data = training_set[i]
    #     local_datapoint, local_target = data
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [datapoints, targets]
        # print(i, type(data))
        local_datapoint, local_target = data

        # for local_datapoint, local_target in training_generator:
        ##
        ## Transfer to GPU
        # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        ##

        # Model computations
        # print('Starting the training...')
        # print(local_datapoint.float())
        prediction = model(local_datapoint.float())
        # print('Prediction done')
        # loss = torch.nn.CrossEntropyLoss(prediction, local_target)
        loss = torch.nn.functional.mse_loss(prediction.float(), local_target.float())
        optimizer.zero_grad()
        loss.backward()
        # print('Gradients computed')
        optimizer.step()
        # print('Model updated')

        # print statistics and plot point
        epoch_iters = len(training_set) / params['batch_size']
        running_loss = running_loss + loss.item()
        if i % 2 == 0:  # print every 2 mini-batches
            print('[%d, %5d] loss: %.9f' %
                  (epoch, i, running_loss / 2))

            if epoch_iters > int(epoch_iters):
                iter_number = epoch * (int(epoch_iters) + 1) + i
            else:
                iter_number = epoch * int(epoch_iters) + i

            ax.plot(iter_number, running_loss, 'ob')
            running_loss = 0.0

# Save the trained model
PATH = './CNN.pth'
torch.save(model.state_dict(), PATH)

plt.show()
