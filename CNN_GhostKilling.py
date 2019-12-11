import torch
from torch.utils import data
from classes import DatasetTracks, NeuralNetModel
import matplotlib.pyplot as plt

model = NeuralNetModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# Parameters
params = {'batch_size': 8,  # from 8 to 64
          'shuffle': False,
          'num_workers': 6}

length = 1000
Dataset = DatasetTracks('../../DataFrames/', length)
training_set, validation_set = data.dataset.random_split(Dataset, [int(length/2), int(length/2)])
print(len(training_set))
print(len(validation_set))
training_generator = data.DataLoader(dataset=training_set, **params)
print('Training generator is ready')
validation_generator = data.DataLoader(dataset=validation_set, **params)
print('Test generator is ready')

model = NeuralNetModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
model = model.float()
loss_fn = torch.nn.BCEWithLogitsLoss()

# plt.figure()
# plt.title('Loss function')
# ax = plt.subplot()
# ax.plot([], [])
train_losses = []
val_losses = []
max_epochs = 2
print('Starting the training...')
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [datapoints, targets]
        # print(i, type(data))
        local_datapoint, local_target = data

        ##
        ## Transfer to GPU
        # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        ##

        # Model computations
        prediction = model(local_datapoint.float())
        # print('Prediction done')
        loss = loss_fn(prediction.float(), local_target.float())
        optimizer.zero_grad()
        loss.backward()
        # print('Gradients computed')
        optimizer.step()
        # print('Model updated')
        train_losses.append(loss.item())
        # print(loss.item())
        # print statistics and plot point
        epoch_iters = len(training_set) / params['batch_size']
        running_loss = running_loss + loss.item()
        if i % 4 == 0:  # print every 4 mini-batches
            print('[%d, %5d] loss: %.9f' %
                  (epoch, i, running_loss / 4))
            running_loss = 0.0

    # Validation
    with torch.no_grad():
        for j, val_data in enumerate(validation_generator, 0):
            val_local_datapoint, val_local_target = val_data

            model.eval()

            val_prediction = model(val_local_datapoint.float())
            val_loss = loss_fn(val_prediction.float(), val_local_target.float())
            val_losses.append(val_loss.item())

# Save the trained model
PATH = '../CNN.pth'
torch.save(model.state_dict(), PATH)

# plt.show()
