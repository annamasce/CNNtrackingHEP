import torch
from torch.utils import data
from classes import DatasetTracks, NeuralNetModel
import matplotlib.pyplot as plt

# Parameters for DataLoader
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
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
model = model.float()
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

use_cuda = True

# Transfer model to GPU
if use_cuda and torch.cuda.is_available():
    model.cuda()

train_losses = []
val_losses = []
max_epochs = 10
print('Starting the training...')
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    for i, data in enumerate(training_generator, 0):
        # get the inputs; data is a list of [datapoints, targets]
        # print(i, type(data))
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
        train_losses.append(loss.item())

        # Print statistics
        running_loss = running_loss + loss.item()
        if i % 4 == 0:  # print every 4 mini-batches
            print('[%d, %5d] loss: %.9f' %
                  (epoch, i, running_loss / 4))
            # ax.plot(epoch*epoch_iters + i, running_loss/4, 'b-')
            running_loss = 0.0



# Save the trained model
PATH = '../CNN.pth'
torch.save(model.state_dict(), PATH)

#load the CNN previously trained
#model = NeuralNetModel()
#model.load_state_dict(torch.load(PATH))

val_running_loss = 0.0
with torch.no_grad():
    for j, val_data in enumerate(validation_generator, 0):
        val_local_datapoint, val_local_target = val_data
        if use_cuda and torch.cuda.is_available():
            val_local_datapoint = val_local_datapoint.cuda()
            val_local_target = val_local_target.cuda()
        model.eval()

        val_prediction = model(val_local_datapoint.float())
        val_loss = loss_fn(val_prediction.float(), val_local_target.float())
        val_losses.append(val_loss.item())
        # print(val_loss.item())

        # epoch_iters = len(validation_set) / params['batch_size']
        # val_running_loss = val_running_loss + val_loss.item()
        # if j % 4 == 0:  # plot every 4 mini-batches
        #     # print('[%d, %5d] val loss: %.9f' %
        #     #       (epoch, i, running_loss / 4))
        #     ax.plot(epoch * epoch_iters + j, val_running_loss / 4, 'r-')
        #     val_running_loss = 0.0


# Plot loss function for training and validation
train_epoch_iters = len(training_set) / params['batch_size']
if train_epoch_iters > int(train_epoch_iters): train_epoch_iters = int(train_epoch_iters) + 1
val_epoch_iters = len(validation_set) / params['batch_size']
if val_epoch_iters > int(val_epoch_iters): val_epoch_iters = int(val_epoch_iters) + 1
plt.figure()
plt.subplot(211)
plt.plot(list(range(train_epoch_iters*max_epochs)), train_losses, 'b-')
plt.xlabel('train iteration')
plt.ylabel('running loss')
plt.subplot(212)
plt.plot(list(range(val_epoch_iters)), val_losses, 'r-')
plt.xlabel('val iteration')
plt.ylabel('val loss')

plt.show()
