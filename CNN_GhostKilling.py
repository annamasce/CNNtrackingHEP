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

max_epochs = 2

training_set = Dataset('../../DataFrames/', 100)
training_generator = data.DataLoader(training_set, **params)

print('Training generator is ready')

validation_set = Dataset('../../DataFrames/', 100)
validation_generator = data.DataLoader(validation_set, **params)

print('Test generator is ready')

model = NeuralNetModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
model = model.float()

for epoch in range(max_epochs):
    # Training
    # running_loss = 0.0
    # for i, data in enumerate(training_generator, 0):
    #     # get the inputs; data is a list of [datapoints, targets]
    #     local_datapoint, local_target = data
    #     detector = 2
    #     plt.imshow(torch.sum(local_datapoint, axis=0)[230:270, 230:270, detector].numpy());
    for local_datapoint, local_target in training_generator:
        ##
        ## Transfer to GPU
        # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        ##

        # Model computations
        print('Starting the training...')
        prediction = model(local_datapoint.float())
        print('Prediction done')
        loss = torch.nn.CrossEntropyLoss(prediction, local_target)
        # loss = torch.nn.functional.mse_loss(prediction, local_target)
        optimizer.zero_grad()
        loss.backward()
        print('Gradients computed')
        optimizer.step()
        print('Model updated')

        # # print statistics
        # running_loss += loss.item()
        # if i % 2 == 0:  # print every 2 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch, i, running_loss / 2))
        #     running_loss = 0.0

# Save the trained model
PATH = './CNN.pth'
torch.save(model.state_dict(), PATH)
