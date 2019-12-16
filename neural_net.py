import torch

class NeuralNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_extracted_features = 3
        # This creates all the parameters that are optimized to fit the model
        self.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=num_extracted_features, kernel_size=(3, 3),
                                     padding=(1, 1))
        self.pointwise_nonlinearity = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=num_extracted_features, out_channels=6, kernel_size=(3, 3),
                                     padding=(1, 1))

    def get_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    def forward(self, inpt):
        # print("Paramters", self.get_num_params())
        # print('Forward function')
        # This declares how the model is run on an input
        x = self.conv1(inpt)
        # print(x.size())
        x = self.pointwise_nonlinearity(x)
        x = self.conv2(x)
        # print(x)
        # print(x.size())
        return x