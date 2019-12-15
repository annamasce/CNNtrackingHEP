import torch
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from torch.utils import data


def create_matrix_1type(index, amplitude, type_xy):
    """
    :param (list of int) index: list of indices (channels where the amplitude is different from zero)
    :param (list of int) amplitude: list of amplitudes corresponding to the indices
    :param (string) type_xy: 'x' or 'y', specifies the considered dimension (for y, transposition is required)
    :return: (numpy.ndarray) m: 512x512 matrix corresponding to the measurement in one coordinate
    """
    n = 512
    m = lil_matrix((1, n))
    m[0, index] = amplitude
    m = np.asarray(m.todense())
    m = np.tile(m[0], (1, n)).reshape(n, n)
    if (type_xy == 'y'): m = m.T
    return m


def create_matchmatrix(mx, my):
    '''

    :param (numpy.ndarray) mx: 512x512 matrix corresponding to the x measurement
    :param (numpy.ndarray) my: 512x512 matrix corresponding to the y measureent
    :return: (numpy.array) target: 512x512 binary matrix: entry=1 only when mx=my
    '''
    target = np.where(np.logical_and(mx == my, mx, my), 1, 0)
    return target


class DatasetTracks(data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_dir, length, train=True):
        # Initialization
        self.dir = data_dir
        self.length = length
        # self.all = []
        self.offset = 0

        # Create 2 different datasets for training and test
        if train:
            self.offset = 0
        else:
            self.offset = length

        # for i in range(start_range, start_range + length):

    # self.all.append(self.read_file(i))
    # print(self.all[0][0].size())

    def __len__(self):
        # Denotes the total number of samples
        return self.length

    def read_file(self, index):
        # Generate one sample of data
        # Select sample
        file_input = '{}/{}_input.pkl'.format(self.dir, index)
        # file_input = self.dir + '0_input.pkl'
        # print('file opened')
        # Create the corresponding data frame
        in_df = pd.read_pickle(file_input)
        # print('dataframe imported from pickle')

        # Create datapoint and target
        grid_size = 512
        n_detectors = 6
        # datapoint = np.zeros((2, grid_size, grid_size, n_detectors))
        datapoint = np.zeros((2, n_detectors, grid_size, grid_size))
        # target = np.zeros((1, grid_size, grid_size, n_detectors))
        target = np.zeros((n_detectors, grid_size, grid_size))

        for layer in range(6):
            boardy = layer * 2
            boardx = layer * 2 + 1
            ch_numbersy = in_df.ChPosition[boardy]
            ch_numbersx = in_df.ChPosition[boardx]
            TOTy = in_df.TOT[boardy]
            TOTx = in_df.TOT[boardx]
            tracky = in_df.TrackID[boardy]
            trackx = in_df.TrackID[boardx]
            mTOTy = create_matrix_1type(ch_numbersy, TOTy, 'y')
            mTOTx = create_matrix_1type(ch_numbersx, TOTx, 'x')
            # Fill datapoint for selected detector (layer) with amplitude data
            datapoint[0, layer, :, :] = mTOTy
            datapoint[1, layer, :, :] = mTOTx
            # Construct the corresponding target from trackIDs in x and y
            mTracky = create_matrix_1type(ch_numbersy, tracky, 'y')
            mTrackx = create_matrix_1type(ch_numbersx, trackx, 'x')
            target[layer, :, :] = create_matchmatrix(mTrackx, mTracky)
        datapoint = torch.from_numpy(datapoint)
        datapoint = datapoint.view([12, 512, 512])
        target = torch.from_numpy(target)
        return datapoint, target

    def __getitem__(self, index):
        # print(index)
        # return self.all[index]
        item = self.read_file(index + self.offset)
        return item


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

def signal_entries(input, output):
    '''
    This function takes 2 input matrices of the same size (x and y layer for a certain detector) and returns the subset of the output matrix corresponding to the possible signal points (match between x and y layers)
    :param tuple input: (tensor ymatrix, tensor xmatrix)
    :param torch tensor output: layer matrix
    :return: (torch tensor) subset od output corresponding to possible signal points
    '''
    ymatrix = input[0]
    xmatrix = input[1]
    ymatrix = ymatrix[:, 0] # take only one column
    xmatrix = xmatrix[0, :] # take only one row
    y_index = ymatrix.nonzero(as_tuple=True)[0]
    x_index = xmatrix.nonzero(as_tuple=True)[0]
    target = torch.index_select(output, 0, y_index)
    target = torch.index_select(target, 1, x_index)
    return target

def transf_prediction(prediction, thr):
    true_tensor = torch.ones(tuple(prediction.size()))
    false_tensor = torch.zeros(tuple(prediction.size()))
    tr_pred = torch.where(prediction>=thr, true_tensor, false_tensor)
    return tr_pred

def accuracy_step(prediction, target):
    corr_tensor = torch.eq(prediction, target)
    print(corr_tensor)
    correct = float(corr_tensor.sum())
    total = corr_tensor.nelement()
    print(total)
    return correct, total

def f1_step(prediction, target):
    true_tensor = torch.ones(tuple(prediction.size()))
    false_tensor = torch.zeros(tuple(prediction.size()))
    true_positives = float((prediction*target).sum())
    #print(true_positives)
    reversed_pred = torch.where(prediction==0, true_tensor, false_tensor)
    reversed_targ = torch.where(target==0, true_tensor, false_tensor)
    false_negatives = float((reversed_pred*target).sum())
    false_positives = float((prediction*reversed_targ).sum())
    actual_positives = true_positives + false_negatives
    pred_positives = true_positives + false_positives
    return true_positives, actual_positives, pred_positives