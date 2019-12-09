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

class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_dir, length):
        # Initialization
        self.dir = data_dir
        self.length = length

    def __len__(self):
        # Denotes the total number of samples
        return self.length

    def __getitem__(self, index):
        # Generate one sample of data
        # Select sample
        file_input = self.dir + str(index) + '_input.pkl'
        # Create the corresponding data frame
        in_df = pd.read_pickle(file_input)

        # Create datapoint and target
        grid_size = 512
        n_detectors = 6
        datapoint = np.zeros((2, grid_size, grid_size, n_detectors))
        target = np.zeros((1, grid_size, grid_size, n_detectors))

        for layer in range(6):
            boardy = layer*2
            boardx = layer*2+1
            ch_numbersy = in_df.ChPosition[boardy]
            ch_numbersx = in_df.ChPosition[boardx]
            TOTy = in_df.TOT[boardy]
            TOTx = in_df.TOT[boardx]
            tracky = in_df.TrackID[boardy]
            trackx = in_df.TrackID[boardx]
            mTOTy = create_matrix_1type(ch_numbersy, TOTy, 'y')
            mTOTx = create_matrix_1type(ch_numbersx, TOTx, 'x')
            # Fill datapoint for selected detector (layer) with amplitude data
            datapoint[0, :, :, layer] = mTOTy
            datapoint[1, :, :, layer] = mTOTx
            # Construct the corresponding target from trackIDs in x and y
            mTracky = create_matrix_1type(ch_numbersy, tracky, 'y')
            mTrackx = create_matrix_1type(ch_numbersx, trackx, 'x')
            target[0, :, :, layer] = create_matchmatrix(mTrackx, mTracky)
        datapoint = torch.from_numpy(datapoint)
        target = torch.from_numpy(target)
        return datapoint, target

class NeuralNetModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    num_extracted_features = 20
    # This creates all the parameters that are optimized to fit the model
    self.conv1 = torch.nn.Conv3d(in_channels=2, out_channels=num_extracted_features, kernel_size=(3,3,3), padding=(1,1,1))
    self.pointwise_nonlinearity = torch.nn.ReLU()
    self.conv2 = torch.nn.Conv3d(in_channels=num_extracted_features, out_channels=1, kernel_size=(3,3,3), padding=(1,1,1))

  def forward(self, inpt):
    # This declares how the model is run on an input
    x = self.conv1(inpt)
    x = self.pointwise_nonlinearity(x)
    x = self.conv2(x)
    return x
