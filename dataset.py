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


class DatasetCreator(data.Dataset):
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