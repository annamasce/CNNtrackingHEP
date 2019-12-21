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


def encoding_position(type_xy, grid_size):
    '''
    This function returns a matrix (dim = grid_size x grid_size) corresponding to the normalized encoding position for one layer type, x or y
    :param string type_xy: layer type: {'x', 'y'}
    :return: numpy ndarray (shape = [grid_size, grid_size])
    '''
    total_grid = 512
    gap_die = 0.220
    gap_SiPM = 0.380
    ch_width = 0.250
    positions = np.zeros(grid_size)

    first_channel = int((total_grid - grid_size) / 2)
    last_channel = int((total_grid + grid_size) / 2)
    for channel in range(first_channel, last_channel):
        # half_layer = 512. / 2 * ch_width + 2 * gap_die + 1.5 * gap_SiPM
        mult = int((channel) / 64)
        shift = int(mult / 2) * gap_die + int(mult / 2) * gap_SiPM + (mult % 2) * gap_die
        pos_ch = channel * ch_width + shift  # - half_layer
        if channel == first_channel:
            pos_firstch = pos_ch
        positions[channel - first_channel] = pos_ch - pos_firstch
    pos_matrix = np.tile(positions, (1, grid_size)).reshape(grid_size, grid_size)
    norm = pos_matrix[0, grid_size - 1] - pos_matrix[0, 0]
    if (type_xy == 'y'):
        pos_matrix = pos_matrix.T
        norm = pos_matrix[grid_size - 1, 0] - pos_matrix[0, 0]
    return pos_matrix / norm - 0.5


def transf_TOT(TOT_ndarray):
    mu = 1.064
    eps = 0.01
    trTOT = np.where(TOT_ndarray, np.log(TOT_ndarray + eps) - mu, 0.)
    return trTOT


class DatasetCreator(data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_dir, length, grid_size):
        # Initialization
        self.dir = data_dir
        self.length = length
        # self.all = []
        self.offset = 0
        self.grid_size = grid_size

    def __len__(self):
        # Denotes the total number of samples
        return self.length

    def read_file(self, index):
        grid_size = self.grid_size
        # Generate one sample of data
        # Select sample
        file_input = '{}/{}_input.pkl'.format(self.dir, index)
        # file_input = self.dir + '0_input.pkl'
        # print('file opened')
        # Create the corresponding data frame
        in_df = pd.read_pickle(file_input)
        # print('dataframe imported from pickle')

        # Create datapoint tensors and target
        n_detectors = 6
        # First feature = TOT (amplitude of the signal)
        data_TOT = np.zeros((2, n_detectors, grid_size, grid_size))
        # Second feature = binary information 1/0 for signal/no-signal
        data_bin = np.zeros((2, n_detectors, grid_size, grid_size))
        # Third feature = encoding position
        data_pos = np.zeros((2, grid_size, grid_size))
        data_pos[0, :, :] = encoding_position('y', grid_size)
        data_pos[1, :, :] = encoding_position('x', grid_size)
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
            # Fill data_TOT for selected detector (layer) with amplitude data
            low_limit = int((512 - grid_size) / 2)
            up_limit = int((512 + grid_size) / 2)
            # Amplitude data transformed to obtain a quasi-gaussian distribution
            data_TOT[0, layer, :, :] = transf_TOT(mTOTy[low_limit:up_limit, low_limit:up_limit])
            data_TOT[1, layer, :, :] = transf_TOT(mTOTx[low_limit:up_limit, low_limit:up_limit])
            data_bin[0, layer, :, :] = np.where(mTOTy[low_limit:up_limit, low_limit:up_limit], 1., 0.)
            data_bin[1, layer, :, :] = np.where(mTOTx[low_limit:up_limit, low_limit:up_limit], 1., 0.)
            # Construct the corresponding target from trackIDs in x and y
            mTracky = create_matrix_1type(ch_numbersy, tracky, 'y')
            mTrackx = create_matrix_1type(ch_numbersx, trackx, 'x')
            targ_layer = create_matchmatrix(mTrackx, mTracky)
            target[layer, :, :] = targ_layer[low_limit:up_limit, low_limit:up_limit]
        data_TOT = torch.from_numpy(data_TOT)
        data_bin = torch.from_numpy(data_bin)
        data_TOT = data_TOT.view([2 * n_detectors, grid_size, grid_size])
        data_bin = data_bin.view([2 * n_detectors, grid_size, grid_size])
        data_pos = torch.from_numpy(data_pos)
        datapoint = torch.cat((data_TOT, data_bin, data_pos))
        target = torch.from_numpy(target)
        return datapoint, target

    def __getitem__(self, index):
        # print(index)
        # return self.all[index]
        item = self.read_file(index + self.offset)
        return item
