from DfClasses import event
import ROOT
import root_pandas
import pandas as pd
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
import argparse
import sys
import timeit

# code_to_time = """
parser = argparse.ArgumentParser()
parser.add_argument('rootfile_input', help='Name of the ROOT file containing the tree')
# parser.add_argument('pklfile_output', help='Path to the dataframe directory')
arguments = parser.parse_args()

input_filename = '/home/anna/Documents/Tracking_ml/RootFiles/' + arguments.rootfile_input
rootfile = ROOT.TFile.Open(input_filename)
rootfile.ls()
data_tree = rootfile.Get('tree_clustering')
df = root_pandas.read_root(input_filename, key='tree_clustering')
print(df.ChPosition[2])


output_path = '/home/anna/Documents/Tracking_ml/DataFrames/'

for event in range(0,len(df.index)):
# for event in range(0):
    if event%100 == 0: print('Processing event {}'.format(event))
    # create a dataframe per event with columns: 'ChPosition', 'TOT', 'TrackID'
    df_event = pd.DataFrame()
    ChPosition_col = []
    TOT_col = []
    TrackID_col = []
    for board in range(0,12):
        # select the indices for the given board
        hits_board = [hit for hit in range(len(df.BoardID[event])) if df.BoardID[event][hit] == board+1]
        # print(hits_board)
        # fill the np.arrays for the given board
        ChPosition_board = np.asarray([df.ChPosition[event][hit] for hit in hits_board])
        TOT_board = np.asarray([df.TOT[event][hit] for hit in hits_board])
        TrackID_board = np.asarray([df.TrackID[event][hit] for hit in hits_board])
        # append them to the dataframe columns
        ChPosition_col.append(ChPosition_board)
        TOT_col.append(TOT_board)
        TrackID_col.append(TrackID_board)
    df_event['ChPosition'] = ChPosition_col
    df_event['TOT'] = TOT_col
    df_event['TrackID'] = TrackID_col
    filename_output = output_path + '{}_input.pkl'.format(event)
    # print(filename_output)
    df_event.to_pickle(filename_output)

    # df_read = pd.read_pickle(filename_output)
    # print(df_read)

    #print(df_event)
    #print(df['TOT'][0])




