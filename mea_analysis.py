# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:05:29 2023

@author: kh19883
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
from collections import Counter


import statsmodels.api as sm
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy import signal
import scipy
from scipy.io import loadmat
from scipy.stats import gaussian_kde
import scipy.stats as stats

import os, sys
sys.path.append('C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics')
from myModules import rhythm as rh
from myModules import raincloud_module as rm

import re
import mat73

sys.path.append('C:\\ProgramData\\Nex Technologies\\NeuroExplorer 5 x64')
import nex
from NexFileData import *
import NexFileWriters




# %% PATH DEFINITIONS AND LOAD FUNCTIONS
"""
Note that paths in this section all end in "...\\" so that only the name of the
file needs be appended to the path in order to define a file location

"""



def get_recording_csv_path():
    return "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted CSV Files\\"

def get_recording_nex_path():
    return "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted NEX Files\\"

def load_summary_table(drop_trh=True):
    if drop_trh==True:
        path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents\PhD Neural Dynamics\ARC-Bursting\MEA Data\arc_experiments_summary.xlsx"
    else:
        path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents\PhD Neural Dynamics\ARC-Bursting\MEA Data\arc_experiments_summary_2.xlsx"
    
    sum_table = pd.read_excel(path, index_col=0)
    
    if drop_trh==True:
        sum_table = sum_table[~sum_table['Experiment'].isin(['TRH 1-2uM'])]
    
    return sum_table

def save_summary_table(sum_table):
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents\PhD Neural Dynamics\ARC-Bursting\MEA Data\arc_experiments_summary.xlsx"
    sum_table.to_excel(path)
    
    
def print_recordings_and_animals(rec_list):
    """
    Given a list or recordings (in the form 'ddmmyy_sX') this function
    calculates the number of recordings and the number of animals. 

    Parameters
    ----------
    rec_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sum_table = load_summary_table(drop_trh=True)
    
    n_cells = sum_table.loc[rec_list, 'N_tidas'].sum()
    n_rec = len(rec_list)
    n_rat = len(pd.Series([rec_list[i][0:6] for i in range(len(rec_list))]).unique())

    print('=============================')
    print('Number of cells: ', n_cells)
    print('Number of recordings: ', n_rec)
    print('Number of animals: ', n_rat)
    print('=============================')
    

# %% MAT --> CSV CONVERSION 
"""
Some notes about the MAT file, m:
- It is a dict of dict (nested dicts termed unit_dict's). Each constituent
                        dict describes the properties of a single sorted unit.
- unit_dict['values'] is a colum matrix of waveforms for evry spike
- unit_dict['times'] is an array of timestamps for the single unit
- unit_dict['title'] is the important one for describing the channel index.

"""


def get_ch_number(s):
    """
    Function to get the channel number from strings of the form 'Ch 60 unit 24'

    Parameters
    ----------
    s : str
        unit_title, e.g. 'Ch 60 unit 24'.

    Returns
    -------
    ch_number : int
        The channel number, e.g. 60.

    """
    # match the pattern 'Ch <number>'
    match = re.search(r'Ch (\d+)', s)
    if match:
        # extract the number and convert it to an integer
        ch_number = int(match.group(1))
        return ch_number
    else:
        # pattern not found
        return None

    

def get_unit_number(s):
    """
    Function to get the unit number from strings of the form 'Ch 60 unit 24'

    Parameters
    ----------
    s : str
        unit_title, e.g. 'Ch 60 unit 24'.

    Returns
    -------
    unit_number : int
        The unit number, e.g. 24.

    """
    # match the pattern 'Ch <number>'
    match = re.search(r'unit (\d+)', s)
    if match:
        # extract the number and convert it to an integer
        unit_number = int(match.group(1))
        return unit_number
    else:
        # pattern not found
        return None

    

def channel_to_electrode(n):
    """
    Function to convert the channel number from Spike2 --> electrode number/position
    The rules of this conversion map are:
        (A) If ChNum is of n*10, then it is EleNum 10n
        (B) If ChNum is anything else, add 10 and swap the digits

    Parameters
    ----------
    n : int
        Channel number extracted from MAT file.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    e : int for float
        The electrode number that corresponds to the MEA grid position.

    """
    # if the channel number is a multiple of 10
    if n % 10 == 0:
        e = int(100 + n/10)
        
    # if the channel number is not a multiple of 10
    else:
            # check that n is a integer less than 100
        if n > 99:
            raise ValueError("Input must be a two-digit integer")

        # add 10 to n and extract its tens and ones digits
        new_n = n + 10
        tens_digit = new_n // 10
        ones_digit = new_n % 10

        # swap the digits and return the result
        e = 10 * ones_digit + tens_digit
        
    return e



def unit_column_name(unit_dict):
    """
    Function to make a string column name from the ElNum and unitNum for a single 
    MAT file unit_dict. This replaces the MAT file unit_title.

    Parameters
    ----------
    unit_dict : dict
        Single unit dictionary from the MAT file.

    Returns
    -------
    str
        String sequence of the new title for a single unit column.
    """
    
    # extract the channel and unit num
    c = get_ch_number(unit_dict['title'])
    u = get_unit_number(unit_dict['title'])
    # convert the channel num to an electrode num
    e = channel_to_electrode(c)
    
    # compile into a string for the cloumn name
    return 'e'+str(e)+' u'+str(u)
 
    

def stimes_from_MAT_2_DF(m):
    """
    Function that takes the spike times from the spike sorted .MAT file and 
    stores them in a pandas DataFrame.


    Parameters
    ----------
    m : MAT file
        MAT file from Spike2 for a whole recording.

    Returns
    -------
    spike_times : DataFrame
        DataFrame of spike times.

    """
    
    # remove the FILE dict from m. This leaves only data dict's within m
    m.pop('file')
    
    
    # loop through all units of the file and add the spike times to a data frame
    spike_time_dict={}
    channel_units  ={}
    column_names   =[]
    for mkey in m:
        
        # the dictionary containing all the data for a single unit
        unit_dict = m[mkey]
        
        # get the electrode number of the unit 
        c_num = get_ch_number( unit_dict['title'] )
        e_num = channel_to_electrode(c_num)
        
        # check if the electode is in an auxilary dictionary
        if e_num in channel_units:
            # then increment the new unit number by 1 (and store current number in d)
            channel_units[e_num] += 1
            new_unit_number = channel_units[e_num]
        else:
            # then just set the new unit number as 1
            channel_units[e_num] = 1
            new_unit_number = 1
            
        # the new column title 
        column_name = f'e{e_num}u{new_unit_number}'
        column_names.append(column_name)
        
        # store the spike times in a dictionary keyed by cXuY
        spike_time_dict[column_name] = pd.Series(unit_dict['times'])   # think I should add a data type here 
        
    # create a data frame for the whole slice
    spike_times = pd.DataFrame(spike_time_dict)
    
    # reorder the columns to be in order
    sorted_columns = sorted(spike_times.columns)
    spike_times = spike_times.reindex(columns=sorted_columns)
    
    return spike_times



def stimes_from_DF_2_CSV(df, exp_name, overwrite=False):
    """
    Function to save the DataFrame as a CSV file

    Parameters
    ----------
    df : DataFrame
        DataFrame of spike times, organised into columns for each unit.
    exp_name : str
        The name of the experiment, e.g. "220324_s1".

    Returns
    -------
    None.

    """
    # all files can be saved in the same directory
    save_path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted CSV Files\\"
    
    # check whether the CSV file already exsists  
    if os.path.exists(save_path+exp_name+".csv"): # file already exists...
        
        # depending upon user input, either do nothing...
        if overwrite==False:
            print(exp_name+".csv already exists. File not overwritten")
        # or overwrite the file
        elif overwrite==True:
            df.to_csv(save_path+"\\"+exp_name+".csv")
            print(exp_name+".csv already exists and it has been overwritten")
    else:
        # save as a CSV file
        df.to_csv(save_path+"\\"+exp_name+".csv")
        print(exp_name+".csv has been created")
        
        
        
def stimes_from_MAT_2_CSV(filename, overwrite=False):
    """
    Converts the spike times found in the MAT file into a CSV fiel

    Parameters
    ----------
    exp_name : str
        Date-slice name of the recording.

    Returns
    -------
    None.

    """
    # extract the experiment name
    exp_name = filename[:-4]
    
    # all MAT files are saved in the same place
    path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted MAT Files\\"

    # load the MAT file
    m = mat73.loadmat(path+filename)

    # extract spike times fron MAT and save as a DataFrame
    df = stimes_from_MAT_2_DF(m)

    # save the spike times as a CSV file
    stimes_from_DF_2_CSV(df, exp_name, overwrite=overwrite)
    
    
    
    
    
def batch_stimes_2_CSV(overwrite=False):
    """
     Function that loops through a directory and generates spike time CSV files 
     from the curated MAT files. 

    Parameters
    ----------
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    # all MAT files are saved in the same directory
    path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted MAT Files\\"

    # all CSV files are stored in a separate directory 
    csv_path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted CSV Files\\"
    
    # loop through all the curated MAT files
    number_of_skipped_files=[]
    for filename in os.listdir(path):
        
        # check if the CSV file already exsists and skip the conversion if it does
        csv_filename = filename[:-4]+".csv"
        if (csv_filename in os.listdir(csv_path)) and (overwrite==False):
            print(csv_filename+" skipped because it already exists")
            number_of_skipped_files.append(csv_filename)
            continue

        else:
            # convert the spike times in the MAT file into a CSV file
            stimes_from_MAT_2_CSV(filename, overwrite=overwrite)
        
    # display how many files were skipped
    print(str(len(number_of_skipped_files))+' file conversions were skipped')
    
    
    
def load_recording_stimes(rec_name):
    
    save_path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted CSV Files\\"

    df = pd.read_csv(save_path+rec_name+".csv", index_col=0)
    
    return df



def stimes_from_DF_2_NEX(df, df_name, save_path='recording default'):
    """
    Function that converts a single CSV file of spike times into a NEX file.
    A NEX file is required for the burst analysis.The file is saved in a directory
    containing NEX files of all recordings. 

    Parameters
    ----------
    exp_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # define where the NEX file will be saved
    if save_path=='recording default':
        save_path = get_recording_nex_path()
    else:
        save_path = save_path


    # create a FileData object with 25k Hz frequency
    fd = FileData()
    fd.TimestampFrequency = 25000 

    # loop through all columns of the CSV adding each one as a 'neuron' to the FileData
    for col in df:
        
        # if the DF is hierarchical, change the column name to include rec+unit
        if isinstance(df.columns, pd.MultiIndex):
            nex_col_name = str(col[0]+"_"+col[1])
        else:
            nex_col_name = col
        
        # add the spike times to a Neuron object in the NEX file
        fd.Neurons.append(Neuron(nex_col_name, list(df[col].dropna())))    

    # save the file as a NEX file
    writer = NexFileWriters.NexFileWriter()
    writer.WriteDataToNexFile(fd, save_path+df_name+".nex")




def batch_stimes_2_NEX(overwrite=False):
    """
     Function that loops through the CSV directory and converts the files into NEX files. 

    Parameters
    ----------
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    # all CSV files are saved in the same directory
    path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted CSV Files\\"

    # all CSV files are stored in a separate directory 
    save_path = "C:\\Users\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\KiloSorted NEX Files\\"
    
    # loop through all the curated MAT files
    number_of_skipped_files=0
    for filename in os.listdir(path):
        
        # check if the NEX file already exsists and skip the conversion if it does
        nex_filename = filename[:-4]+".nex"
        if (nex_filename in os.listdir(save_path)) and (overwrite==False):
            number_of_skipped_files += 1
            continue

        else:
            # convert the spike times in the MAT file into a CSV file
            stimes_from_CSV_2_NEX(filename[:-4])
        
    # display how many files were skipped
    print(str(number_of_skipped_files)+' file conversions were skipped')



# %% hDF FILES

def load_or_create_hDF(hdf_path):
    """
    Function that returns a hDF file for storing spike time columns from multiple
    recordings. 
    
    The fucntion will check if the file given by **hdf_path** exists. If it does,
    it will load the file and if it does not, it will create the file

    Parameters
    ----------
    hdf_path : Path
        FULL path (including file and type).

    Returns
    -------
    hdf : TYPE
        DESCRIPTION.

    """
    
    # check if the hDF file exists
    if os.path.exists(hdf_path):
        # load the file if it does exist
        hdf = pd.read_csv(hdf_path, header=[0,1], index_col=0)
        print('hDF file loaded')

    else:
        # create the file if it does not exist
        columns = pd.MultiIndex.from_tuples([], names=["Recording", "Neuron"])
        hdf = pd.DataFrame(columns=columns)
        hdf.to_csv(hdf_path)
        #print('This hDF does not exist, so an empty hDF has been made for it.')
        print('hDF file created')
        
    return hdf



def update_hDF(hdf, units_to_add, recording_dir_path, hdf_path, overwrite=True):
    """
    - Function that adds columns of spike times to a hDF file (**hdf**).
    - The dict **units_to_add** details the recordings and the units within the 
        recording to add to the hDF
    - Recording files are loaded as DFs from their CSV files
    - The units are added. If the recording already exists in the hDF the adding 
        units can be skipped or all the units for that recording can be replaced
        depending upon **overwrite**
    - The hDF is saved

    Parameters
    ----------
    hdf : Data Frame
        DESCRIPTION.
    units_to_add : dict
        Dict of units to add, e.g. d={'yymmdd_sx': ['e14u1']}
    recording_dir_path : path
        directory of the recording CSV files.
    hdf_path : path
        full name of the hDF file.
    overwrite : bool, optional
        If True, the units in units_to_add will replace any units from the same 
        recording. The default is True.

    Returns
    -------
    hdf : TYPE
        DESCRIPTION.

    """
    
    # loop through the dict of neurons, load the recording and add the neuron
    for rec in units_to_add:
        # laod the recording from the CSV file
        df_ = pd.read_csv(recording_dir_path + rec + ".csv", index_col=0)

        # islolate the neurons to be added to the hDF
        if units_to_add[rec] == 'all':
            units_to_add[rec] = df_.columns.to_list()
            print('all units from '+rec+' being added')            
            
        df_add = df_[units_to_add[rec]]

        # change the columns of df_add to make it heirarchical
        cols = pd.MultiIndex.from_product([[rec], units_to_add[rec]])
        df_add.columns = cols
        df_add = df_add.reindex(sorted(df_add.columns), axis=1) # sort the columns in neruon order

        # check if neurons from this recording have been added to the hDF previously
        if rec in hdf.columns.get_level_values(0).unique():
            print('Neurons from recording '+rec+' already exist.')

            if overwrite==True:
                # replace the table for the recording with the new table
                hdf = hdf.drop(columns=rec, level=0)     # remove previous data for the recording
                hdf = pd.concat([hdf, df_add], axis=1)   # replace with updated neuron list
                print('They have been updated.')
            else:
                print('They have not been added.')
                continue

        # add the new table into the hDF        
        else:
            hdf = pd.concat([hdf, df_add], axis=1)
            print('New data added')
            
        # save the updated hDF file
        hdf.to_csv(hdf_path)
            
    return hdf 


# %% TIDA CSV FILES

def collect_tidas_from_experiments(time='all'):
    """
    Function that goes though individual experiment TIDA CSV files and lumps 
    all TIDA activity into a single data frame. 
    
    The **time** parameter (={'all', 'day', 'night'}) specifies which recording 
    condition the data is from.

    """
    
    
    
    # ===============================================================================
    def collect_night_or_day_tidas(time): 
        """
        This nested function generates a daytime or a nighttime data frame of
        spike times
        """
    
        # choose daytime, nighttime or all recordings
        if time   == 'day':
            path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\Experiment Analysis\\day-TIDAs\\"
    
        elif time == 'night':
            path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\Experiment Analysis\\night-TIDAs\\"        
    
        # loop through the TIDA experiment files and add all TIDA data to a single array
        hdf_list=[]
        for filename in os.listdir(path):
            spike_time_df = pd.read_csv(path+filename, header=[0,1], index_col=0, dtype=float)
            spike_time_df.sort_index(axis=1, inplace=True)
            hdf_list.append(spike_time_df)
        tidas = pd.concat(hdf_list, axis=1)
        tidas.reset_index(drop=True, inplace=True)
    
    
        return tidas
    # ===============================================================================


    # create either a night or day data frame
    if time != 'all':
        tidas = collect_night_or_day_tidas(time)

    # create a data frame with both night and day recordings
    elif time == 'all':
        df1 = collect_night_or_day_tidas(time='day')
        df2 = collect_night_or_day_tidas(time='night')
        tidas = pd.concat([df1, df2], axis=1)

    return tidas



def collect_bursters_from_experiments():
    """
    Function that goes though individual experiment Bursting CSV files and lumps 
    all Burst activity into a single data frame. 

    """
    
    
    
    path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\Experiment Analysis\\non-TIDA Bursters\\"        
    
    # loop through the TIDA experiment files and add all TIDA data to a single array
    hdf_list=[]
    for filename in os.listdir(path):
        spike_time_df = pd.read_csv(path+filename, header=[0,1], index_col=0, dtype=float)
        spike_time_df.sort_index(axis=1, inplace=True)
        hdf_list.append(spike_time_df)
    bursters = pd.concat(hdf_list, axis=1)
    bursters.reset_index(drop=True, inplace=True)
    
    
    return bursters


def load_tida_spike_times():
    """
    Function to load the TIDA spike time CSV file into a Pandas Data Frame.

    Returns
    -------
    tida_st : TYPE
        DESCRIPTION.

    """
    
    # Save the resulting data frame as a CSV file containing all recorded TIDA cells 
    path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\Experiment Analysis\\Baseline TIDA\\"
    file = "full_TIDA_spike_times.csv"
    
    tida_st = pd.read_csv(path+file, header=[0,1], index_col=0)
    
    return tida_st


def load_tida_oscillation_properties():
    """
    Function to load the TIDA oscillation properties CSV file into a Pandas Data Frame.

    Returns
    -------
    tida_st : TYPE
        DESCRIPTION.

    """
    
    # Save the resulting data frame as a CSV file containing all recorded TIDA cells 
    path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\Experiment Analysis\\Baseline TIDA\\"
    file = "full_TIDA_oscillation_properties.csv"
    
    osc_props = pd.read_csv(path+file, header=0, index_col=[0,1])
    
    return osc_props


# %% GENERAL FUNCTIONS

def df_between(df, t0, t1):
    """
    Functions creates a DF of spike times for spikes in **df** with times 
    between **t0** and **t1**.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    t0 : TYPE
        DESCRIPTION.
    t1 : TYPE
        DESCRIPTION.

    Returns
    -------
    df_section : TYPE
        DESCRIPTION.

    """
    # create a data frame of spike times between t0 and t1
    df_section_dict={}
    for col in df:
        s = df[col]
        df_section_dict[col] = s[s.between(t0, t1)]
    df_section = pd.DataFrame(df_section_dict)

    return df_section




def h_index_from_burst_results(index_list):
    """
    This function creates a heirecial index, suitable for the normal hDF, using 
    the index of the burst results DF.
    
    For example,
    "yymmdd_s1_e101u1" --> ("yymmdd_s1", "e101u1")

    Parameters
    ----------
    x : Series, DataFrame etc 
        A NON-heirecial data structure with index string structure shown above.

    Returns
    -------
    index : pandas heirecial index
        DESCRIPTION.

    """

    h_index=[]
    for string in index_list:
        rec_string = string[:9]
        unit_string = string[10:]

        h_index.append((rec_string, unit_string))

    index = pd.MultiIndex.from_tuples(h_index)
    
    return index



def burst_index_from_h_index(index_list):
    """
    Function that takes a heirecial index and converts it to the index used in 
    the burst analysis files
    
    For example,
    ("yymmdd_s1", "e101u1") --> "yymmdd_s1_e101u1"
    

    Parameters
    ----------
    index_list : TYPE
        DESCRIPTION.

    Returns
    -------
    b_index : TYPE
        DESCRIPTION.

    """
        
    b_index = []
    for h_index in index_list:
        b_index.append( h_index[0]+"_"+h_index[1] )
        
    return b_index



def firing_rates(stimes, bin_width=1):
  
    # Define the bins
    max_time = int(stimes.max().max()) + bin_width
    bins = np.arange(0, max_time, bin_width)
    
    # Calculate firing rate histograms and store in a list
    firing_rates_list = []
    for neuron in stimes.columns:
        hist, _ = np.histogram(stimes[neuron], bins=bins)
        firing_rate = hist / bin_width  # Firing rate per second
        firing_rates_list.append(firing_rate)

    # Create a DataFrame from the list of firing rates
    firing_rates = pd.DataFrame(firing_rates_list, index=stimes.columns).transpose()
    
    # Set the index - this determines how data is plotted
    # (i+0.5) plots at the center of the bin
    # (i+1) plots at righthand side (point represents the last bin_width worth of data)
    firing_rates.index = [(i+1) * bin_width for i in firing_rates.index]
        
    return firing_rates



def remove_outliers_iqr(data, threshold=1.5):
    
    if isinstance(data, np.ndarray):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        
        
    if isinstance(data, pd.Series):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
    return filtered_data


def electrode_number(s):
    """
    Given a unit key such as 'e21u1' or 'e106u3', this function will extract
    the electrode number, i.e. 21 or 106, in this example.

    Parameters
    ----------
    s : str
        Unit string.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    pattern = r"e(\d{2,3})u(\d)"
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    else:
        return None


def values_below_df_diagonal(matrix):
    """
    Given a square matrix, this function generates a list of all
    elements below the leading diagonal.
    
    """
    values = []
    n = len(matrix)
    for i in range(n):
        for j in range(i):
            values.append(matrix[i][j])
    return values


def diagonal_mask(df, include_diag=False):
    """
    Function to replace a square data frame with NaNs above the leading
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    include_diag : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # set the value of k
    if include_diag==True:
        k=0
    elif include_diag==False:
        k=-1

    # Create a mask for elements below the diagonal
    size = df.shape[0]
    mask = np.tril(np.ones((size, size), dtype=bool), k=0)

    # Apply the mask
    return df.where(mask)


def dict_of_lists_to_list(dict_of_lists):
    """
    Given a dictionary of lists, such as
    
    my_dict = {
    'a': [1, 2, 3],
    'b': [4, 5],
    'c': [6, 7, 8, 9] },
    
    this function stores every element in a list, e.g: [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Parameters
    ----------
    dict_of_lists : TYPE
        DESCRIPTION.

    Returns
    -------
    all_elements : TYPE
        DESCRIPTION.

    """
    # Initialize an empty list to store all the elements
    all_elements = []

    # Loop through each list in the dictionary and extend all_elements
    # list with each list's items
    for items in dict_of_lists.values():
        all_elements.extend(items)

    return all_elements


def find_common_list_elements(list1, list2):
    common_elements = []
    for item in list1:
        if item in list2 and item not in common_elements:
            common_elements.append(item)
    return common_elements


# %% BURST ANALYSIS



def run_burst_analysis(doc, pars='default', template="burst_interval_01", 
                       channel_str_length=13):
    """
    Function to run a NeuroExplorer burst analysis over the whole doc file 
    and store both the full burst analysis data and the summary in a data frame.
    
    channel_str_length = 13 for recording files
    channel_str_length = 16 for hDF files

    Parameters
    ----------
    doc : NEX doc file
        Doc file to analyize .
    pars : dict
        Dictionary of paramter names and values for the particular analysis.
    template : str
        Name of the template to use. Default in the inverval template

    Returns
    -------
    burst_res : Data Frame
        DESCRIPTION.
    burst_sum : TYPE
        DESCRIPTION.

    """

    if pars=='default':
        pars = {"Max Int. (sec)":  1.5,
                "Max End Int.": 2,
                "Min Interburst Int.": 2,
                "Min Burst Duration": 0.1,
                "Min Num. Spikes": 3
                }
                   

    # set algorithm parameters by modifying the exsisting template parameters
    for par_name in pars:
        nex.ModifyTemplate(doc, template, par_name, str(pars[par_name]))
        
    # # set the time to analyize over
    # if tspan!=None:
    #     print('selecting window of data')
    #     nex.ModifyTemplate(doc, template, "Select Data From (sec)", "0")
    #     nex.ModifyTemplate(doc, template, "Select Data To (sec)",   "100")

    # run the analysis
    nex.ApplyTemplate(doc, template)
    results = doc.GetAllNumericalResults()
    summary = doc.GetAllNumResSummaryData()

    # make lists of column names for results and summary files
    res_col_names = doc.GetNumResColumnNames()
    sum_col_names = doc.GetNumResSummaryColumnNames() 

    # data frame for results file
    results = np.transpose(results)
    results_df = pd.DataFrame(results, columns=res_col_names, dtype=float)
    burst_res = convert_to_hDF(results_df, channel_str_length=channel_str_length)

    # data frame for summary file
    summary = np.transpose(summary)
    burst_sum = pd.DataFrame(summary[:,1:], columns=sum_col_names[1:],
                             index=summary[:,0], dtype=float)
    
    return burst_res, burst_sum    






def convert_to_hDF(df, channel_str_length=7, num_burst_props=8):
    """
    Function that imports Burst Analysis data from excel (orginally from
    NeuroExplorer) and stores the data in a hierarchical DataFrame (FD) in which the
    Burst Paramters form the first layer of column names.

    The output UNITS is the list of units that have been analysed, useful
    for a further function
    
    DataFrame organized as:

                  Duaration | End | Start | Start | MeanISI | PeakFreq | #Spikes
               ch_a,...,ch_n
                 ------------------------------------------------------------
    burst_index |

    

    Parameters
    ----------
    df : Data Frame
        Data frame of full burst results from NeuroExplorer.
    channel_str_length : int, optional
        Length of the string that makes the channel name. The default is 7 (for MCS).
    num_burst_props : int, optional
        The number of burst properties that NE spits out. Currently, it is 8.

    Returns
    -------
    None.

    """


    # First column names (the units)
    units = []
    for i in range(0, int((len(df.columns)+1)/num_burst_props)):
        units.append(df.columns[num_burst_props*i][0:channel_str_length].strip())

    # Second column names (burst paramters)
    burst_params = []
    for i in range(0, num_burst_props):
        burst_params.append(df.columns[i][channel_str_length:].strip())

    # Form MultiIndex column headers
    df.columns = pd.MultiIndex.from_product([units, burst_params])

#    # Filter out the non-rythmic channels
#    df = df[rhythmic_channels.columns.tolist()]
#    # Replave the units definition
#    units = rhythmic_channels.columns.tolist()

    # Swap the column levels so that top column level is the paramters
    fd = df.swaplevel(axis=1)
    fd = fd.sort_index(axis=1)

    return fd





def burst_analysis_between(hdf, t0, t1, results_path=None, close_nex_doc=True, pars='default', channel_str_length=16):
    """
    Function that runs the ***run_burst_analysis*** function on a section of the
    data in **hdf** between the times **t0** and **t1**. 

    Parameters
    ----------
    hdf : TYPE
        DESCRIPTION.
    t0 : TYPE
        DESCRIPTION.
    t1 : TYPE
        DESCRIPTION.
    results_path : TYPE, optional
        Path of the directory in which to save the SUM and RES tables. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.
    summ : TYPE
        DESCRIPTION.

    """

    # create a stime data frame between two times
    df = df_between(hdf, t0, t1)

    # save the time-filtered data frame as a NEX file
    save_path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\Code\\Data Analysis\\KiloSort pipeline\\Test burst algorithm\\Temporary NEX files\\"
    file_name = "burst_sample_hdf_between_"+str(t0)+"_"+str(t1)
    stimes_from_DF_2_NEX(df, file_name, save_path=save_path)

    # load the NEX doc and set the start time
    doc = nex.OpenDocument(save_path+file_name+".nex")
    nex.SetDocStartTime(doc, t0)
    
    # run the NEX burst analysis
    res, summ = run_burst_analysis(doc, pars=pars, template="burst_interval_01", channel_str_length=channel_str_length)

    # save the burst results and summary
    if results_path==None:
        res.to_csv("burst_results_between_"+str(t0)+"_"+str(t1)+".csv")
        summ.to_csv("burst_summary_between_"+str(t0)+"_"+str(t1)+".csv")

    else:
        res.to_csv(results_path+"burst_results_between_"+str(t0)+"_"+str(t1)+".csv")
        summ.to_csv(results_path+"burst_summary_between_"+str(t0)+"_"+str(t1)+".csv")

    # delete the NEX file
    os.remove(save_path+file_name+".nex")
    
    # close the NEX document
    if close_nex_doc==True:
        nex.CloseDocument(doc)
    
    return res, summ





def baseline_burst_stats(fd, bl_window, units):
    # Function to calculate the burst statistics (mean & STD) of the burst
    # parameters calculated in NeuroExplorer. DataFrame (BURST_STATS) organized
    # such that the top column is the paramters, lower column is the units

    # Burst parameters for statistical analysis
    params = ["BurstDuration", "SpikesInBurst",
              "MeanISIinBurst", "PeakFreqInBurst"]

    # Make baseline vs drug comparison DataFrame
    burst_stats = pd.DataFrame(index=range(
        2), columns=range(len(params)*len(units)))
    burst_stats.index = pd.Index(['mean', 'std'])
    burst_stats.columns = pd.MultiIndex.from_product([params, units])
    burst_stats = burst_stats.sort_index(axis=1)

    # Assign statistics to DataFrame
    for param in params:

        # Baseline measurments
        bl_param = fd[param][(fd["BurstStart"] > bl_window[0]) & (
            fd["BurstStart"] < bl_window[1])]
        # Watch out that index is in the right order on either side !!
        burst_stats.loc["mean", param] = bl_param.mean().values
        burst_stats.loc["std", param] = bl_param.std().values

    return burst_stats



def burst_metric(summary, hdf=True):
    """
    Function that calculates the burst metric M=S\eta from the NEX-MI summary
    data     

    Parameters
    ----------
    summary : TYPE
        DESCRIPTION.

    Returns
    -------
    metric : TYPE
        DESCRIPTION.

    """
    # calculate the series for all neurons in summary.index and sort by metric value
    metric = summary['Mean Burst Surprise'] * summary['% of Spikes in Bursts']/100
    metric = metric.sort_values()
    # if the original file is hDF then chnage the index accordingly
    if hdf==True:
        metric.index = h_index_from_burst_results(metric.index.to_list())
    
    return metric


def burst_detection(summary, m_threshold=1.5, hdf=True):
    """
    Function that determines the burstings neurons from a data frame of spike 
    times
    
    A burst metric for each time series is calculated and neurons that exceed
    the metric, as detailed in **summary** are counted as bursting neurons
    
    

    Parameters
    ----------
    summary : TYPE
        DESCRIPTION.
    m_threshold : TYPE, optional
        DESCRIPTION. The default is 1.5.
    hdf : Bool, optional
        Whether or not the stimes DF originally analized is heirachical
        or not. The default is True.

    Returns
    -------
    bursting_neurons : TYPE
        DESCRIPTION.
    M_bursting : TYPE
        DESCRIPTION.

    """
    # calculate the metric 
    M = burst_metric(summary, hdf=hdf)
    
    # threshold the neurons
    bursting_neurons = M[M>m_threshold].index.to_list()
    M_bursting = M[M>m_threshold]
    
    return bursting_neurons, M_bursting





def burst_analysis_across_conditions(tida_hdf, time_windows, results_path_dict,
                                     conditions=['control', 'response']):
    """
    Function analizes bursts at 2 or 3 time windows for each recording separatly
    
    2 for Control & Response, 3 for Control, Response and Wash
    
    The time that each recordin gets analized is specified by **time_windows**

    Parameters
    ----------
    tida_hdf : TYPE
        DESCRIPTION.
    time_windows : TYPE
        Dict of time windows such that time_window[rec][cond] = [t0, t1].
    results_path_dict : TYPE
        Dict of paths to save the analysis CSV file to for each condition.
    conditions : TYPE, optional
        DESCRIPTION. The default is ['control', 'response'].

    Returns
    -------
    Res : TYPE
        DESCRIPTION.
    Sum : TYPE
        DESCRIPTION.

    """

    # dictionaries for storing results
    Res={}
    Sum={}


    # loop through each recording that contains TIDA cells
    recordings_containing_tida_cells = tida_hdf.columns.get_level_values(0).unique()
    for rec in recordings_containing_tida_cells:

        # get a hDF of TIDA neurons for a single recording
        hdf_rec = tida_hdf.loc[:, (rec, slice(None))]

        # loop through each condition and perform analysis at the designated time window 
        Res[rec]={}
        Sum[rec]={}
        for i, cond in enumerate(conditions):
            # control burst analysis
            t0 = time_windows[rec][cond][0]
            t1 = time_windows[rec][cond][1]
            results_path = results_path_dict[cond]
            res, summ = burst_analysis_between(hdf_rec, t0, t1, results_path=results_path+rec+"_", close_nex_doc=True, pars='default')

            Res[rec][cond] = res
            Sum[rec][cond] = summ
            
            
    return Res, Sum

def get_property_from_summary(summ, prop):

    # the measurment is some function of the burst summary
    if   prop == 'Surprise':
        meas = summ['Mean Burst Surprise']
    
    elif prop == 'Metric':
        meas = burst_metric(summ)
        meas.index = burst_index_from_h_index(meas.index)
    
    elif prop == 'Duration':
        meas = summ['Mean Burst Duration']
        
    elif prop == 'Mean Freq.':
        meas = summ['Mean Freq.']
        
    elif prop == 'IBI':
        meas = summ['Mean Interburst Interval']
        
    elif prop == 'ISI':
        meas = summ['Mean ISI in Burst']
    
    return meas
            
            



def burst_measurement_across_conditions(Sum, method='Surprise', conditions=[]):
    """
    Function that measures a property of the burst by seaching the summary file
    

    Parameters
    ----------
    Sum : Dict of DataFrames
        Dict of data frames where each DF is the summary file for a recording 
        under a condition, eg Sum[rec][cond]=summ.
    method : TYPE, optional
        DESCRIPTION. The default is 'Surprise'.

    Returns
    -------
    measure : TYPE
        DESCRIPTION.

    """
    
    # set up some objects for storing data
    first_key = next(iter(Sum.keys())) # # recording name
    conditions = list(Sum[first_key])
    measure={}
    for cond in conditions: measure[cond]=[]


    # loop through each recording and each condition
    for rec in Sum:
        for cond in Sum[rec]:
            
            # calculate the burst property and store it in a dictionary
            summ = Sum[rec][cond]
            meas = get_property_from_summary(summ, prop=method)
            measure[cond].append( meas )



    # OUTSIDE LOOP concat measurments into a Series for each condition
    list_for_dataframe=[]
    for cond in conditions:
        measure[cond] = pd.concat(measure[cond])
        measure[cond].index = h_index_from_burst_results(measure[cond].index)
        list_for_dataframe.append( measure[cond] )

    # merge series togther into a data frame
    measure = pd.DataFrame(list_for_dataframe, index=conditions).transpose()
    
    # index=['control', 'response']
    return measure



def rolling_window_times(window_length=200, overlap=50, t_end=8000, t_start=0):
    """
    Function that generates a list of [t_start, t_end] times that define the
    boundary of a time window.
    

    Parameters
    ----------
    window_length : TYPE, optional
        DESCRIPTION. The default is 200.
    overlap : TYPE, optional
        DESCRIPTION. The default is 50.
    t_end : TYPE, optional
        DESCRIPTION. The default is 8000.
    t_start : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    t_array : TYPE
        DESCRIPTION.

    """
    
    # print error
    if overlap > window_length:
        print('Error: Overlap > window_length')

    # define amount to add after every interation 
    step = window_length - overlap

    t_array = [[0, t_start+window_length]]
    t_final=0;   i=1
    while t_final < t_end:

        # calulate the time window
        t0 = t_array[-1][0] + step
        t1 = t0 + window_length

        # update list
        t_array.append( [t0, t1] )

        # update loop parameters
        t_final = t1
        i = i+1
        
    return t_array



def rolling_burst_analysis(hdf, time_windows):
    """
    Function that performs burst analysis for in every time window defined in 
    **time_windows**.
    
    The output is a dictionary labelled by the left-hand window boundary

    Parameters
    ----------
    hdf : TYPE
        DESCRIPTION.
    time_windows : TYPE
        DESCRIPTION.

    Returns
    -------
    sum_t : TYPE
        DESCRIPTION.

    """
    # loop through the time windows and perform burst analysis within
    sum_t={}
    for window in time_windows:
        t0 = window[0]
        t1 = window[1]
        path = r"C:/Users/kh19883/OneDrive - University of Bristol/Documents/PhD Neural Dynamics/ARC-Bursting/Code/Data Analysis/KiloSort pipeline/Test burst algorithm/Burst result files/"
        res, summ = burst_analysis_between(hdf, t0, t1, results_path=path, close_nex_doc=True, pars='default')

        # save summary in a dict
        sum_t[t0] = summ
        
    return sum_t


def burst_prop_time_series(sum_t, prop='Duration'):
    """
    Function that generates a time series of a burst property using the results 
    of the rolling burst analysis.

    Parameters
    ----------
    sum_t : TYPE
        DESCRIPTION.
    prop : TYPE, optional
        DESCRIPTION. The default is 'Duration'.

    Returns
    -------
    prop_ts : TYPE
        DESCRIPTION.

    """
    
    prop_ts={}
    for t0, summ in sum_t.items():    
        #metric_ts[t0] = ma.burst_metric(summ)
        prop_ts[t0] = get_property_from_summary(summ, prop)

    # convert dictionary into a data frame (columns are times)
    prop_ts = pd.concat(prop_ts, axis=1)
    
    # make index hierachical if it is not already
    if not isinstance(prop_ts.index, pd.MultiIndex):
        prop_ts.index = h_index_from_burst_results(prop_ts.index)

    
    return prop_ts




# %% MANNUAL SURPRISE
from scipy.stats import poisson


def calculate_spiketrain_surprise(spike_times, burst_starts, burst_ends):
    """
    Function that calculates the mean response of a spike train 

    Parameters
    ----------
    spike_times : list
        List of spike times.
    burst_starts : list
        List of burst start times.
    burst_ends : list
        List of burst end times.

    Returns
    -------
    surprise_mean : series
        .
    surprise_std : series 
        DESCRIPTION.

    """

    # define quantities for calculating lambda
    total_duration = spike_times[-1] - spike_times[0]
    average_rate = len(spike_times) / total_duration

    
    Surprise=[]
    # Calculate Poisson surprise for each burst
    for start, end in zip(burst_starts, burst_ends):
        # Count spikes within the burst interval
        spikes_in_burst = [spike for spike in spike_times if start <= spike <= end]

        actual_spikes = len(spikes_in_burst)

        # Calculate expected spikes under Poisson assumption
        lambda_value = lambda_value = average_rate * (end - start)

        # Calculate Poisson surprise
        poisson_pmf = poisson.pmf(actual_spikes, lambda_value)
        surprise = -np.log10(poisson_pmf)

        Surprise.append(surprise)

    surprise_mean = np.mean(Surprise)
    surprise_std = np.std(Surprise)
    
    return surprise_mean, surprise_std



def caclulate_hdf_surprise(hdf, Res, time_windows, condition='control'):
    
    """
    Function that calculates the mean suprise of multiple spike trains organised 
    into a hDF.

    Parameters
    ----------
    hdf : TYPE
        DESCRIPTION.
    Res : Dict
        res = Res[rec][cond].
    time_windows : dict
        [t0, t1] = time_windows[rec][cond].
    condition : str, optional
        Which condition to calculate for. The default is 'control'.

    Returns
    -------
    surprise_mean : TYPE
        DESCRIPTION.
    surprise_std : TYPE
        DESCRIPTION.

    """


    surprise_mean = pd.Series(index=hdf.columns, dtype=float)
    surprise_std = pd.Series(index=hdf.columns, dtype=float)

    for neuron in hdf.columns:

        # neuron name in the burst index style
        rec = neuron[0]
        res_neuron_name = rec+"_"+neuron[1]
        
        # get the times to calculate the surprise between
        t0 = time_windows[rec][condition][0]
        t1 = time_windows[rec][condition][1]
        
        # get the spikes that are within the time window and make a list 
        spike_times = hdf[neuron]
        spike_times = spike_times[spike_times.between(t0, t1)].to_list()

        # make a list of burst start and end times
        res = Res[rec][condition]
        burst_starts = res[('BurstStart', res_neuron_name)].dropna().to_list()
        burst_ends   = res[('BurstEnd', res_neuron_name)].dropna().to_list()

        # calculate the surprise
        surp_mean, surp_std =  calculate_spiketrain_surprise(spike_times, burst_starts, burst_ends)

        surprise_mean.loc[neuron] = surp_mean
        surprise_std.loc[neuron] = surp_std
        
    return surprise_mean, surprise_std



def surprise_response_bool(hdf, Res, time_windows, std_threshold=3):
    """
    Function to detect a significant surprise response between conditions.
    Detection is based upon surpassing a STD threshold

    Parameters
    ----------
    hdf : TYPE
        DESCRIPTION.
    Res : TYPE
        DESCRIPTION.
    time_windows : TYPE
        DESCRIPTION.
    std_threshold : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    response : TYPE
        DESCRIPTION.

    """

    conditions =['control', 'response']


    surprise_mean={}
    surprise_std={}
    for cond in conditions:
        surprise_mean[cond], surprise_std[cond] = caclulate_hdf_surprise(hdf, Res, time_windows, condition=cond)


    # series for storing the response boolean in
    response = pd.Series([0]*len(surprise_mean['control']), index=surprise_mean['control'].index, dtype=int)

    # response occurs when the difference in surprises crosses a threshold
    difference = surprise_mean['response'] - surprise_mean['control']
    responsive_units = abs(difference) > std_threshold * surprise_std['control']

    # assign boolen response as increasing (+1), decreasing (-1), or not changing (0)
    response[responsive_units] = np.sign(difference)
    
    return response





# %% PATCH DATA


def wholeCell_cellAttached_comparison(common_exps_only=False, plot_comparison=True):
    
    # load the whole cell and the cell-attached burst analysis
    path = r'C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\Patch Data\\Lyons\\Analysis\\'
    # whole-cell
    res_w = pd.read_csv(path+'whole_cell_MIBurst_results.csv', index_col=0, header=[0,1])
    sum_w = pd.read_csv(path+'whole_cell_MIBurst_summary.csv', index_col=0)
    # cell attached
    res_c = pd.read_csv(path+'cell_attached_MIBurst_results.csv', index_col=0, header=[0,1])
    sum_c = pd.read_csv(path+'cell_attached_MIBurst_summary.csv', index_col=0)

    # create a list of experiments common to both data
    w_exps = sum_w.index.to_list()
    c_exps = sum_c.index.to_list()
    common_exps = [item for item in w_exps if item in c_exps]


    # only analize the common data if statistical comparison is being made
    if common_exps_only==True:
        sum_w = sum_w.loc[common_exps]
        sum_c = sum_c.loc[common_exps]



    ### create lists of important burst quantites
    # mean duration of a burst
    duration_w = sum_w['Mean Burst Duration'].to_list()
    duration_c = sum_c['Mean Burst Duration'].to_list()
    # mean number of spikes in a burst
    nspikes_w = sum_w['Mean Spikes in Burst'].to_list()
    nspikes_c = sum_c['Mean Spikes in Burst'].to_list()
    # mean ISI within a burst 
    isi_w = sum_w['Mean ISI in Burst'].to_list()
    isi_c = sum_c['Mean ISI in Burst'].to_list()
    # mean IBI
    ibi_w = sum_w['Mean Interburst Interval'].to_list()
    ibi_c = sum_c['Mean Interburst Interval'].to_list()
    # variabtion in burst duration
    duration_std_w = sum_w['St. Dev. of Burst Duration'].to_list()
    duration_std_c = sum_c['St. Dev. of Burst Duration'].to_list()
    # variation in IBI
    ibi_std_w = sum_w['St. Dev. of Interburst Int.'].to_list()
    ibi_std_c = sum_c['St. Dev. of Interburst Int.'].to_list()
    

    data = [[duration_w, duration_c],
           [nspikes_w, nspikes_c],
           [isi_w, isi_c],
           [ibi_w, ibi_c],
           [duration_std_w, duration_std_c],
           [ibi_std_w, ibi_std_c]]
    labels = ['Burst duration', 'Number of spikes per burst', 'ISI within a burst', 'IBI', 'Burst Duration STD', 'IBI STD']

    
    # plot the comparison between data
    if plot_comparison==True:
        fig, Ax = plt.subplots(1, len(data), figsize=(15,3))
        for i, data_pair in enumerate(data):
            ax = Ax[i]
            ax = sns.boxplot(data  =data_pair, ax=ax)
            ax = sns.swarmplot(data=data_pair, color='k', size=3, ax=ax)
            ax.set_title(labels[i])
            ax.set_xticklabels(['WC', 'CA'])


    # save as a CSV
    l=[]
    for x in data:
        for y in x:
            l.append(y)
    df = pd.DataFrame(l).transpose()
    df.columns = pd.MultiIndex.from_product([labels, ['wc', 'ca']])

    if common_exps_only==True:
        df.to_csv(path+"burst_comparison_common.csv")
    else:
        df.to_csv(path+"burst_comparison_all.csv")
        
        
        
    return df




def get_patchclamp_burst_property_means(prior='pooled', remove_outliers=False,
                                        outlier_threshold=1.5):
    # function to remove nan values
    def remove_nan_values(input_array):
        cleaned_array = input_array[~np.isnan(input_array)]
        return cleaned_array
    
    path = r'C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\Patch Data\\Lyons\\Analysis\\'
    df = pd.read_csv(path+"burst_comparison_all.csv", index_col=0, header=[0,1])
    
    
    # isolate only the data we want to consider
    if   prior == 'pooled':
        df = df
        
    elif prior == 'best':
        df = df.drop(('ISI within a burst', 'wc'), axis=1) # results of stats analysis
        df = df.drop(('Number of spikes per burst', 'wc'), axis=1)
        
    elif prior == 'whole-cell':
        df = df.drop(columns=df.columns[df.columns.get_level_values(1)=='ca'])
        
    elif prior == 'cell-attached':
        df = df.drop(columns=df.columns[df.columns.get_level_values(1)=='wc'])
        
        


    # create a lists of pooled data
    data={}
    data['duration']     = remove_nan_values(df['Burst duration'].values.flatten())
    data['nspikes']      = remove_nan_values(df['Number of spikes per burst'].values.flatten())
    data['isi']          = remove_nan_values(df['ISI within a burst'].values.flatten())
    data['ibi']          = remove_nan_values(df['IBI'].values.flatten())
    data['duration_std'] = remove_nan_values(df['Burst Duration STD'].values.flatten())
    data['ibi_std']      = remove_nan_values(df['IBI STD'].values.flatten())
    
    # remove big outliers from the data 
    if remove_outliers==True:
        for prop in data:
            data[prop] = remove_outliers_iqr(data[prop], threshold=outlier_threshold)
    
    return data






# %% TIDA SCORE



def get_data_intervals(method='range', remove_outliers=True, outlier_threshold=1.5,
                       data=None):
    """
    Function that generates an interval for the patch-clamp burst properties. 
    
    The simplest interval is the range, which is just the (min, max) of the 
    measured properties.

    Parameters
    ----------
    method : TYPE, optional
        DESCRIPTION. The default is 'range'.

    Returns
    -------
    intervals : TYPE
        DESCRIPTION.

    """
    
    # get the patch clamp data
    if data == None:
        data = get_patchclamp_burst_property_means(prior='best',
                                                   remove_outliers=remove_outliers,
                                                   outlier_threshold=outlier_threshold)
    else:
        data = data

    # create an interval for every property in the data
    intervals={}
    for prop in data:
        
        if method=='range': # simple range from the minimum to the maximum of the data
            intervals[prop] = (np.min(data[prop]), np.max(data[prop]))
            
        # elif method=='confidence':
        #     intervals[prop] = confidence_interval(data[prop], confidence_level=0.95)
            
    return intervals




def interval_tida_score(bursting_res, intervals):
    
    tida_score={}
    percents ={}
    
    bursting_neurons = list(bursting_res.columns.get_level_values(1).unique())
    
    # loop through each bursting neuron 
    for neuron in bursting_neurons:

        res_neuron = bursting_res.xs(neuron, level=1, axis=1)

        # get a true/false for each burst having a property within the ranges for TIDA neurons
        boolean={}
        boolean['duration'] = res_neuron['BurstDuration'].dropna().between(0, intervals['duration'][1] )    # intervals['duration'][0]
        boolean['nspikes']  = res_neuron['SpikesInBurst'].dropna().between(intervals['nspikes'][0], intervals['nspikes'][1] )
        boolean['isi']      = res_neuron['MeanISIinBurst'].dropna().between(intervals['isi'][0], intervals['isi'][1] )
        # IBI is a bit different as we have to calculate it
        ibi = res_neuron['BurstStart'] - res_neuron['BurstEnd'].shift(periods=1) 
        boolean['ibi'] = ibi.dropna().between(intervals['ibi'][0], intervals['ibi'][1])

        # calculate the percent of bursts that have TIDA-like properties
        percent_in_interval={}
        for prop in boolean:
            percent_in_interval[prop] = boolean[prop].sum() / len(boolean[prop])

        # TIDA score is the proudct of percentages
        tida_score[neuron] = np.prod( list(percent_in_interval.values()) )

        # saving the original percentages is useful for debugging
        percents[neuron] = percent_in_interval

    # convert to a series
    tida_score = pd.Series(tida_score).sort_values()
    tida_score.index = h_index_from_burst_results(tida_score.index)
    
    return tida_score



def tida_interval_detection(bursting_res, interval_tida_threshold=0.5):
    
    # generate the intervals that define a TIDA cell
    intervals = get_data_intervals(method='range', remove_outliers=True, outlier_threshold=1.5)
    
    # calculate a TIDA score using the burst results and the intervals 
    tida_score = interval_tida_score(bursting_res, intervals)
    tida_score = tida_score[tida_score > interval_tida_threshold] 
    
    # define TIDAs as those cells that surpass a threshold
    tida_neurons = tida_score.index.to_list()
      
    
    return tida_neurons, tida_score













def generate_kdes(prior='best', plot_result=True, ax=None, remove_outliers=True,
                  outlier_threshold=1.5, data=None):
    """
    Function that generates the kernal density estimate (KDE) for each type of 
    burst property (duration, number of spikes, mean ISI and mean IBI).
    
    Each KDE is a function that returns the likelihood of a value being drawn 
    from that distribution. Each KDE is stored in a dictionary.

    Parameters
    ----------
    prior : TYPE, optional
        DESCRIPTION. The default is 'pooled'.
    plot_result : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if data == None:
        # get lists of data about the properties of patch-clamped TIDA cell bursts 
        data = get_patchclamp_burst_property_means(prior=prior,
                                                   remove_outliers=remove_outliers,
                                                   outlier_threshold=outlier_threshold)
    else:
        # use the data provided to the function
        data = data
    
    # loop through distributions and calculate KDE
    kde={}
    for burst_property, array in data.items():

        kde[burst_property] = gaussian_kde(array)

    if plot_result==True:
        fig, Ax = plt.subplots(1, len(data), figsize=(15, 3), tight_layout=True)
        labels = ['Burst duration [s]', 'Number of spikes per burst',
                  'ISI within a burst [s]', 'IBI [s]', 'Burst duration STD [s]',
                  'IBI STD [s]']
        for i, (burst_property, array) in enumerate(data.items()):  

            ax = Ax[i]
            # Generate points for plotting the estimated PDF
            x = np.linspace(min(array), max(array), 1000)
            estimated_pdf = kde[burst_property](x)

            # Plot the estimated PDF
            ax.plot(x, estimated_pdf, label='Estimated PDF')
            ax.hist(array, density=True, alpha=0.5, bins=10, label='Histogram')
            ax.set_xlabel(labels[i])
            if i==0: ax.set_ylabel('Density')
            #ax.set_title(labels[i])

        ax.legend()
        plt.show()


    return kde



def tida_likelihood(summary, kde_dict, weights='even', log=True,
                    Wd=1, Wn=1, Wi=1, Wb=1, Wds=1, Wbs=1):
    """
    This function defines a TIDA score for every neuron in the **summary** DF.
    
    For each neuron, the likelihood of a burst property measurment being drawn
    from the confirmed/patch-clamp TIDA distribution is calculated.
    
    The final TIDA score is the product of all likelihoods. A log transform may 
    or may not be applied.

    Parameters
    ----------
    summary : TYPE
        DESCRIPTION.
    kde_dict : TYPE
        DESCRIPTION.
    log : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    tida_score : TYPE
        DESCRIPTION.

    """
    
    tida_score={}
    for neuron in summary.index:

        # calculate the likihoods for each burst measurment
        duration_likelihood = kde_dict['duration'](summary.loc[neuron, 'Mean Burst Duration'])
        nspikes_likelihood  = kde_dict['nspikes'](summary.loc[neuron, 'Mean Spikes in Burst'])
        isi_likelihood      = kde_dict['isi'](summary.loc[neuron, 'Mean ISI in Burst'])
        ibi_likelihood      = kde_dict['ibi'](summary.loc[neuron, 'Mean Interburst Interval'])
        dur_std_likelihood  = kde_dict['duration_std'](summary.loc[neuron, 'St. Dev. of Burst Duration'])
        ibi_std_likelihood  = kde_dict['ibi_std'](summary.loc[neuron, 'St. Dev. of Interburst Int.'])

        # calculate joint likelihood
        likelihood_list=[]
        weights = [Wd, Wn, Wi, Wb, Wds, Wbs]
        lhoods  = [duration_likelihood, nspikes_likelihood, isi_likelihood, ibi_likelihood, dur_std_likelihood, ibi_std_likelihood]
        for i, W in enumerate(weights):
            if W != 0:
                likelihood_list.append( W * lhoods[i][0] )
        joint_likelihood = np.prod(likelihood_list)
            
        #joint_likelihood = (Wd*duration_likelihood) * (Wn*nspikes_likelihood) * (Wi*isi_likelihood) * (Wb*ibi_likelihood) * (Wds*dur_std_likelihood) * (Wbs*ibi_std_likelihood)
        
        # store the likelihoos in a dictionary
        tida_score[neuron] = joint_likelihood

    # concert to a series and return
    tida_score = pd.Series(tida_score).sort_values()
    tida_score.index = h_index_from_burst_results(tida_score.index)
    
    # log transform if the option is selcected
    if log==True:
        tida_score = np.log(tida_score)
    
    return tida_score




def tida_likelihood_detection(summary, log_tida_threshold=-16, log=True, return_all_tida_score=False,
                              Wd=1, Wn=1, Wi=1, Wb=1, Wds=1, Wbs=1):
    """
    Function that detects TIDA neurons from a data frame of burst results summary.
    
    The **summary** DF should be SPECIFICLY bursting neurons
    

    Parameters
    ----------
    summary : TYPE
        DESCRIPTION.
    log_tida_threshold : TYPE, optional
        DESCRIPTION. The default is -16.

    Returns
    -------
    tida_neurons : TYPE
        DESCRIPTION.
    tida_score : TYPE
        DESCRIPTION.

    """
    
    # generate KDEs for TIDA detection
    kde_dict = generate_kdes(plot_result=False)
    
    # calculate a TIDA score for all cells
    tida_score_all = tida_likelihood(summary, kde_dict, log=log,
                                 Wd=Wd, Wn=Wn, Wi=Wi, Wb=Wb, Wds=Wds, Wbs=Wbs)
    
    # cells with a TIDA score that surpass a threshold are classified as TIDA cells
    tida_score = tida_score_all[tida_score_all > log_tida_threshold]
    tida_neurons = tida_score.index.to_list()
    
    
    # can use useful to knowe the TIDA score for all neurons, not just TIDAs, so
    # this larger array can be outputted too
    if return_all_tida_score==True:
        return tida_neurons, tida_score, tida_score_all
    
    elif return_all_tida_score==False:
        return tida_neurons, tida_score
    
    
    
def burst_and_tida_detection(hdf, table,
                             burst_metric_detection_threshold=1.5,
                             tida_detection_method='likelihood',
                             tida_likelihood_detection_threshold=-18,
                             tida_interval_detection_threshold=0.9,
                             plot_tida_raster=True, baseline_start=0):
    """
    Function that detects both bursting neurons and TIDA neurons from a hDF of
    spike times.

    Parameters
    ----------
    hdf : hDF
        hDF of spike times for all neurons in an experiment.
    table : TYPE
        DESCRIPTION.
    burst_metric_detection_threshold : TYPE, optional
        DESCRIPTION. The default is 1.5.
    tida_likelihood_detection_threshold : TYPE, optional
        DESCRIPTION. The default is -18.
    plot_tida_raster : TYPE, optional
        Controls whether to plot the raster plot of TIDA neurons. The default is True.
    baseline_start : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    bursting_neurons : TYPE
        DESCRIPTION.
    tida_neurons : TYPE
        DESCRIPTION.
    tida_score : TYPE
        DESCRIPTION.

    """
    
    # ------------------------ Burst analysis ------------------------ 
    # run burst analysis over the baseline period
    t0 = baseline_start               # baseline start 
    t1 = table['BaselineEnd'].min()   # baseline end
    results_path = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\MEA Data\\2023 Analysis\\Experiment Baseline Burst Analysis"
    res, summ = burst_analysis_between(hdf, t0, t1, results_path=results_path, close_nex_doc=True, pars='default')

    # ------------------------ Burst detection ------------------------ 
    # detect bursting cells
    bursting_neurons, M = burst_detection(summ, m_threshold=burst_metric_detection_threshold)

    # get the burst results data for only the bursting cells
    bursting_sum = summ.loc[burst_index_from_h_index(bursting_neurons)]
    bursting_res = res.loc[:, (slice(None), burst_index_from_h_index(bursting_neurons))]

    # ------------------------ TIDA detection ------------------------ 
    if tida_detection_method=='likelihood':
        # detect TIDA cells using the likelihood method
        tida_neurons, tida_score = tida_likelihood_detection(bursting_sum,
                                                             log_tida_threshold=tida_likelihood_detection_threshold,
                                                            Wbs=1, Wds=1) 
    if tida_detection_method=='interval':
        # detect TIDA cells unsing the interval method
        tida_neurons, tida_score = tida_interval_detection(bursting_res,
                                                           interval_tida_threshold=tida_interval_detection_threshold)
        
    # get TIDA spike trains 
    tida_hdf = hdf[tida_neurons]
    
    # remove any 'TIDA' neurons that do not fire in the first 100s
    col_idx_to_drop = (tida_hdf.iloc[0, :]>100).argmax()
    cols_to_drop = tida_hdf.columns[col_idx_to_drop]
    
    # and redefine the TIDA population
    tida_hdf = tida_hdf.drop(cols_to_drop, axis=1)
    tida_neurons = tida_hdf.columns.to_list()
    tida_score = tida_score.drop(cols_to_drop)

    # get the burst analysis results for the TIDA neurons
    #tida_sum = summ.loc[burst_index_from_h_index(tida_neurons)]
    tida_res = res.loc[:, pd.IndexSlice[:, burst_index_from_h_index(tida_neurons)]]
    
    
    # get a list of recordings that contain TIDA cells
    tida_recs = tida_hdf.columns.get_level_values(0).unique()
    tida_animals = list(set([rec[0:6] for rec in tida_recs]))

    print(str(len(tida_neurons))+' TIDA neurons have been detected across '+str(len(tida_recs))+' recordings and '+str(len(tida_animals))+' animals.')
    # ------------------------ Optional plotting ----------------------- 
    if plot_tida_raster==True:
        # order by TIDA score and plot
        tida_hdf_ordered = tida_hdf.reindex(columns=tida_score.index)
        fig, ax = plt.subplots(figsize=(15, 0.5*tida_hdf.shape[1]))
        ax = burst_raster_plot(tida_hdf_ordered, tida_res, 0, 1000, ax=ax, hdf=True)

        
    return bursting_neurons, tida_neurons, tida_score
    

# %% PERIOD 



# Define a function to find the period of oscillation using Fourier transform
def timeseries_fourier_period(s, sampling_interval='auto'):
    
    # calculate the sampling frequency automatically from the data frame
    if sampling_interval == 'auto':
        if type(s) == pd.Series:
            sampling_interval =  s.index[1] - s.index[0]
        else:
            raise ValueError('The sampling interval must be explicitly \
                             provided because the signal is not a Series object')
                             
                             
    # Perform Fourier transform
    fft_result = np.fft.fft(s - s.mean())
    
    # Find the frequency corresponding to the maximum amplitude
    freqs = np.fft.fftfreq(len(fft_result), d=sampling_interval)
    period_index = np.argmax(np.abs(fft_result))
    dominant_freq = freqs[period_index]
    
    # Calculate the period from the dominant frequency
    period = 1 / abs(dominant_freq)
    return period


def dataframe_fourier_period(df):
    
    # Iterate through each column (excluding 'Time') and find the period
    periods = pd.Series(index=df.columns, dtype='float')
    for neuron in df.columns:
        s = df[neuron]
        period = timeseries_fourier_period(s)
        periods.loc[neuron] = period
        
    return periods


def period_across_conditions(hdf, time_windows, conditions=['control', 'response']):
    
    # make firing rate data frame
    frates = firing_rates(hdf, bin_width=1)
    
    # loop through the conditions
    p_conds={}
    for cond in conditions:
        # loop through the recordings 
        p_recs=[]
        for rec in frates.columns.get_level_values(0).unique():
            # set the times over which to measure the period - depends upon condition and recording 
            t0 = time_windows[rec][cond][0]
            t1 = time_windows[rec][cond][1]
            df = frates.loc[t0:t1, (rec, slice(None))]
            # calculate the period
            p_recs.append(dataframe_fourier_period(df))
        # concat into a series
        p_conds[cond] = pd.concat(p_recs)

    per_measure = pd.concat(p_conds, axis=1)
    return per_measure


# %% RESPONSE 


def summary_property_response_bool(Sum, prop='Duration', std_threshold=3):
    """
    Function that detects a response in one of the properties calculated by the 
    burst analysis (and accessable from the summary data frame)

    Parameters
    ----------
    Sum : dict
        summ = Sum[rec][condition].
    prop : TYPE, optional
        DESCRIPTION. The default is 'Duration'.
    std_threshold : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    response : Series 
        Series of either 1 (increase response), -1 (decrease response) or
        0 (no response)

    """

    property_column     = {'Duration':'Mean Burst Duration',        'IBI':'Mean Interburst Interval'}
    property_std_column = {'Duration':'St. Dev. of Burst Duration', 'IBI':'St. Dev. of Interburst Int.' }


    response_list=[]
    for rec in Sum:
        # find the STD of the property in the control window
        control_avg = Sum[rec]['control'][property_column[prop]]
        control_std = Sum[rec]['control'][property_std_column[prop]]

        response_avg = Sum[rec]['response'][property_column[prop]]

        # series for storing results
        response = pd.Series([0]*len(control_avg), index=control_avg.index, dtype=int, name=prop)

        # define a response as occuring when the property changes by a multiple of its control STD
        difference = response_avg - control_avg
        responsive_units = abs(difference) > std_threshold * control_std

        # store the response as an increase (+1), a decrease (-1) or absent (0)
        response[responsive_units] = np.sign(difference)
        response_list.append(response)

    response = pd.concat(response_list)
    response.index = h_index_from_burst_results(response.index)

    return response




def firing_rate_response_bool(hdf, measure, std_threshold=2.5):

    # calculate baseline firing rates with a 200s window
    frates = firing_rates(hdf, bin_width=200)

    # extract the response firing rates from the summary file
    response_frates = measure['Mean Freq.']['response']

    # take the first 800s of the spike train as the baseline
    baseline_frates = frates.loc[:800, :]

    # baseline mean and STD
    control_frates_mean = baseline_frates.mean().reindex(index=response_frates.index)
    control_frates_std = baseline_frates.std().reindex(index=response_frates.index)

    # series for storing response boolean
    response = pd.Series([0]*len(control_frates_mean), index=control_frates_mean.index)

    # determine whether response is significant
    difference = response_frates - control_frates_mean
    responsive_units = abs(difference) > std_threshold*control_frates_std

    response[responsive_units] = np.sign(difference)
    
    return response




def calculate_boolean_response(hdf, Res, Sum, measure, time_windows, surprise_std_th=2.5,
                               summary_std_th=2.5, freq_std_th=2.5 ):
    """
    Function to detect a response of a spike train across conditions. Several 
    quantities are tested. 

    Parameters
    ----------
    hdf : hDF
        DESCRIPTION.
    Res : dict
        res = Res[rec][condition].
    Sum : TYPE
        Similar for Res
    time_windows : dict
        Similar for Res.
    surprise_std_th : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    response_bool : TYPE
        DESCRIPTION.

    """

    # detect response using properties in the summary file
    summary_properties = ['Duration', 'IBI']
    response_bool_dict={}
    for prop in summary_properties:
        response_bool_dict[prop] = summary_property_response_bool(Sum, prop=prop, std_threshold=summary_std_th)

    # detect a response in the burst surprise (calculated manually) 
    # if surprise_std_th != None:
    #     response_bool_dict['Surprise'] = surprise_response_bool(hdf, Res, time_windows, std_threshold=surprise_std_th)
    
    # detect response in the firing rate
    response_bool_dict['Mean Freq.'] = firing_rate_response_bool(hdf, measure, std_threshold=freq_std_th)

    # concatenate into a data frame
    response_bool = pd.concat(response_bool_dict, axis=1)

    return response_bool




def pos_neg_response(response_measure, prop, plot=True, return_tonic=False, tonic_threshold=100):
    """
    Function that splits the response measure into positive-response neurons 
    and negative response neurons. It also determines how many cells have switched
    to tonic firing using the burst duration.

    Parameters
    ----------
    response_measure : TYPE
        DESCRIPTION.
    prop : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    pos_response : TYPE
        DESCRIPTION.
    neg_response : TYPE
        DESCRIPTION.

    """
    # define negative and positive responses
    pos_response = response_measure[prop][ response_measure[prop]['response'] > response_measure[prop]['control'] ]
    neg_response = response_measure[prop][ response_measure[prop]['response'] < response_measure[prop]['control'] ]

    # need to remove tonic firing neurons from the duration response
    tonic_response=[]
    if prop=='Duration':
        tonic_response = pos_response[pos_response['response']>tonic_threshold]
        pos_response = pos_response.drop(tonic_response.index)


    if prop=='Duration':
        print(prop+' increase: '+str(len(pos_response)+len(tonic_response))+' cells, '+str(len(tonic_response))+' of which go tonic')
    else:
        print(prop+' increase: '+str(len(pos_response))+' cells')
    print(prop+' decrease: '+str(len(neg_response))+' cells')



    if plot==True:
        # plot both the negative and positive responses
        num_axes = sum([not neg_response.empty, not pos_response.empty])
        fig, ax = plt.subplots(1, num_axes, figsize=(2*num_axes, 3), tight_layout=True)
        fig.suptitle(prop) 

        # plot both responses if they exist
        if isinstance(ax, np.ndarray):
            ax[0] = plot_measure_across_conditions(pos_response, ax=ax[0])
            ax[0].set_title('positive')
            ax[1] = plot_measure_across_conditions(neg_response, ax=ax[1])
            ax[1].set_title('negative')

        # only plot the significant response
        else:    
            if not pos_response.empty:
                ax = plot_measure_across_conditions(pos_response, ax=ax)
                ax.set_title('positive')
            else:
                ax = plot_measure_across_conditions(neg_response, ax=ax)
                ax.set_title('negative')
     
    if return_tonic==True:
        return pos_response, neg_response, tonic_response
    else:
        return pos_response, neg_response



# %% STATS


def print_summary_stats(measure, print_numbers=True):
    
    # print some numbers
    if print_numbers==True:
        keys = [key for key in measure.keys()]
        prop = keys[0]
        print('Numbers -----------------------')
        print('N_cells = ',measure[prop].shape[0])
        print('N_recs = ',len(measure[prop].index.get_level_values(0).unique().to_list()))
        print('N_rats = ',len(set([i[0:6] for i in measure[prop].index.get_level_values(0).unique().to_list()])))
        print('\n')

    for prop in measure:
        print(' xxx '+prop+' xxx')
        df = measure[prop]
        for col in df:
            print(col)
            x = df[col].dropna().to_numpy()

            # print the median +- the MAD
            print('median = '+str(np.median(x)),' with MAD =',stats.median_abs_deviation(x))

        print('----------------------------------------------------')  




def remove_difference_outliers(df, outlier_threshold=1.5):
    """
    Function to remove the outliers in difference for the pair measurments in 
    each column of **df**
    

    Parameters
    ----------
    df : Data Frame
        Two columns of paired measurments. Pairing should be indicated by the index.
    outlier_threshold : TYPE, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    df_nooutliers : TYPE
        DESCRIPTION.

    """

    # calculate the differnce
    difference = df.diff(axis=1).iloc[:, 1]
    difference.name = 'differnece'

    # remve outliers
    difference = remove_outliers_iqr(difference, threshold=outlier_threshold)

    # generate paired measurments with differnece no outliers 
    df_nooutliers = df.loc[difference.index]

    return df_nooutliers 



def check_ttest_assumptions(df):
    """
    Function to check the assumptions for the paired t-test. 

    Parameters
    ----------
    df : Data Frame
        Two columns of paired measurments. Pairing should be indicated by the index.

    Returns
    -------
    p_value_shapiro : TYPE
        DESCRIPTION.
    p_value_var : TYPE
        DESCRIPTION.

    """
    # calculate the differnce
    difference = df.diff(axis=1).iloc[:, 1]
    difference.name = 'differnece'

    # check the normality assumption. p>0.05 --> normal distribution
    _, p_value_shapiro = stats.shapiro(difference)
    
    # Check homogeneity of variances using Levene's test
    _, p_value_var = stats.levene(df.iloc[:,0], df.iloc[:,1])
    
    # # print some results
    # if p_value_shapiro >= 0.05: print(':-) Data differences are normally distributed')
    # else: print(':-( Data differences are NOT normally distributed. Consider a transformation of the data')
    
    # if p_value_var >= 0.05: print(':-) Variences are homogenous')
    # else: print(':-( Variences are NOT homogenous. Consider a transformation of the data')
    
    return p_value_shapiro, p_value_var



def paired_ttest(df):
    """
    Function that performs the t-test

    Parameters
    ----------
    df : Data Frame
        Two columns of paired measurments. Pairing should be indicated by the index.

    Returns
    -------
    t_statistic : TYPE
        DESCRIPTION.
    p_value : TYPE
        DESCRIPTION.

    """
    
    t_statistic, p_value = stats.ttest_rel(df.iloc[:,0], df.iloc[:,1])
    
    return t_statistic, p_value






def paired_difference_test(df, transform=None, threshold=0.05):
    
    # set these 
    statistic = None
    p_value = None
    DForZ = None
    
    # remove difference outliers 
    df_nooutliers = remove_difference_outliers(df)

    # transform if necessary
    if transform==None:
        df_trans = df_nooutliers
    if transform=='log':
        df_trans = np.log(df_nooutliers +  1e-5)
    if transform=='sqrt':
        df_trans = np.sqrt(df_nooutliers)

    # check ttest assumptions
    p_value_shapiro, p_value_var = check_ttest_assumptions(df_trans)
   
    if p_value_shapiro < 0.05:
        print('Differences not normally distributed')
    if p_value_var < 0.05:
        print('Error: Variances not homogenous')

    # only perform the t-test if both assumptions are met
    if (p_value_shapiro > 0.05) and (p_value_var > 0.05):
        print('t-test assumptions are met')
        test = 't-test'
        statistic, p_value = stats.ttest_rel(df_trans.iloc[:,0], df_trans.iloc[:,1])
        
    else:
        test = 'Wilcoxon signed-rank test'
        statistic, p_value = stats.wilcoxon(df_trans.iloc[:,0], df_trans.iloc[:,1])
        
    # print a summary
    print(test, 'results: ')
    print(' - statistic = ',statistic)
    print(' - t-test p-value = ',p_value)
    #print(' - FDorZ = ', DForZ)
    if p_value < threshold: # H0 is that they are drawn form the same distribution
        print(' - measurments are statistically different!' )
    else:
        print(' - no statistical difference')
       
          
          
    return statistic, p_value




def check_ind_ttest_assumptions(x, y, alpha=0.05):

    # Assumption 1: Normality
    _, p_value1 = stats.normaltest(x)
    _, p_value2 = stats.normaltest(y)

    normality_assumption_met = (p_value1 > alpha) and (p_value2 > alpha)

    # Assumption 2: Equal Variances (Levene's Test)
    _, p_value_levene = stats.levene(x, y)

    equal_variances_assumption_met = p_value_levene > alpha
    
    
    return [normality_assumption_met, equal_variances_assumption_met]






def independent_difference_test(x, y, transform=None, threshold=0.05, equal_var=True):
    """
    x and y need to be numpy arrays

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    transform : TYPE, optional
        DESCRIPTION. The default is None.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    statistic : TYPE
        DESCRIPTION.
    p_value : TYPE
        DESCRIPTION.

    """
    
    # set these 
    statistic = None
    p_value = None
    
    # remove difference outliers 
    x = remove_outliers_iqr(x)
    y = remove_outliers_iqr(y)

    # transform if necessary
    if transform==None:
        x = x
        y = y
    if transform=='log':
        x = np.log( x + abs(np.min(x)) + 0.1)
        y = np.log( x + abs(np.min(x)) + 0.1)
    if transform=='sqrt':
        x = np.sqrt(x)
        y = np.sqrt(y)

    # check ttest assumptions
    normality_cond, var_cond = check_ind_ttest_assumptions(x, y)
   
    if not normality_cond:
        print('Differences not normally distributed')
    if not var_cond:
        print('Error: Variances not homogenous')

    # only perform the t-test if both assumptions are met
    if normality_cond and var_cond:
        print('t-test assumptions are met')
        test = 't-test'
        statistic, p_value = stats.ttest_ind(x, y, equal_var=equal_var)
        
    else:
        test = 'Mann-Whitney U Test'
        statistic, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
        df = None
        
    # print a summary
    print(test, 'results: ')
    print(' - statistic = ',statistic)
    print(' - t-test p-value = ',p_value)
    #if test == 't-test': print(' - df = ',df)
    if p_value < threshold: # H0 is that they are drawn form the same distribution
        print(' - measurments are statistically different!' )
    else:
        print(' - no statistical difference')
       
          
    return statistic, p_value 

    
    

    

# %% VISUALISATION & PLOTS

def raster_plot(df, t0=0, t1=100, columns_to_plot='all', ax=None, lw=1, ticks_as_names=True,
                color='C0', alpha=1):
    """
    Function to plot a raster diagram for the DF of spike times **df** between 
    t0 and t1 [in seconds]

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    t0 : TYPE, optional
        DESCRIPTION. The default is 0.
    t1 : TYPE, optional
        DESCRIPTION. The default is 100.
    columns : TYPE, optional
        Columns of df to plot. The default is 'all'. A list of column names
        could also be given. 
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if columns_to_plot == 'all':
        columns_to_plot = df.columns

    stime_list=[]
    for col in columns_to_plot:

        # for each neuron, make a list of stime's between t0 and t1
        s = df.loc[:,col][df.loc[:,col].between(t0, t1)]
        l = list(s.dropna())

        # make a list of lists for all neurons to be plotted
        stime_list.append(l)

    if ax==None:
        # create a figure and axes
        fig, ax = plt.subplots()
        
    # plot the raster diagram
    ax.set_xlim(t0, t1)
    ax.eventplot(stime_list, linelengths=0.8, linewidth=lw, color=color)
    
    # set the y-tick labels as the names of the neurons
    if ticks_as_names==True:
        ax.set_yticks(np.arange(len(stime_list)))
        ax.set_yticklabels(columns_to_plot)
    
    
    return ax




def burst_raster_plot(df, res, t0=0, t1=100, columns_to_plot='all', ax=None, lw=1, ticks_as_names=True,
                color=[39/255,64/255,139/255], height=0.85, hdf=True):
    """
    Function that plots both the raster diagram and the boxes showing when
    bursts start and end.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    res : TYPE
        DESCRIPTION.
    t0 : TYPE, optional
        DESCRIPTION. The default is 0.
    t1 : TYPE, optional
        DESCRIPTION. The default is 100.
    columns_to_plot : TYPE, optional
        DESCRIPTION. The default is 'all'.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    lw : TYPE, optional
        DESCRIPTION. The default is 1.
    ticks_as_names : TYPE, optional
        DESCRIPTION. The default is True.
    color : TYPE, optional
        DESCRIPTION. The default is 'C0'.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """


    
    if columns_to_plot=='all':
        columns_to_plot = df.columns
        
    if hdf==True:
        units = [c[0]+"_"+c[1] for c in columns_to_plot] # for hDF 
    else:
        units = columns_to_plot

    stime_list=[]
    bbox_list=[]
    for i,col in enumerate(columns_to_plot):

        # for each neuron, make a list of stime's between t0 and t1
        s = df.loc[:,col][df.loc[:,col].between(t0, t1)]
        l = list(s.dropna())

        # create a collection of patches for each neuron (the burst boxes)
        rect = []
        num_bursts = len(res['BurstStart'][units[i]])
        # for j in range(num_bursts):
        #     if res['BurstStart'][units].iloc[j, i] < t1:
        #         rect.append(Rectangle([res['BurstStart'][units].iloc[j, i], i-height/2],
        #                                       res['BurstDuration'][units].iloc[j, i],
        #                                       height))
        for j in range(num_bursts):
            if res['BurstStart'][units].loc[j, units[i]] < t1:
                rect.append(Rectangle([res['BurstStart'][units].loc[j, units[i]], i-height/2],
                                              res['BurstDuration'][units].loc[j, units[i]],
                                              height))

        # make a list of lists for all neurons to be plotted
        stime_list.append(l)
        bbox_list.append(rect)

    if ax==None:
        # create a figure and axes
        fig, ax = plt.subplots()

    # plot the raster diagram
    ax.set_xlim(t0, t1)
    ax.eventplot(stime_list, linelengths=0.8, linewidth=lw, color=color)

    # plot the burst boxes
    for rect in bbox_list:
        ax.add_collection(PatchCollection(rect, color=[255/255,211/255,67/255]))

    # set the y-tick labels as the names of the neurons
    if ticks_as_names==True:
        ax.set_yticks(np.arange(len(stime_list)))
        ax.set_yticklabels(columns_to_plot)

    
    return ax



def examine_firing_rate_response(rec_name, table, applications=3, index_to_plot=['rand'], ax=None, rate_bin_width=60):
    """
    This function plots the firing rate of several units over the whole duration 
    of the recording **rec_name** and i plots the times of drug applications 
    specified in **table** 
    
    The purpose of the function is to allow the user to inspect the recording
    and determine the latency of a drug response.

    Parameters
    ----------
    rec_name : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.
    index_to_plot : TYPE, optional
        DESCRIPTION. The default is ['rand'].
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    rate_bin_width : TYPE, optional
        DESCRIPTION. The default is 60.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    # create firng rate data frame
    stimes = load_recording_stimes(rec_name)
    frates = firing_rates(stimes, bin_width=rate_bin_width)

    if ax==None:
        # make figure and axes 
        fig, ax = plt.subplots(figsize=(15,3))

    # generate a random neuron to plot and display the neuron name
    if index_to_plot[0]=='rand':
        i = np.random.randint(0, frates.shape[1]-1)
        index_to_plot[0] = i
        ax.set_title(frates.columns[i]+", i="+str(i), loc='left')

    # plot the selected neurons and display recording name
    frates.iloc[:, index_to_plot].plot(ax=ax)
    ax.text(0.95, 1.01, rec_name, transform=ax.transAxes, va='bottom', ha='right')

    # plot lines for every application
    for i in range(applications):
        ax.axvline(table.loc[rec_name, "DrugTime_"+str(i+1)], ls='--', color='grey')
        ax.text(table.loc[rec_name, "DrugTime_"+str(i+1)], ax.get_ylim()[1], table.loc[rec_name, 'Drug_'+str(i+1)])
        
    return ax



def plot_measure_across_conditions(measure, ax=None, horizontal_margin=0.25):
    """
    Function that takes the measuremnt comparison data frame **measure** and plots
    each column at a different x-point to compare values

    Parameters
    ----------
    measure : TYPE
        DESCRIPTION.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    horizontal_margin : TYPE, optional
        DESCRIPTION. The default is 0.25.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if ax==None:
        fig, ax = plt.subplots(figsize=(3,4))

    # plot connecting lines
    for i in range(measure.shape[0]):
        # control-response
        ax.plot([0,1], [measure['control'][i], measure['response'][i]], color='lightgrey')
        # response-wash
        if measure.shape[1]==3:
            ax.plot([1,2], [measure['response'][i], measure['wash'][i]], color='lightgrey')

    # plot points
    for i, condition in enumerate(measure.columns):
        ax.plot([i]*len(measure[condition]), measure[condition], 'o', color='k', alpha=0.5)

    # axes format 
    ax.set_xlim(-horizontal_margin, (measure.shape[1]-1)+horizontal_margin)
    ax.set_xticks([i for i in range(measure.shape[1])])
    ax.set_xticklabels(measure.columns.to_list())

    # calculate summary of numbers used
    N_slices = len(measure.index.get_level_values(0).unique())
    l = list(set( [string[0:6] for string in measure.index.get_level_values(0)] ))
    N_animals = len(l)

    # print numbers summary
    print('N_cells = ',len(measure))
    print('N_slices = ',N_slices)
    print('N_animals = ',N_animals)
    print(' ------------')
    
    return ax



def plot_tida_firing_rates(tida_hdf, table, bin_width=100, xlims=[0, 5000]):
    """
    Function that plots the firing rate time series for all identified TIDA cells
    with different recordings plotted on different axes.

    Parameters
    ----------
    tida_hdf : hDF
        hDF of TIDA neuron spike times.
    table : TYPE
        DESCRIPTION.
    bin_width : int, optional
        width of the smoothing window/histogram bin. The default is 100.
    xlims : TYPE, optional
        DESCRIPTION. The default is [0, 5000].

    Returns
    -------
    None.

    """
    
    # calculate firing rates
    frates = firing_rates(tida_hdf, bin_width=bin_width)

    # get a list of recordings that contain TIDA cells
    tida_recs = tida_hdf.columns.get_level_values(0).unique()

    # plotting details
    axes_rows = int(-(- len(tida_recs)/2 // 1))
    fig, Ax = plt.subplots(axes_rows, 2, figsize=(15, 2*axes_rows), tight_layout=True)
    Ax = Ax.flatten()

    # loop through recordings and plot the TIDA firing rates for all cells
    for i, rec in enumerate(tida_recs):
        ax=Ax[i]
        ax.plot(frates.loc[:, (rec, slice(None))])
        ax.set_title(rec, loc='right')
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(xlims[0], xlims[1], 100), minor=True)

        # plot the drug application times
        drug_cols = [col for col in table.columns if col.startswith('Drug')]
        for i in range(int(table[drug_cols].shape[1]/2)):
            if table.loc[rec, "DrugTime_"+str(i+1)] < ax.get_xlim()[1]: # this only plots times within xlims defined above
                ax.axvline(table.loc[rec, "DrugTime_"+str(i+1)], ls='--', color='grey')
                x = 0 if i % 2 == 0 else 0.3 # this makes the axes text easier to read
                ax.text(table.loc[rec, "DrugTime_"+str(i+1)], ax.get_ylim()[1] -x, table.loc[rec, 'Drug_'+str(i+1)])
        ax.grid(True, which='both')

    
def maximum_firing_rate_times(hdf, drug_application, tida_recs, table, 
                              response_window_end='next drug', window_length=2000,
                              bin_width=100):
    """
    Function that plots a histogram of the times of maxium change to a recordings
    firing rate during a drug application window.
    - All neurons are considered. 
    - Both positive and negative changes to the firing rate are considered.
    - Times between the drug application and the next drug application are considered.
    - If theres no next drug application, the window lasts **window_length** s
    
    The function of the script is to indentify the times that the slice responds
    most prominently to the application.

    Parameters
    ----------
    hdf : TYPE
        DESCRIPTION.
    drug_application : TYPE
        DESCRIPTION.
    tida_recs : TYPE
        DESCRIPTION.
    table : TYPE
        DESCRIPTION.
    window_length : TYPE, optional
        DESCRIPTION. The default is 2000.
    bin_length : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    peak_times : TYPE
        DESCRIPTION.

    """

    # calculate the firing rates for the spike trains
    frates = firing_rates(hdf, bin_width=bin_width)

    # plotting details 
    axes_rows = int(-(- len(tida_recs)/2 // 1))
    fig, Ax = plt.subplots(axes_rows, 2, figsize=(7, 1.5*axes_rows), tight_layout=True)
    Ax = Ax.flatten()

    peak_frate={};  peak_times={}
    for i, rec in enumerate(tida_recs):
        rec_frates = frates.loc[:, (rec, slice(None))] 
        rec_frates = abs( rec_frates - rec_frates[:600].mean() ) # absolute means that neagtive deflections are detected 

        # get the response window start from the table
        drug_time_col_idx = np.where( table.loc[rec]==drug_application)[0][0] + 1
        response_window_start = table.loc[rec, table.columns[drug_time_col_idx]]

        # determine the window end as either the next drug application, or 2000s~30m into the future
        if response_window_end == 'next drug':
            response_window_end   = table.loc[rec, table.columns[drug_time_col_idx+2]]
            if pd.isna(response_window_end): response_window_end = response_window_start + window_length
        elif response_window_end == None:
            response_window_end = response_window_start + window_length

        # determine the times of peak abs(Change in FR) and store in a dictionary of Series'
        bins = np.arange(response_window_start, response_window_end, bin_width)
        #bins = 30
        count, edges = np.histogram(rec_frates.loc[response_window_start:response_window_end].idxmax(), bins=bins)
        peak_frate[rec] = pd.Series(count, index=edges[:-1]).sort_values(ascending=False).iloc[:10]
        peak_times[rec] = peak_frate[rec].index.to_list()

        # plot the histograms for visaul inspection
        ax=Ax[i]
        rec_frates.loc[response_window_start:response_window_end].idxmax().plot.hist(bins=bins, ax=ax)
        ax.axvline(response_window_start, ls='--', color='grey')
        ax.set_title(rec, loc='right')

    # create a DF to veiw peak times
    peak_times = pd.DataFrame.from_dict(peak_times, orient='index').transpose() 
    
    return peak_times




def pos_neg_neut_piechart(total, pos, neg, ax=None, edgecolor='limegreen',
                          title='something', edgethickness=10, add_text=True,
                          alpha=0.6):
    
    if pos+neg > total:
        raise ValueError("The total is less than the sum of pos+neg")
    
    non = total - (pos+neg)
    sizes = [pos, neg, non]  

    # Create a figure if non is given
    if ax==None:
        fig, ax = plt.subplots(figsize=(6, 6))  # Set the figure size
    
    # plot the pi chart
    ax.pie(sizes, startangle=90, colors=['C1', 'C0', 'lightgrey'],
           wedgeprops={'alpha': alpha})  # Create the pie chart
    # plot the ring around the pie
    circle = plt.Circle((0, 0), 1, color='none', ec=edgecolor, lw=edgethickness)
    ax.add_artist(circle)
    
    if add_text == True:
        ax.set_title(title)  # Add a title
        text = 'pos = '+str(pos)+', neg = '+str(neg)
        ax.text( -1, -1.5, text)
        

    # Display the pie chart
    ax.axis('equal')
    
    return ax



def paired_raincloud(df, orient='h', ax=None, palette=['lightgrey', [223/255, 157/255, 158/255]], dot_colors=['grey', 'C3'],
                     draw_box=False):
    
    if dot_colors==None:
        dot_colors = palette
        
    if ax==None:
        fig, ax = plt.subplots()

    # Loop through the rows
    for j in range(df.shape[0]):
        x_values = df.iloc[j, :].values  # Get all values in the current row
        y_values = np.arange(len(x_values)) + np.random.normal(0, 0.05, len(x_values))  # Add jitter to y-coordinates



        if orient=='h':
            ax.plot(x_values[0], y_values[0], '.', color=dot_colors[0])  #palette Plot points
            ax.plot(x_values[1], y_values[1], '.', color=dot_colors[1])  # Plot points
            ax.plot(x_values, y_values, '-', color='gray', alpha=0.5, zorder=-1)  # Plot lines connecting points

        if orient=='v':
            ax.plot(x_values[0], y_values[0], '.', color=palette[0])  #palette Plot points
            ax.plot(x_values[1], y_values[1], '.', color=palette[1])  # Plot points
            ax.plot(x_values, y_values, '-', color='gray', alpha=0.5, zorder=-1)  # Plot lines connecting points

    # plot the distributions    
    ax = rm.RainCloud(data=df, bw=0.2, width_viol=1, orient=orient, ax=ax,
                      hatchs=[' ', ' ', ' '], jitter=0.1, alpha=1, draw_box=draw_box,
                      palette=palette, point_size=0)
    
    return ax



# %% SYNCHRONY

def normalised_correlation(x, y, mode='full'):
    """
    Function to calculate the cross-corrleation defined by
    
     cc_xy(t) = 1/(N-t) * Sum_{i=1}^{N-t}[ (x-x.mean)/x.var  * (y-y.mean)/y.var ]
     
     as detailed in Quiroga et al. 2002. Note that I use the STD instead of the VAR.
     
    """
    
    # de-mean and normalise from each signal 
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    
    # correlate the detrended signals
    cc = np.correlate(x, y, mode=mode)
    
    # normalise the correlation with a lag-dependent normalisation coeffcient
    N = len(x)
    norm_cof = np.concatenate([np.arange(1, N+1), np.flip(np.arange(1, N+1))[1:]])
    cc = cc / norm_cof
    
    # store as a series object
    lags = np.concatenate([-np.arange(N-1,0,-1), np.arange(0,N)])
    cc = pd.Series(cc, index=lags)
    
    return cc




def fourier_spectrum(s, sampling_interval='auto', plot=False, ax=None, color='C0'):
    """
    Plot the Fourier spectrum of a signal, s, sampled at regular intervals.
    s should be de-meaned.

    Parameters:
        s (array-like): The input signal.
        sampling_interval (float): The sampling interval (in seconds)
        between consecutive samples.

    Returns:
        f_spec (the Fourier spectrum as a pd.Series).
    """
    
    # calculate the sampling frequency automatically from the data frame
    if sampling_interval == 'auto':
        if type(s) == pd.Series:
            sampling_interval =  s.index[1] - s.index[0]
        else:
            raise ValueError('The sampling interval must be explicitly \
                             provided because the signal is not a Series object')

    # Compute the Fourier Transform
    fourier_spectrum = np.fft.fft(s-np.mean(s))
    
    # Compute the frequency range
    frequency_range = np.fft.fftfreq(len(s), d=sampling_interval)
    
    # Make a series
    fourier_spec = pd.Series(abs(fourier_spectrum), index=frequency_range).sort_index()
    
    # Calculate the dominant frequency
    dom_freq = fourier_spec.loc[0:].idxmax()
    
    # Plot the Fourier spectrum
    if plot==True:
        if ax==None:
            fig, ax = plt.subplots(figsize=(10, 3))

        ax.plot(fourier_spec.loc[0:], color=color)
        ax.plot(dom_freq, fourier_spec.loc[dom_freq], 'ro')
        ax.vlines(dom_freq, ax.get_ylim()[0], fourier_spec.loc[dom_freq], ls='--', color='r')
        ax.set_title('Fourier Spectrum')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        #plt.show()
        
        return fourier_spec, dom_freq, ax
    else:
        return fourier_spec, dom_freq
    
    
    

    
def butterworth_filter(s, cf_low=0.01, cf_high=1.5, sampling_interval='auto', order=4):
    
    # calculate the sampling frequency automatically from the signal if it is a Series
    if sampling_interval == 'auto':
        if type(s) == pd.Series:
            sampling_interval =  s.index[1] - s.index[0]
        else:
            raise ValueError('The sampling interval must be explicitly provided because the signal is not a Series object')
            
            
    # calculate the upper cut-off frequency
    _, dom_freq = fourier_spectrum(s, sampling_interval='auto', plot=False)
    
    # set the bandpass filter frequencies
    cf = [cf_low, cf_high * dom_freq]
        

    # # calculate the cutoff frequencies
    # if cf == '2dom_bandpass':
    #     # set the hugh-cutoff frequency as 2*dominant_freq, caclulated with an FFT
    #     _, dom_freq = fourier_spectrum(s, sampling_interval='auto', plot=False)
    #     cf = [0.01, 1.5 * dom_freq]
        
    # elif isinstance(cf, list):
    #     cf = cf
        
    # check that the cut-off frequencies are within the correct bounds
    if cf[1] > (0.5 / sampling_interval):
        cf[1] = 0.499
        print(f"Warning: An oscillation of {round(dom_freq,3)} Hz is \
              present and 2*{round(dom_freq,3)} > 1 / 2*sampling_interval. \
                  Upper limit of the filter has been set to 0.5 Hz. \
                      Decreasing sample_interval will ensure signal is not altered.")

    
    # set up the butterworth bandpass filter
    order = 4
    sos = signal.butter(order, cf, 'bandpass', fs=1/sampling_interval, output='sos')

    # filter the signal
    s = signal.sosfiltfilt(sos, s, padtype='even', axis=0)
    
    return s

    
    

def hilbert_sync(x, y, rolling_n=5, bw_filter=True, bw_cf_low=0.01, bw_cf_high=1.5,
                 sampling_interval='auto'):
    """
    Function to calculate the phase difference and synchrony between two signals 
    **x** and **y** using the definition of phase from the analytic signal 
    (calculated via a Hilbert transform).

    Parameters
    ----------
    x : array-like. Series is best.
        DESCRIPTION.
    y : array-like. Series is best.
        DESCRIPTION.
    rolling_n : int, optional
        Window size used to calculate the rolling sync metric. The default is 5.
    bw_filter : Bool, optional
        If True, the signals are bandpass filtered using a butterworth filter 
        of order=4. The default is True.
    bw_cf : str or list of [cf_low, cf_high], optional
        The cutoff frequencies for the bandpass filter. The default is 
        '2dom_bandpass', which uses a FFT to calculate the dominant frequency
        of the signal and uses the cutoff frequencies [0.01, 2*dom_freq]. The 
        lower cutoff eliminates slow noise with a period > 100s 

    Returns
    -------
    hil_sync : TYPE
        DESCRIPTION.
    rol_hil_sync : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.

    """
    
    # prepare the signals
    x = x - np.mean(x)
    y = y - np.mean(y)
        
    # filter the signals
    if bw_filter == True:
        x = butterworth_filter(x, cf_low=bw_cf_low, cf_high=bw_cf_high, sampling_interval=sampling_interval)
        y = butterworth_filter(y, cf_low=bw_cf_low, cf_high=bw_cf_high, sampling_interval=sampling_interval)

    # calculate the analytic signals
    as1 = hilbert(x)
    as2 = hilbert(y)

    # calculate the phase data
    phs1 = np.angle(as1)
    phs2 = np.angle(as2)

    # calculate the phase difference
    phi = np.unwrap(phs1) - np.unwrap(phs2) # phi > 0 ==> x1 is phase advanced

    # define the synchronization measure
    hil_sync = abs(np.mean(np.exp(1j*phi)))

    # define the time-dependent synchrony metric
    rol_hil_sync = abs(np.convolve(np.exp(1j*phi), np.ones(rolling_n)/rolling_n, mode='valid'))
    
    # return a Series object if the inputs were Series
    if type(x)==pd.Series:
        phi          = pd.Series(phi, index=x.index)
        rol_hil_sync = pd.Series(rol_hil_sync, index=np.convolve(x.index, np.ones(rolling_n)/rolling_n, mode='valid'))
    # ... else numpy arrays will be returned
    
    return hil_sync, rol_hil_sync, phi



def Hil_sync_matrix(df, hil_sync_threshold=0.65):
    """
    Function to calculate the pairwise synchrony between every column of the 
    Data Frame **df** using the Hilbert sync metric
    
    Parameters
    ----------
    df : Data Frame
        Columns are FIRING RATE histograms of neural activity.
    return_phi : TYPE, optional
        DESCRIPTION. The default is False.
    
    Returns
    -------
    TYPE
        DESCRIPTION.
    
    """
    
    # create an empty data frame for the sync metric matrix
    hsync_df = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    # create an empty data frame for the phase differnece plots
    phi_df = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    
    # create an empty dict for the phase difference trajectories
    phi_dict = {}
    
    
    # calculate pairwise Hilbert synchrony for every pair of neurons in DF
    for i, col_0 in enumerate(df.columns):
        phi_inner_dict={}
        for j, col_1 in enumerate(df.columns):
            # calculate the Hilbert synchrony and the phase differnece
            hil_sync, rol_hil_sync, phi = hilbert_sync(df[col_0], df[col_1], rolling_n=5, bw_filter=True)
            
            # store Hilbert sync metrix as a matrix
            hsync_df.loc[col_0, col_1] = hil_sync
            
            # store phase difference as a matrix, with NaN where no sync exists
            if hil_sync > hil_sync_threshold:
                phi_df.loc[col_0, col_1] = stats.circmean(phi, high=np.pi, low=-np.pi)
            else:
                phi_df.loc[col_0, col_1] = np.nan
                
            # store the full phase difference trajectories in a dictionary
            if j in np.arange(i):
                phi_inner_dict[col_1] = rh.mod2pi(phi)
        phi_dict[col_0] = phi_inner_dict
            
    
    return hsync_df, phi_df, phi_dict


def cc_sync_matrix(df):
    """
    Function to calculate the pairwise synchrony between every column of the 
    Data Frame **df** using the cross-corelation sync metric

    Parameters
    ----------
    df : Data Frame
        Columns are FIRING RATE histograms of neural activity.
    return_phi : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # create an empty data frame for the sync metric matrix
    cc_sync = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    # calculate the cross-correlation for every pair of neurons
    for col_0 in df.columns:
        for col_1 in df.columns:
            cc = normalised_correlation(df[col_0], df[col_1])

            # define the cross-correlation score as the correlation at lag=0
            cc_sync.loc[col_0, col_1] = cc.loc[0]
            
    return cc_sync


def intra_and_inter_hemisphere_pairwise_sync(sync_df, rhs_chns, lhs_chns):
    """
    Function to create two lists of pairwise synchrony values:
        - INTER-hemi sync, and 
        - INTRA-hemi sync,
    given a list of channels that belong to the R-arc and the L-arc

    Parameters
    ----------
    sync_df : TYPE
        DESCRIPTION.
    rhs_chns : TYPE
        DESCRIPTION.
    lhs_chns : TYPE
        DESCRIPTION.

    Returns
    -------
    intra_sync : TYPE
        DESCRIPTION.
    inter_sync : TYPE
        DESCRIPTION.

    """
    
    
    # define a sync sub-matrix for pairs within each hemisphere,
    # and pairs between each hemisphere (ven)
    rhs_sync = sync_df.loc[rhs_chns, rhs_chns]
    lhs_sync = sync_df.loc[lhs_chns, lhs_chns]
    ven_sync = sync_df.loc[rhs_chns, lhs_chns]

    # obtain the sync values below the leading diagonal so that each pair is 
    # represented only once
    rhs_sync_list = values_below_df_diagonal(rhs_sync.values)
    lhs_sync_list = values_below_df_diagonal(lhs_sync.values)
    ven_sync_list = ven_sync_list = ven_sync.values.flatten()

    # create a within- and between-hemisphere sync array
    intra_sync = rhs_sync_list + lhs_sync_list
    inter_sync = ven_sync_list
    
    return intra_sync, inter_sync


def sync_across_recordings(frates_df, recs, t0=0, t1=600):
    """
    Function that calculates a dictionary of pairwise synchrony matrices for
    all recordings in **recs** between **t0** and **t1**. 

    Parameters
    ----------
    frates_df : TYPE
        DESCRIPTION.
    recs : TYPE
        DESCRIPTION.
    t0 : TYPE, optional
        DESCRIPTION. The default is 0.
    t1 : TYPE, optional
        DESCRIPTION. The default is 600.

    Returns
    -------
    sync_df_dict : TYPE
        DESCRIPTION.
    phi_df_dict : TYPE
        DESCRIPTION.

    """
    
    sync_df_dict={}
    phi_df_dict={}
    for rec in recs:

        # calculate Hilbert synchrony metric
        hsync_df, phi_df, phi_dict = Hil_sync_matrix(frates_df.loc[t0:t1, rec])

        # store sync_df in a dictionary
        sync_df_dict[rec] = hsync_df
        phi_df_dict[rec]  = phi_df
    
    
    return sync_df_dict, phi_df_dict





def compare_inter_vs_intra_hemisphere_sync(sync_df_dict, phi_df_dict, sync_threshold=0.7):
    """
    Function that uses the pairwise sync matrix to generate a list of pairwise
    sync measures for:
        1. Pairs within the same hemisphere, and
        2. Pairs between the two hemispheres
    for each recording. The average phase difference, phi, is also calculated.
    
    *** NOTE *** --------------------------------------------------------------
    This function should be broken down into two functions:
        (A) Function to generate (within, between) lists for a single recording.
        (B) Function to itterativly apply (A) to a list of recordings
    ************ --------------------------------------------------------------

    Parameters
    ----------
    sync_df_dict : TYPE
        DESCRIPTION.
    phi_df_dict : TYPE
        DESCRIPTION.
    sync_threshold : TYPE, optional
        DESCRIPTION. The default is 0.7.

    Returns
    -------
    within_sync : TYPE
        DESCRIPTION.
    between_sync : TYPE
        DESCRIPTION.
    within_phi : TYPE
        DESCRIPTION.
    between_phi : TYPE
        DESCRIPTION.

    """
    
    # laod the table of summy info regarding the recordings of the experiment
    sum_table = load_summary_table(drop_trh=True)
    
    # lists to store all sync values
    within_sync={}; between_sync={}
    within_phi={};  between_phi={}

    betweenhemi_sync_recs = []

    # loop through all suitable recordings
    for rec in sync_df_dict.keys():

        hsync_df = sync_df_dict[rec]
        phi_df   = phi_df_dict[rec]

        # determine RHS from LHS channels
        ventricle_col = int(sum_table.loc[rec, 'Ventricle LHS'])
        rhs_chns = [chn for chn in hsync_df.columns.to_list() if float(electrode_number(chn)) <= ventricle_col * 10 + 6]
        lhs_chns = [chn for chn in hsync_df.columns.to_list() if float(electrode_number(chn)) >  ventricle_col * 10 + 6]

        # store intra- and inter-hemisphere sync values
        within_hemi_sync, between_hemi_sync = intra_and_inter_hemisphere_pairwise_sync(hsync_df, rhs_chns, lhs_chns)
        within_sync[rec]  = within_hemi_sync
        between_sync[rec] = between_hemi_sync

        # store intra- and inter-PD values
        within_hemi_phi, between_hemi_phi = intra_and_inter_hemisphere_pairwise_sync(phi_df, rhs_chns, lhs_chns)
        within_phi[rec]  = within_hemi_phi
        between_phi[rec] = between_hemi_phi

        # print a message if there is inter-hemispheric synchrony 
        if (between_hemi_sync > sync_threshold ).any() == True:
            print(rec + ' has inter-hemi sync')
            betweenhemi_sync_recs.append(rec)

    # convert global sync measurments to lists
    within_sync  = dict_of_lists_to_list(within_sync)
    between_sync = dict_of_lists_to_list(between_sync)

    # convert global phi measurments to lists
    within_phi  = dict_of_lists_to_list(within_phi)
    between_phi = dict_of_lists_to_list(between_phi)
    
    return within_sync, between_sync, within_phi, between_phi





# %% ARTIFICIAL DATA 

class ArtificialSpikeTrain:
    """
    Class to generate an artificial spike train.
    
    """
    def __init__(self, burst_period=10, duty_cycle=0.2, spike_frequency=100, period_var=1/10, tonic_frequency=1, phase_diff=0):
        self.burst_period    = burst_period  # The average duration of the burst in seconds
        self.duty_cycle      = duty_cycle    # fraction of the burst spent spiking
        self.spike_frequency = spike_frequency  # Burst frequency in Hz
        self.period_var      = period_var # variation in th period, as a fraction of the period
        self.phase_diff      = phase_diff # \in [0,1] where 1 corresponds to a phase difference of a full period 

    def generate_burst(self):
        """
        Generate the spike times within a single burst. The ISI's are choosen 
        from an exponential distribution with rate *spike_frequency*.
        """
        num_spikes_in_burst = int(self.spike_frequency * self.burst_period * self.duty_cycle)
        isi_within_burst = np.random.exponential(1/self.spike_frequency, num_spikes_in_burst)
        firing_times_within_burst = np.cumsum(isi_within_burst)
        return firing_times_within_burst

    def generate_burst_spike_train(self, duration, noise=0):
        """
        Generates a train of bursts for *duration* seconds.
        """
        burst_times = np.arange(0, duration, self.burst_period)
        firing_times = []
        for burst_time in burst_times:
            # generate a random latency for each burst
            latency = self.period_var*np.random.uniform(-self.burst_period, self.burst_period)
            # generate a burst starting at the burst time 
            burst = self.generate_burst() + burst_time + latency
            firing_times.extend(burst)
            
        # add exponential process noise
        if noise > 0:
            firing_times.extend( self.generate_exponential_spike_train(duration, tonic_frequency=noise) )
            
        # remove negative firing times
        firing_times = np.array(firing_times)
        firing_times = firing_times[firing_times>0]
        
        # add the phase difference
        firing_times = firing_times + self.burst_period*self.phase_diff
        
        return firing_times
    
    
    def generate_exponential_spike_train(self, duration, tonic_frequency=1):
        """
        Generates a spike train with ISI's drawn from an exponential distribution
        with rate *spike_frequency*
        """
        num_spikes = int(tonic_frequency * duration)
        isi = np.random.exponential(1/tonic_frequency, num_spikes)
        firing_times = np.cumsum(isi)
        return firing_times





    