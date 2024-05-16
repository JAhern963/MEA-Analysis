
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:38:07 2021

@author: kh19883

Module holding some data analysis functions.

"""

import pandas as pd
import numpy as np

import scipy
from scipy import signal
from scipy.signal import find_peaks, correlate
from scipy.fftpack import fft, ifft, fftfreq
from scipy import stats

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator


import sys
import os
sys.path.append('C:\\ProgramData\\Nex Technologies\\NeuroExplorer 5 x64')
import nex


#import seaborn as sns


# %% BASIC THINGS ------------------------------------------------------------

def pi_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def two_pi_wrap(x):
    return x % (2 * np.pi)


def mod2pi(x, break_series=False, threshold=1.57):
    
    # make mod2pi transformation
    xt = (x + np.pi) % (2 * np.pi) - np.pi
    
    # if true, replace value to the right of the discontinuity with nan
    if break_series==True:
        discont_idx = np.where(abs(np.diff(xt))>threshold)[0] + 1
        if len(discont_idx)>0:
            xt[discont_idx]=np.nan
    
    return xt


def thesis_fig(fig, name, chap, dpi=300):
    
    
    if chap==0:
        chapter = 'chapter00_Intro'
    elif chap==1:
        chapter = 'chapter01_CircDVC'
    elif chap==2:
        chapter = 'chapter02_TidaModel'
    elif chap==4:
        chapter = 'chapter04_CircTida'
    elif chap==5:
        chapter = 'chapter05_TidaNet'
    
    
    figs_dir = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\Thesis\\"+chapter+"\\figs"
    fig.savefig(figs_dir+"\\"+name, bbox_inches='tight', dpi=dpi)
    
    
    
def save_fig(fig, name, dpi=300):
    
    figs_dir = "C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics\\ARC-Bursting\\Writing\\MEA data manuscript\\figs"
    fig.savefig(figs_dir+"\\"+name, bbox_inches='tight', dpi=dpi)




def pi_axes(ax, axes='y', unit='rad', grid=True):
    
    
    if axes=='y':
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        if unit=='rad':
            ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        if unit=='hour':
            ax.set_yticklabels([-12, -6, 0, 6, 12])
        minor_locator = AutoMinorLocator(2)
        ax.yaxis.set_minor_locator(minor_locator)
        
    if axes=='x':
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        if unit=='rad':
            ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        if unit=='hour':
            ax.set_xticklabels([-12, -6, 0, 6, 12])
        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        
        
    if axes=='both':
        # x-axis
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        if unit=='rad':
            ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        if unit=='hour':
            ax.set_xticklabels([-12, -6, 0, 6, 12])
        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        
        # y-axis
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        if unit=='rad':
            ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        if unit=='hour':
            ax.set_yticklabels([-12, -6, 0, 6, 12])
        minor_locator = AutoMinorLocator(2)
        ax.yaxis.set_minor_locator(minor_locator)
    
    if grid==True:
        ax.grid(which='both', alpha=0.5)
        
    
    return ax





def convert_2_ZT(time, lights_on=815):
    """
    Function for converting 24-hour time into an organisms ZT time. NOTE that 
    the first decimal digit idicates the minuite measurement. E.G. ZT 8.3 means 
    8:30 in ZT time.
    
    Parameters
    ----------
    time : int or float
        The time, in 24-hour format (0-2400).
    lights_on : int or float, optional
        The time that lights are turned on. The default is 815.

    Returns
    -------
    ZT_time : int or float
        Zeitgeber Time.
    """

    # Convert to ZT
    ZT_time = (time - lights_on)/100
    
    # If ZT time is negative, convert into mod24 arithmatic
    if ZT_time < 0: ZT_time = ZT_time + 24
    
    return ZT_time


def rolling_complex(op, data, window=10):
    """
    Parameters
    ----------
    op : str
        The mathematical operation to apply to the rolling window data
    data : complex series or array like
        A series of complex numbers 
    window : int, optional
        The number of samples to include in a single rolling window.
        The default is 10.

    Returns
    -------
    rollingOp : Series
        A series of complex numbers output from the rolling window calculation.
        defined in OP.

    """

    # Two Series of real numbers for each complex component
    rc = pd.Series(np.real(data))
    ic = pd.Series(np.imag(data))

    # Create rolling objects to perform a calculation with
    rollingReal = rc.rolling(window, center=True)
    rollingImag = ic.rolling(window, center=True)

    # Carry out a mathematiocal operation defined by OP
    if op == 'mean':
        rollingRealOp = rollingReal.mean()
        rollingImagOp = rollingImag.mean()

    # Combine the two complex components into a single number
    rollingOp = rollingRealOp + 1j * rollingImagOp

    return rollingOp


def rolling_median_filt(x, window_size, threshold, double=True):
    '''


    Parameters
    ----------
    x : Series 
        Time-series to filter 
    window_size : int
        Number of sample sizes to average over.
    threshold : float
        Threshold for the difference between signal and median

    double : Bool, optional
        If double==True then a backward pass is also applied. The default is
        True.

    Returns
    -------
    med_filt : Series
        Median filtered time-series of x.

    '''

    # Function for a single pass median filter
    def rolling_median_filter_single_pass(x, window_size, threshold):
        # Calculate the rolling median and the differences
        median = x.rolling(window_size, center=False).median()
        diff = abs(x - median)

        # Filter the signal by replacing outliers with a suitable value
        med_filt = x.where(diff < threshold, median)

        return med_filt

    # Filter once with the function above
    med_filt = rolling_median_filter_single_pass(x, window_size, threshold)

    # Double filter the signal to minimalise phase distortion
    if double == True:
        med_filt_2 = rolling_median_filter_single_pass(
            med_filt.iloc[::-1], window_size, threshold)
        med_filt_2 = med_filt_2.iloc[::-1]
        med_filt = med_filt_2

    return med_filt




def print_n_numbers(pooled_summary):
    """
    Function that prints the number summary for the expeirments analized in the 
    pooled_summary file

    Parameters
    ----------
    pooled_summary : Data frame
        File containing analized data.

    Returns
    -------
    number_animals : TYPE
        DESCRIPTION.

    """
    
    # Function for findig number of animals (unique dates)
    # This assumes 1 animal per day of reccording
    def number_of_animals(pooled_summary):
        # list of experiments
        experiment_list = pooled_summary.groupby(level=0).mean().index
        # list of dates
        experiment_list = pd.Series([x[0:6] for x in experiment_list])
        # no. animals  =  no. of unique reccording dates
        number_animals = len(experiment_list.unique())
        
        return number_animals
    
    
    # print the numbers
    print('Number of cells: ',pooled_summary.shape[0])
    print('Number of slices: ',len(pooled_summary.groupby(level=0)))
    print('Number of animals: ', number_of_animals(pooled_summary))
    
    
    
    
    
def day_night_exp_list(cutoff_24hr=1501, tidas=True):    
    

    # load workbook
    wb = workbook_df(output_df=True, sync=False)

    # filter wb for experiments with detetcted TIDA cells
    if tidas==True:
        wb = wb[(wb["No. TIDAs"].notna()) & (wb["No. TIDAs"]!=0)]

    full_exp_list = wb.index.tolist()


    # make two lists for day and night recordings
    wb_day   = wb[wb["TOC"]<=cutoff_24hr] # daytime recordings 
    wb_night = wb[wb["TOC"]>cutoff_24hr] # nightime recordings
    if wb_night.shape[0]+wb_day.shape[0]!=wb.shape[0]:
        raise(ValueError("Day and Night workbook length does not equal full length"))

    day_exp_list   = wb_day.index.tolist()
    night_exp_list = wb_night.index.tolist()
    
    
    return full_exp_list, day_exp_list, night_exp_list





import shutil
def clear_directory(directory_path):
    try:
        # Remove the entire directory
        shutil.rmtree(directory_path)

        # Recreate an empty directory
        os.makedirs(directory_path)
        
        print(f"Contents of {directory_path} cleared successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")



# %% MOVING MEAN ADAPTION -----------------------------------------------------
# Following 2 functions are for smoothing data. Modified from a Python libary
#
def expandarr(x, k):
    # make it work for 2D or nD with axis
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd)*x[0], x, np.ones(kadd)*x[-1]]


def movmoment(x, k, windowsize=3, lag='centered'):
    '''non-central moment


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        k-th moving non-central moment, with same shape as x


    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.

    '''

    windsize = windowsize
    # if windsize is even should it raise ValueError
    if lag == 'lagged':
        # lead = -0 + windsize #windsize//2
        lead = -0  # + (windsize-1) + windsize//2
        sl = slice((windsize-1) or None, -2*(windsize-1) or None)
    elif lag == 'centered':
        lead = -windsize//2  # 0#-1 #+ #(windsize-1)
        sl = slice((windsize-1)+windsize//2 or None, -
                   (windsize-1)-windsize//2 or None)
    elif lag == 'leading':
        # lead = -windsize +1#+1 #+ (windsize-1)#//2 +1
        lead = -windsize + 2  # -windsize//2 +1
        sl = slice(2*(windsize-1)+1+lead or None, -
                   (2*(windsize-1)+lead)+1 or None)
    else:
        raise ValueError

    avgkern = (np.ones(windowsize)/float(windowsize))
    xext = expandarr(x, windsize-1)
    # Note: expandarr increases the array size by 2*(windsize-1)

    #sl = slice(2*(windsize-1)+1+lead or None, -(2*(windsize-1)+lead)+1 or None)
    # print(sl)

    if xext.ndim == 1:
        return np.correlate(xext**k, avgkern, 'full')[sl]
        # return np.correlate(xext**k, avgkern, 'same')[windsize-lead:-(windsize+lead)]
    else:
        print(xext.shape)
        print(avgkern[:, None].shape)

        # try first with 2d along columns, possibly ndim with axis
        return signal.correlate(xext**k, avgkern[:, None], 'full')[sl, :]


# %% EXCEL FUNCTIONS ----------------------------------------------------------


def workbook_df(output_df=True, sync=False):
    """
    Function to update and/or retrieve the Pandas data frame "arc_experiments".
    Syncing/updating loads to the current spreadsheet and saves as a DF
    The output loads the most reccently saved DF/pickle version of the summary 
    sheet and returns that.

    """

    # Update the current DF with most recent speadsheet
    if sync == True:
        # Load the Summary spreadsheet into a DF
        workbook = pd.read_excel(r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"
                                 "\PhD Neural Dynamics\ARC-Bursting\MEA Data\ARC Experiments.xlsx",
                                 sheet_name='Summary',
                                 header=4,
                                 index_col='Experiment',
                                 usecols="B:N,S")

        # Save the DF as a pickle object which can be called later
        pd.to_pickle(workbook, r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"
                     "\PhD Neural Dynamics\ARC-Bursting\MEA Data\\arc_experiments.pkl")

    if output_df == True:

        # Load most reccent pickle version of summary sheet
        workbook = pd.read_pickle(r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"
                                  "\PhD Neural Dynamics\ARC-Bursting\MEA Data\\arc_experiments.pkl")

        return workbook


def delete_summary_row(experiment):
    """
    Function to delete the row in the rhythmicity summary DF for the given 
    experiment.

    """
    # Open summary file
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Rhythmicity.pkl"
    summary = pd.read_pickle(path)

    # Delete the row specified in the arguments
    summary = summary.drop(labels=experiment)

    # Save the summary DF
    pd.to_pickle(summary, path)


# %% LOAD SUMMARIES

def load_rhythmicity_summary():
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Rhythmicity.pkl"
    summary = pd.read_pickle(path)
    return summary



def load_GRP200nM_summary():
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\GRP200nM.pkl"
    summary = pd.read_pickle(path)
    return summary



def load_GRP200nM_NR_fr_summary(make_new=False):
    """
    The make_new argument can be used to load a fresh file, which can be useful
    for reseting analysis when rows need to be removed.

    """
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\GRP200nM_NR_fr_response.pkl"
    summary = pd.read_pickle(path)
   
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path



def load_GRP200nM_RI_summary(make_new=False):
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\GRP200nM_RI_response.pkl"
    summary = pd.read_pickle(path)
    
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path




def load_GRP200nM_RIntr_summary():
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\GRP200nM_RIntr_response.pkl"
    summary = pd.read_pickle(path)
    return summary, path


def load_NMB300nM_NR_fr_summary(make_new=False):
    """
    The make_new argument can be used to load a fresh file, which can be useful
    for reseting analysis when rows need to be removed.

    """
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\\NMB300nM_NR_fr_response.pkl"
    summary = pd.read_pickle(path)
   
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path


def load_NMB300nM_RI_summary(make_new=False):
    """
    The make_new argument can be used to load a fresh file, which can be useful
    for reseting analysis when rows need to be removed.

    """
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\\NMB300nM_RI_response.pkl"
    summary = pd.read_pickle(path)
   
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path


def load_NMC200nM_NR_fr_summary(make_new=False):
    """
    The make_new argument can be used to load a fresh file, which can be useful
    for reseting analysis when rows need to be removed.

    """
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\\NMC200nM_NR_fr_response.pkl"
    summary = pd.read_pickle(path)
   
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path


def load_NMC200nM_RI_summary(make_new=False):
    """
    The make_new argument can be used to load a fresh file, which can be useful
    for reseting analysis when rows need to be removed.

    """
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\\NMC200nM_RI_response.pkl"
    summary = pd.read_pickle(path)
   
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path



def load_VIP600nM_Per_summary(make_new=False):
    """
    The make_new argument can be used to load a fresh file, which can be useful
    for reseting analysis when rows need to be removed.

    """
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\Summary Data\Drug Response\\VIP600nM_Per_response.pkl"
    summary = pd.read_pickle(path)
   
    # make an empty file for a fresh analysis, using old file as a template
    if make_new==True:
        summary = make_new_pooled_file(path, summary, output=True)
        
    return summary, path



def make_new_pooled_file(path, column_template_df, output=False):
    
    # make a new data frame for pooled files
    pooled_file = pd.DataFrame(index=pd.MultiIndex.from_product([[], []]), columns=column_template_df.columns)

    # sve file as a pickle object at the path loaction
    pooled_file.to_pickle(path)
    
    if output==True:
        return pooled_file




def addrow_to_pooled_df(experiment, row, pooled_summary, path):
    """
    Function that adds a data frame (the "row") to another hierachical data
    frame. The hierachical DF will represent a pooled analysis, whilst each DF
    "row" will represent the analysis of a single slice/experient.

    Parameters
    ----------
    experiment : str
        Experiment to be added.
    row : Data frame
        DF of the analysis for the current experiment.
    pooled_summary : Data frame
        Hierarchical data frame of pooled analysis.
    path : str
        Path/location of the pooled_summary file.

    Returns
    -------
    None.

    """
    # MAKE a hiera data frame template - should not be needed
    ###grp_sum = pd.DataFrame(index=pd.MultiIndex.from_product([['yymmdd_sx'], ['dummy']]), columns=row.columns)
    
    # make an empty hiera data frame for a single expeirment, based upon the sigle experiemnt df
    empty = pd.DataFrame(index=pd.MultiIndex.from_product([[experiment], row.index]), columns=row.columns)
    
    # Put the slice response DF into the hiera format
    new_hiera_row = row.reindex(empty.index, level=1)
    
    # drop the current experiment, if present, from the population file to clear the last analysis
    if ((experiment, ) in pooled_summary .index)==True:
        pooled_summary = pooled_summary .drop(experiment)
        
    # add the current analysis to the pooled summary file
    pooled_summary = pd.concat([pooled_summary, new_hiera_row]).sort_index()

    # save the appended data frame
    pd.to_pickle(pooled_summary, path)
    
    
    
    


def load_firing_rates(experiment, rhythmic_units=True):

    # Import firing rate data from pickle object
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\\"
    path_1 = path+'TTX Filtered Units\\'+experiment+'.pkl'

    # Load the firing rate DF
    df = pd.read_pickle(path_1)

    # Return firing rate data frame and Index obj of rhythmic units
    if rhythmic_units == True:
        tida_units, ntr_units = load_rhythmic_units(experiment, path)
        return df, tida_units, ntr_units

    # Or just return the firing rate data frame
    else:
        return df



def load_spike_times(experiment, rhythmic_units=True):

    # Import spike times (SP)
    path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\\"
    path1 = path+'\\Spike Times\\'+experiment+'.pkl'
    st = pd.read_pickle(path1)

    # # Find the TTX filterd units using the TTX filtered firing rates
    # fr_path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
    #     "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
    #     "\TDistEM_DOM10_I5\Pandas Objects\\"
    # fr_path = fr_path+'TTX Filtered Units\\'+experiment+'.pkl'
    # fr = pd.read_pickle(fr_path)
    fr = load_firing_rates(experiment, rhythmic_units=False)

    # TTX filtered spike times
    st = st[fr.columns]
    
    # Return firing rate data frame and Index obj of rhythmic units
    if rhythmic_units == True:
        tida_units, ntr_units = load_rhythmic_units(experiment, path)
        
        return st, tida_units, ntr_units
    
    
    

    return st


def load_rhythmic_units(experiment, path):

    # Identify TIDA and NTR units, if they exsist
    rhythmic_summary = load_rhythmicity_summary()

    if rhythmic_summary.loc[experiment, "TIDA-like"] > 0:
        path_2 = path+'TIDA Units\\'+experiment+'.pkl'
        tida_units = pd.read_pickle(path_2).columns
    else:
        tida_units = []

    if rhythmic_summary.loc[experiment, "NTR"] > 0:
        path_3 = path+'NTR Units\\'+experiment+'.pkl'
        ntr_units = pd.read_pickle(path_3).columns
    else:
        ntr_units = []

    return tida_units, ntr_units





    
    
    
    
    
def load_pooled_tida_spiketimes(experiments='all', save_as_csv=False,
                                filename=None, return_path=False):
    

    # define the experiments to include in the TIDA array
    full_list, day_list, night_list = day_night_exp_list()

    if experiments=='all':
        experiment_list = full_list
    elif experiments=='day':
        experiment_list = day_list
    elif experiments=='night':
         experiment_list = night_list



    df = pd.DataFrame(columns=pd.MultiIndex.from_product([[], []]), index=[], dtype=float)

    # Loop through all experiments in the list
    for experiment in experiment_list:    

        # Import spike time data from pickle object 
        st, tida_units, ntr_units = load_spike_times(experiment, rhythmic_units=True)
        ts = st[tida_units]

        # save as a data frame and merge into a pooled data frame, df
        df_col = pd.DataFrame(ts.to_numpy(), index=ts.index, columns=pd.MultiIndex.from_product([[experiment], ts.columns]))
        df = df.join(df_col, how='outer')

    if save_as_csv==True:
        if filename==None:
            raise ValueError('Specify a file name to save the TIDA spike time CSV as.')
        # Make a NEX readable array of TIDA time stamps
        # merge upper col names into the lower col names
        new_cols=[]
        for i in range(df.shape[1]):
            exp  = df.columns[i][0]
            unit = df.columns[i][1]
            new_cols.append(exp+'_'+unit)
        nex_df = pd.DataFrame(df.to_numpy(), index=df.index, columns=new_cols)

        # save the data frame as a csv
        path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes\TDistEM_DOM10_I5\Text Files"
        path = path+'\\'+filename
        nex_df.to_csv(path, index=False)
        
    if return_path==True:
        return df, path 
    
    else:
        return df





# %% DATA VISUALISATION -------------------------------------------------------

def plot_experiment_firing_rate(df, channel, rows, smooth=5):
    """
    Function to plot the smoothed (5s default) firing rate time series of a
    single channel.

    Inputs:
        df      --- Data frame of 1s binned firing rates for an experiment
        channel --- String form of the channel, e.g. 'ch_XYa'
        rows    --- Number of plots for the data
        smooth  --- Window size (in samples) over which to smooth the firing 
                    rate.
    Outputs:
        fig     --- Figure object of the experiment
    """

    # Round time axis to nearest multiple of rows
    new_t_len = rows * round(len(df)/rows)
    # Calculate window of data to show in each plot
    plot_window = new_t_len / rows
    # Calculate start time
    t_start = df.index[0]

    # Calculatemax firing rate for ylim
    max_firing = df[channel].max()

    # Make figure object
    fig = plt.figure(figsize=(30, 2*rows))
    Ax = []
    for i in np.arange(rows):
        ax = fig.add_subplot(rows, 1, i+1)
        ax.set_ylim([0, max_firing+1])
        df[channel][t_start+i*plot_window:t_start+plot_window *
                    (i+1)].rolling(smooth, center=True).mean().plot(ax=ax)
        Ax.append(ax)
    ax.set_xlabel('Time [s]')

    return fig, Ax


def rasterplot(spike_times, t_span,
               units='all', ax=None, t0_threshold=100, linelengths=0.5):
    """

    Function to plot a raster diagram for the UNITS in the SPIKE_TRAIN data
    frame.

    """
    # set times to calculate raster for
    t0 = t_span[0]
    t1 = t_span[1]

    # set units to 'all' of the avalilable units
    if units == 'all':
        units = spike_times.columns
        linelengths = 0.01

    # Find index of each row closest to t=start_time within a tolerance
    # Loop ensures a tolerance such that all indices are found
    if t0 >= t0_threshold:
        num_zeros = 1
        tolerance = 1
        while num_zeros > 0:
            idx0 = spike_times[units].sub(t0).abs().lt(tolerance).idxmax()
            tolerance = tolerance + 1
            num_zeros = (idx0 == 0).sum()
    else:
        # Start from t=0
        idx0 = pd.Series(np.zeros(spike_times[units].shape[1], dtype=np.int8),
                         index=spike_times[units].columns)
        print("Warning: Choice of t0 may be too low. t0 has automatically been set to 0. \n\
        Change the t0_threshold and observe the simulation time if the orginal t0 is desired.")

    # Find index of each row closest to t=end_time within a tolerance
    # Loop ensures a tolerance such that all indices are found
    num_zeros = 1
    tolerance = 1
    while num_zeros > 0:
        idx1 = spike_times[units].sub(t1).abs().lt(tolerance).idxmax()
        tolerance = tolerance + 1
        num_zeros = (idx1 == 0).sum()

    # Make list of series
    raster_times = []
    for chn in units:
        raster_times.append(spike_times[chn][idx0[chn]:idx1[chn]])

    # Make an axis object if none is provided
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Format the axes
        ax.set_xlabel('time [s]', fontsize=15)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['top'].set_visible(False)

    ax.tick_params(labelsize=12, left=False, labelleft=False)
    # Plot the raster
    raster = ax.eventplot(raster_times, linelengths=linelengths)
    plt.show()

    return raster


def raster_firing_rate(experiment, unit, tspan, ax=None, spike_height=1.3,
                       smooth_para=3, linelengths=1, color='C0', lw=3,
                       fr_ylim=None, alpha=0.7):
    """
    Function to display the raster plot and the 1s binned firing rate histogram of 
    the UNIT in EXPERIMENT.

    Input:
        experiment --- string in the form 'yymmdd_sX'
        unit --- string in the form 'ch_XYi'
        tspann --- list in the form [t_start, t_end]
        fig --- figure to plot in. Default is provided if fig=None
        overlay --- int. The line_offset paramter to decide raster-firing rate
                        separation.
        smooth_para --- int; number of samples to smooth the firing rate plot by
        linelengths --- ax.eventplot() argument for length of raster lines

    Output:
        A figure/plot

    """
    global fr_snippet

    # Import spike times (SP)
    dir_path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\Spike Times"
    exp_path = dir_path+'\\'+experiment+'.pkl'
    st = pd.read_pickle(exp_path)

    # Import firing rates (FR)
    dir_path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
        "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
        "\TDistEM_DOM10_I5\Pandas Objects\TTX Filtered Units"
    exp_path = dir_path+'\\'+experiment+'.pkl'
    fr = pd.read_pickle(exp_path)

    # Import experiment summary data
    #wb = workbook_df(output_df=True)

    # Times to plot over
    t_start = tspan[0]
    t_end = tspan[1]
    fr_snippet = fr[unit][t_start:t_end].rolling(
        smooth_para, center=True).mean()
    st_snippet = st[unit][st[unit].between(t_start, t_end)]

    # Format where to plot things
    # if overlay == True:
    #     raster_offset = 0

    # if overlay == False:
    #     raster_offset = -2

    # Default figure
    if ax == None:
        fig = plt.figure(figsize=(20, 3))
        ax = fig.add_subplot(111)

    # plot firing rate before axis formating to aid the formating
    ax.plot(fr_snippet, lw=lw, color=color, alpha=alpha)

    # format the axes
    ax = format_axes(ax)
    ax.set_yticks(ax.get_yticks()[1:-1])
    ax.spines.left.set_bounds((ax.get_yticks()[0], ax.get_ylim()[1]))
    ax.set_xlabel('Time [s]', fontsize=15)
    ax.set_ylabel('Firing rate [Hz]', fontsize=15, x=-0.05, y=0.4)
    
    
    # plot the spike train
    overlay=spike_height*ax.get_ylim()[1]
    ax.eventplot(st_snippet.dropna(), color=color, lineoffsets=overlay, linelengths=linelengths)
    
    # Plot where a drug is applied, if applicable
    # Load the workbook to extract drug application time and lenght
    wb = workbook_df(output_df=True)
    application_time = 30  # s

    # Drug time could be NaN. Only plot drug application when !=NaN
    if pd.notna(wb['Drug 1'][experiment]) == True:

        # This loop only plots drug application if it is within tspan
        if (wb['Drug 1 time'][experiment] > tspan[0]-application_time) and (
                wb['Drug 1 time'][experiment] < tspan[1]):

            # Plot the application time at the max firing rate
            line_pos = 1.25*fr_snippet.max()
            line_pos = ax.get_ylim()[1]
            ax.hlines(line_pos, wb['Drug 1 time'][experiment],
                      wb['Drug 1 time'][experiment]+application_time, lw=3, color='r')
            ax.text(wb['Drug 1 time'][experiment], line_pos+0.2,
                    wb['Drug 1'][experiment], fontsize=13)


    return ax 


def nonzero_pie(numbers, labels, ax=None, col_list=None, explode=0.1, output=False,
                fontsize=14, alpha=None):
    """
    Simple function for plotting a pie chart which excludes any 0% wedges. All
    paramters are the same as for the matplotlib.pie() function.

    """
    
    if ax==None:
        fig, ax = plt.subplots()
        
    # Makes labels with numerical summary
    def add_nums_to_labs(labels, numbers):
        new_labs=[]
        for i,lab in enumerate(labels):
            percent = 100 * numbers[i]/sum(numbers)
            p_str = '{p:.1f}% '.format(p=percent)
            new_labs.append(lab+' \n '+p_str+'('+str(numbers[i])+')')
        return new_labs
    labels = add_nums_to_labs(labels, numbers)
    
    
    # Only consider the NON-ZERO elements of the NUMBERS list
    labs = []
    nums = []
    cols = []
    for i in range(len(numbers)):
        if numbers[i] > 0:
            labs.append(labels[i])
            nums.append(numbers[i])
            if col_list != None:
                cols.append(col_list[i])
            else:
                cols = None
    total = sum(nums)
    
    # Explode paramter, either equal or unequal
    if type(explode) is list:
        ex = explode
    else:
        ex = [explode]*len(labs)
        

    # Pie chart text function
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)
        return my_autopct


    # Alpha for pie chart
    if alpha!=None:
        # convert colors to rgba and adjust alpha
        col_list_a=[]
        for col in cols:
            col_rgba = colors.to_rgba(col)
            col_rgba = list(col_rgba)
            col_rgba[-1]=alpha
            col_rgba = tuple(col_rgba)
            col_list_a.append(col_rgba)
        cols = col_list_a


    # Pie chart plotting
    ax.margins(x=0.1)
    ax.pie(x=nums,
           labels=labs,
           textprops={'fontsize': fontsize},
           colors=cols,
           wedgeprops={'linewidth': 1},
           #autopct=make_autopct(nums),
           shadow=False,
           startangle=0,
           explode=ex
           )
    ax.axis('equal')
    ax.text(0.5, -1.2, "n = "+str(int(total))+" units", fontsize=fontsize-2)

    # Maybe add some printout if the zero values?

    if output == True:
        return fig, ax, nums, labs
    
    return fig, ax
    
    
    
    
def format_axes(ax, tick_label_size=13):
    """
    Function to format axes in a nice way 

    Parameters
    ----------
    ax : Axes object
  
    Returns
    -------
    ax : Axes
        The altered axes object.
    """
    
    ax.xaxis.set_tick_params(which='major', direction='out',
                             top='on', labelsize=tick_label_size)
    ax.yaxis.set_tick_params(which='major', direction='out',
                             top='on', labelsize=tick_label_size)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    
    return ax




def compare_distributions(a, b, c=None, transform='log', p_th=0.05):
    """
    Function to visually and statistically compare up to 3 different, equal length
    1D distributions. 

    Parameters
    ----------
    a : 1D Series
        A distribution to compare.
    b : 1D Series
        A distribution to compare.
    c : 1D Series, optional
        A distribution to compare. The default is None.
    transform : str, optional
        Type of transformation to apply to the distributions. The default is 'log'.
    p_th : float, optional
        Threshold for a statistically significant p value. The default is 0.05.

    Returns
    -------
    AB_sig : Bool.
    
    BC_sig : Bool

    AC_sig : Bool.

    """
    
    
    # no transformation
    if transform==None:
        at=a; bt=b
        if c is not None: ct=c
        
    # power-law transformation of distributions
    elif transform=='log':
        at = np.log(a)
        bt = np.log(b)
        if c is not None: ct = np.log(c)
        
    
        
    # plot the transformed distributions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(at, color='C0', alpha=0.5)
    ax.hist(bt, color='C1', alpha=0.5)
    if c is not None: ax.hist(ct, color='C2', histtype=u'step', lw=2)
    plt.show()
    
    # test for normality 
    print('a: Before drug firing rate normality tests:')
    normal_test(at)
    print('\n')
    print('b: After drug firing rate normality tests:')
    normal_test(bt)
    if c is not None:
        print('\n')
        print('c: Wash out firing rate normality tests:')
        normal_test(ct)
    
    # t-test for unequal means: A and B
    s, p = stats.ttest_rel(at, bt)
    print('\n')
    if p<p_th: print("paired t-test: A and B are Significantly different, p={:g}".format(p))
    else: print("paired t-test:  A and B not different, p={:g}".format(p))
    if p<p_th: AB_sig=True #ax = rh.plot_sig_line(ax, [0,1], sig_bar_height)
    else:      AB_sig=False

    if c is not None:
        # t-test for unequal means: B and C
        print('dont go here')
        print('\n')
        if p<p_th: print("paired t-test: B and C are Significantly different, p={:g}".format(p))
        else: print("paired t-test: B and C not different, p={:g}".format(p))
        if p<p_th: BC_sig=True #ax = rh.plot_sig_line(ax, [0,1], sig_bar_height)
        else:      BC_sig=False
    
        # t-test for unequal means: B and C
        s, p = stats.ttest_rel(at,ct)
        print('\n')
        if p<p_th: print("paired t-test: A and C are Significantly different, p={:g}".format(p))
        else: print("paired t-test: A and C not different, p={:g}".format(p))
        if p<p_th: AC_sig=True #ax = rh.plot_sig_line(ax, [0,1], sig_bar_height)
        else:      AC_sig=False
    else:
        BC_sig=None; AC_sig=None
    
    return AB_sig, BC_sig, AC_sig


    

def significance_plot(a, b, c=None, mc='green', malpha=0.2, ax=None, ms=12):
    """
    Function to plot a standard 'before after' comparison plot

    Parameters
    ----------
    a : Array_like
        First array of data points.
    b : Array_like
        Second array of data points.
    mc : srt, optional
        Color of dots. The default is 'green'.
    malpha : float, optional
        Alpha of dots. The default is 0.2.
    ax : Axes, optional
        The default is None.
    ms : float, optional
        Size of dots. The default is 12.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """

    if ax == None:
        fig = plt.figure(figsize=(4,5))
        ax = fig.add_subplot(111)

    # convert series to lists
    a = a.tolist()
    b = b.tolist()
    if c is not None: c = c.tolist()

    # plot lines
    for i in range(len(a)):
        ax.plot([0,1], [a[i], b[i]] , '-', color='lightgrey', alpha=0.3)
        if c is not None: ax.plot([1,2], [b[i], c[i]] , '-', color='lightgrey', alpha=0.3)

    # plot markers
    ax.plot([0]*len(a), a, 'o', alpha=malpha, ms=ms, color=mc)
    ax.plot([1]*len(b), b, 'o', alpha=malpha, ms=ms, color=mc)
    if c is not None: ax.plot([2]*len(c), c, 'o', alpha=malpha, ms=ms, color=mc)
    
    # set up axes
    ax = format_axes(ax)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticks(ax.get_yticks()[1:-1])
    ax.spines.left.set_bounds((ax.get_yticks()[0], ax.get_ylim()[1]))
    #ax.set_yticks(ax.get_yticks()[1:-1])
    
    x_tick_max = 1
    if c is not None: x_tick_max = 2
    ax.set_xticks(np.arange(0,x_tick_max+1))
    ax.set_xlim([-0.25, x_tick_max+0.25])
    
    return fig, ax





def plot_sig_line(ax, x, y, s='*'):
    """
    Function to plot the bar between data arrays to indicate a significant 
    differnece between the means of the distributions

    Parameters
    ----------
    ax : Axes
        Axes to plot on.
    x : Array_like
        List of start and end positions for the line.
    y : Float
        Height of the line.
    s : Str
        String to have above bar. Default is '*'

    Returns
    -------
    None.

    """
    
    # Fit the line between the data points
    x0 = x[0]+0.05
    x1 = x[1]-0.05
    
    # horizonal line 
    ax.hlines(y, x0, x1, color='k', lw=2)

    # vertical nubs
    dy = 0.05*ax.get_ylim()[1]
    ax.vlines(x0, y-dy, y, color='k', lw=2)
    ax.vlines(x1, y-dy, y, color='k', lw=2)

    # asterix 
    ax.text((x1+x0)/2, y , s, fontsize=25, ha='center')
    #plt.show()
    
    return ax
    
    




# %% RHYTHMICITY ANALYSIS -----------------------------------------------------

def autocorr_RI(ac, lags):
    """
    Function that calculates the Rhythmicity Index (RI) as the height
    of the second maximum of the autocorrelation function (AC) and 
    the respective Lag (RII) which serves as a period estimate. 
    Input:
       - AC : AutoCorrelation function, array
    Outputs:
       - RII : Rhytmicity Index Index, estimate of the period
       - RI : Rhythmicity Index, height of 2nd max peak of AC
    """

    pks = find_peaks(ac, height=0)[0]  # index of all peaks
    ri = abs(np.sort(ac[pks])[-2])     # value of 2nd largest peak (RI)
    # time/lag of RI (period estimation) #something iffy about this?
    rii = abs(lags[ac == ri][0])
    return rii, ri


def autocorr(df):
    # Function to calculate autocorrealtion function and rhythmicity index
    # for each unit. AUTOCORR_RI is called to calculate Rhythmicity Index and
    # Period estimate.
    # Input:
    #    - DF : DataFrame of array Firing Rates
    # Outputs:
    #    - ACOR : DataFrame of autocorrelation functions for each unit
    #    - RI : DataFrame of Rhythmicity Indices and Period estimates

    N = df.shape[0]
    lags = np.arange(-N+1, N)
    acor = pd.DataFrame(columns=df.columns, index=lags)
    ri_acor = pd.DataFrame(index=df.columns, columns=['Freq. est', 'RI'])

    for i in range(df.shape[1]):
        ac = np.zeros((2*N-1, 1))
        x = df.iloc[:, i] - df.iloc[:, i].mean()
        ac = (1/N) * scipy.signal.correlate(x, x, 'full')

        # Maually set zero-flat acor RI's to 0
        if ac.any() == False:
            rii = np.nan
            ri = 0
        else:
            ac /= max(ac)
            rii, ri = autocorr_RI(ac, lags)
        ri_acor.iloc[i, 0] = 1/rii
        ri_acor.iloc[i, 1] = ri
        acor.iloc[:, i] = ac
    return acor, ri_acor


def Fourier_Fisher_RI(df, smooth_window=1, rolling=False):
    """
    Function to calculate Fourier periodogram and rhythmicity index (i.e 
    Fisher's g-statistic) for each unit across the length of df.
    Input:
       - DF : DataFrame of array Firing Rates

    Outputs:
       - RI_FSPEC : DataFrame of Rhythmicity Indices and frequency estimates
       - AMP_SPEC : DataFrame of power spectra
       - PHS_SPEC : DataFrame of phase spectra
    """

    # Frequency construction
    T = df.shape[0]
    N = df.shape[0]
    dt = T/N
    scale = 2*dt**2/T
    #f = scipy.fftpack.fftfreq(N, dt)
    f = fftfreq(N, dt)
    #####f = scipy.fftpack.fftshift(f)

    # DataFrame initialization
    # could make faster by removing?
    amp_spec = pd.DataFrame(columns=df.columns, index=f)
    phs_spec = pd.DataFrame(columns=df.columns, index=f)
    ri_fspec = pd.DataFrame(index=df.columns, columns=[
                            'RI', 'Dom. Freq.', 'Phase'])

    # Loop through units
    for i in range(df.shape[1]):

        # Smooth the firing rate data if required, default does not smooth
        smooth_unit = df.iloc[:, i].rolling(
            smooth_window, center=True).mean().fillna(value=0)

        # Fourier transform
        xf = scipy.fftpack.fft(np.array(smooth_unit - smooth_unit.mean()))
        # Fourier Spectra DataFrame
        Sxx = np.real(scale * (xf * xf.conj()))
        amp_spec.iloc[:, i] = Sxx
        # Phase of each freqnecy component
        phs_spec.iloc[:, i] = np.angle(xf)

        # Rhytmicity index into DataFrame (axis 0)
        # Rhythmicity index (Fisher's g-statistic)
        ri_fspec.iloc[i, 0] = max(Sxx) / sum(Sxx)

        # Dominant frequency into DataFrame (axis=1)
        if max(Sxx) == 0:
            # Empty unit, set freq to np.nan
            ri_fspec.iloc[i, 1] = np.nan
        else:
            ri_fspec.iloc[i, 1] = abs(
                f[Sxx == max(Sxx)][0])  # Dominant frequency

        # Set long-time trends as non-rhythmic
        ri_fspec.iloc[:, 0][ri_fspec.iloc[:, 1] == f[0]] = 0  # f[0]=0
        ri_fspec.iloc[:, 0][ri_fspec.iloc[:, 1] == f[1]] = 0  # f[1]=f_res
        ri_fspec.iloc[:, 0][ri_fspec.iloc[:, 1] == f[2]] = 0  # f[2]=2*f_res

        # Phase of dominant frequency into DataFrame, max so ensure positive phase
        ri_fspec.iloc[i, 2] = np.angle(xf)[Sxx == max(Sxx)].max()

    # If rolling==True then we do not require the spectra for every time window
    # so only return thr RI indicies and frequencies

    if rolling == False:
        return ri_fspec, amp_spec, phs_spec

    elif rolling == True:
        return ri_fspec


def firing_rate_stft(df, nperseg=100, single_plot=False, dom_plot=True,
                     plot_unit='ch_101a', ax=None, overlap=None):
    """
    Function that uses the Short-Time Fourier Transform (STFT) to calculate the
    dominant frequency component of a signal as a function of time. A
    spectrogram is also calculated and can be plotted for single units.


    Input:
       - DF      : DataFrame of array Firing Rates
       - T_START : Start time [s] for STFT calculation
       - T_END   : End tiem for STFT calculation
       - NPERSEG : Number of points per window
       - SINGLE_PLOT : True if a unit spectrogram is desired
       - PLOT_UNIT   : Unit to plot spectrogram for
       - AX          : Axes to plot spectrogram

    Outputs:
       - DOM_F : DataFrame of dominant frequency estimates for each unit,
       indexed by time.

    """

    # Creat the DF for storing dom frequency evolution
    Dom_f = pd.DataFrame(columns=df.columns)

    # Loop through units to perform whole slice STFT
    for unit in df.columns:
        # STFT each unit in DF
        data = df[unit]
        f, t, z = signal.stft(data, nperseg=nperseg, detrend='linear',
                              noverlap=overlap)

        # Arrays for time and frequency
        farr = np.linspace(f[0], f[-1], z.shape[0])
        tarr = np.linspace(t[0], t[-1], z.shape[1])

        # Find dominant freqency
        idx = np.argmax(abs(z), axis=0)
        dom_f = np.array([farr[i] for i in idx])

        # Store dominant frequency array into the DF
        Dom_f[unit] = dom_f
        Dom_f.index = tarr

        # Plot the spectrogram for a sigle unit
        if single_plot == True:
            if unit == plot_unit:
                # Spectrogram
                ax.pcolormesh(t, f, np.abs(z), vmin=0,
                              shading='gouraud')  # 'gouraud')
                ax.set_title('STFT Magnitude')
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [sec]')

                if dom_plot == True:
                    # Dominant frequency
                    ax.plot(tarr, dom_f, '-o', color='r', lw=3)

    return Dom_f


def rhythmic_units(df, ri_df, threshold):
    # Function to determine the rhythmic channels. Phasic is defined such that unit
    # firing rate surpasses a rhythmicity index threshold (RI_THRESHOLD).
    # Input:
    #    - DF : DataFrame of array Firing Rates
    #    - RI_DF : DataFrame of the RI function analysis
    #    - THRESHOLD : Rhythmicity Index Threshold
    # Outputs:
    #    - RHYTHMIC_UNITS : DataFrame of unit firing rates defined as rhythmic

    rhythmic_index = ri_df[(ri_df.iloc[:, 2] > threshold)].index
    rhythmic_units = df[rhythmic_index]
    return rhythmic_units


def RI_filter(df, ri_df, ri_threshold, method, return_tonic=False):
    """
    For a data frame of RI's, RI_filter returns the units with an 
    RI > ri_threshold.

    Note: I should proably just change the FT and AutoCor functions so that the
    RI is in axis 0 - that way there would be no need for method

    """

    if method == 'Fourier':
        phasic_idx = ri_df[(ri_df.iloc[:, 0] >= ri_threshold)].index
        tonic_idx = ri_df[(ri_df.iloc[:, 0] > ri_threshold)].index

    elif method == 'Autocor':
        phasic_idx = ri_df[(ri_df.iloc[:, 1] >= ri_threshold)].index
        tonic_idx = ri_df[(ri_df.iloc[:, 1] > ri_threshold)].index

    phasic_units = df[phasic_idx]
    tonic_units = df[tonic_idx]

    if return_tonic == False:
        return phasic_units

    elif return_tonic == True:
        return phasic_units, tonic_units


def frequency_filter(df, ri_df, f_limits, method):
    # Function to determine the phasic units. Phasic is defined such that unit
    # firing rate surpasses a rhythmicity index threshold (RI_THRESHOLD) and
    # the frequency of said units firing rate is within an acceptable range
    # between PHASIC_MIN and PHASIC_MAX.
    # Input:
    #    - DF : DataFrame of array Firing Rates
    #    - RI_ACOR : DataFrame of unit autocorrelation functions
    #    - RI_THRESHOLD : Rhythmicity Index Threshold
    #    - PHASIC_MIN : Minimum frequency defined as phasic
    #    - PHASIC_MAX : Maximum frequency defined as phasic
    # Outputs:
    #    - PHASIC_UNITS : DataFrame of unit firing rates defined as phasic

    # Only frequency filter the units detected as rhythmic
    ri_df = ri_df.loc[df.columns]

    # Filter rhythmic units by their dominant frequency
    if method == 'Fourier':
        phasic_index = ri_df[(ri_df.iloc[:, 1] < (f_limits[1]))
                             & (ri_df.iloc[:, 1] > (f_limits[0]))].index

    elif method == 'Autocor':
        phasic_index = ri_df[(ri_df.iloc[:, 0] < (f_limits[1]))
                             & (ri_df.iloc[:, 0] > (f_limits[0]))].index

    insidef_units = df[phasic_index]
    outsidef_units = df.drop(columns=insidef_units.columns)

    return insidef_units, outsidef_units


# %% SYNCHRONY ANALYSIS -------------------------------------------------------


def sync_crosscorr(df, t_span):
    """
    Function for assessing the synchrony between rhythmic units using the
    cross-correlation function

    Inputs:
        - DF : Data frame of firing rates classed as rhythmic. Note that this
        analysis does not work well with non-rhythmic units
        - T_SPAN : List of teo numbers [t0,t1] which define the region of data 
        to use in the analysis

    Outputs:
        - CCORR : Data frame with N*N column vectors, each corresponing to a 
        cross-correlation function
        - CCORR_SCORE : Data frame with size {N, N}, where each element is a 
        single quantitive summary of any two signals cross-correlation


    """

    # Make storage objects for ccorr function and ccorr score
    ccor_col = []
    cs_index = []
    # Loop to add the correct column and row names
    for i in range(df.shape[1]):
        cs_index.append(df.iloc[:, i].name[3:])

        for j in range(df.shape[1]):
            string = df.iloc[:, i].name[3:]+'-'+df.iloc[:, j].name[3:]
            ccor_col.append(string)

    ccor = pd.DataFrame(columns=ccor_col)
    ccor_score = pd.DataFrame(index=cs_index, columns=cs_index, dtype=float)

    n = 0
    for i in range(df.shape[1]):     # rows
        for j in range(df.shape[1]):  # columns

            # Use data from only a section of the reccording
            t0 = t_span[0]
            t1 = t_span[1]

            # Use Scipy.signal correlate function
            cc = correlate(df.iloc[t0:t1, i]-df.iloc[t0:t1, i].mean(),
                           df.iloc[t0:t1, j]-df.iloc[t0:t1, j].mean())

            # Calculate the lags manually
            lags = np.arange(-len(df.iloc[t0:t1, i])+1, len(df.iloc[t0:t1, i]))

            # Normalise the cross-correlation
            cc = cc / max(cc)

            # Initial cross-correlation score: magnitude of 2nd largest peak
            peaks, _ = find_peaks(cc)
            cs = np.sort(cc[peaks])[-2]

            # Store ccor function and cs in separate DF's
            ccor.iloc[:, n] = cc
            ccor_score.loc[df.iloc[:, i].name[3:], df.iloc[:, j].name[3:]] = cs

            # Updatethe index varaibel for the ccor data frame
            n = n+1

    # Set index for correlation function as lags
    ccor.index = lags

    return ccor, ccor_score


# %% DRUG ANALYSIS ------------------------------------------------------------

def peak_drug_response(df, avg_window, drug_on, search_window=600, TTX=False):
    # Function to find the peak firing rate of a single unit in response to a drug
    # application. METHOD is to calculate a center-smoothed firing rate with the
    # smoothing window equal to the drug response window. The max of the smoothed
    # firing rate is the WINDOW averaged peak response. The max is only considered
    # in the region subsequent to the drug application time.
    #
    # This method leads to windows at different positions for different channels
    #
    # WARNING: If the rise time of the response is rapid (relative to WINDOW) then
    # peak firing rate may be reduced.


    # Response search window is different for TTX:
    if TTX==True:
        drug_start = df.index[-1] - 90
        drug_end = df.index[-1] - 30
        
    else:
        # Define the window in which to search for peak drug response
        drug_start = drug_on
        drug_end = drug_start + search_window  # 10min
         

    # Find maximum of smoothed firing rate traces
    peak = df[drug_start:drug_end].rolling(
        avg_window+1, center=True).mean().max(axis=0)
    # Find time of maximum responses
    peaktime = df[drug_start:drug_end].rolling(
        avg_window+1, center=True).mean().idxmax(axis=0)
    
    ### NOTE: the rolling window average actually takes 29 timepoints, not 30,
    ### hence the '+1' in the rolling argument. 

    # Store peak FR and time of peak in a DataFrame
    response = pd.concat([peak, peaktime], axis=1)
    response.columns = ['Peak FR', 'Time of peak']

    return response


def sig_response(dfa, m, drug_on, drug, std_window, disp=False):
    # Function to test for a significant response to a drug when compared to baseline activity

    # Calculate std in a period preceeding drug application
    stds = dfa[drug_on-std_window:drug_on].std()

    # Calculate max drug response
    smooth_window = 30  # defult
    pdr = peak_drug_response(dfa, smooth_window, drug, drug_on)

    # Test for significant response
    signi_response_bool = pdr["Peak FR"] > (m * stds)

    # Find units that response (true) and do not (false)
    true_idx = signi_response_bool[signi_response_bool].index
    false_idx = signi_response_bool[~signi_response_bool].index
    res = dfa[true_idx]
    nores = dfa[false_idx]

    if disp == True:
        print(' From ' + str(dfa.shape[1]) + ' active units, ' + str(res.shape[1]) +
              ' respond to ' + str(drug) + ' and ' + str(nores.shape[1]) + ' do not respond')

    return res, nores


def FR_drug_response(df, experiment, m=3, control_window=100, response_window=100,
                     search_window=600, spont_act_th=0.08, equal_control_response=True, wash_after_search=False):
    """
    A function that determines the response of a firing rate time series to an
    application of a drug. First, the response is detected as the maxium FR 
    amplitude in a SEARCH_WINDOW length of time subsequent to drug application.
    Next, a RESPONSE_WINDOW average around the caculated response is taken as
    the FR response to the drug. Th response amplitude is compared with the
    CONTROL_WINDOW averaged FR immediatly prior to drug application. If the
    response amplitude (reponse - control) is M times larger than the STD of the 
    control FR, then a response is detected.

    Notes:
        - Need to change to look at Drug 2 responses too
        - Could include 2 dicts as additional output, each giving the 30s of FR
        data used to calculate the summaries in the data frame? 
        
    
    Parameters
    ----------
    df : data frame
        data frame of firing rates.
    experiment : str
        name of the experiment.
    m : int, optional
        Number of STD's to use for significance detection. The default is 3.
    control_window : int, optional
        Length of the time window [s] to calculate the average control firing rate.
        The default is 30.
    response_window : int, optional
        Length of the time window [s] to calculate the average control firing rate.
        The default is 30.
    search_window : int, optional
        Length of the time window [s] in which to search for a maximum firing 
        rate response. The default is 600.
    spont_act_th : float, optional
        Threshold, in Hz, to define a spontaneously active unit. Default is 0.5
    
    Returns
    -------
    fr_response : Data frame
        A summary sheet of the response to the applied drug.

    """
    

    # Load the summary data for all ARC experiments
    wb = workbook_df(output_df=True)

    
    #### Define the three windows
    # Time window for BASELINE
    control_t0 = wb.loc[experiment]['Drug 1 time'] - control_window
    control_t1 = wb.loc[experiment]['Drug 1 time']

    # Time window to SEARCH for a response in, e.g. 10 mins after application
    search_t0 = wb.loc[experiment]['Drug 1 time']
    search_t1 = wb.loc[experiment]['Drug 1 time'] + search_window
    
    # Time window for WASH
    # end the wash window at the application time of the next drug
    if pd.isna(wb.loc[experiment,'Drug 2 time'])==False: 
        wash_t1 = wb.loc[experiment,'Drug 2 time']
    elif pd.isna(wb.loc[experiment,'Drug 3 time'])==False:
        wash_t1 = wb.loc[experiment,'Drug 3 time'] 
    else:
        wash_t1 = wb.loc[experiment,'TTX']
    wash_t0 = wash_t1 - control_window
    
    if wash_after_search==True:
        wash_t0 = search_t1
        wash_t1 = wash_t0 + control_window
        
        
    

        
    #### Calculate averages in the three windows
    # Calculate average FR in a *control_window* length window
    control_mean = df[int(control_t0):int(control_t1)].mean()
    control_std  = df[int(control_t0):int(control_t1)].std()
    
    # Calculate the maxium response during the search window
    # Calculate the maxium FR respose during the search window
    smooth_para = 30
    smooth_fr = df[int(search_t0):int(search_t1)].rolling(smooth_para+1, center=True).mean()
    max_res     = abs(smooth_fr-control_mean).max()
    response_t  = abs(smooth_fr-control_mean).idxmax()
    
    # Assess whether mresponse occours at the end of the window
    boo = (response_t == search_t1)
    if boo.any() == True:
        print(' A response has been found at the end of the search window, '
              ' this may indicate a monotonic firing rate', response_t[boo])   

    # Calculate average FR in a *drug_window* length window around
    # the maxium
    if equal_control_response==True:
        response_window = control_window
    response_mean = pd.Series([], dtype='float64')
    response_std = pd.Series([], dtype='float64')
    for unit in df.columns:
        # Define the time for the average calculation
        response_t0 = response_t[unit] #- response_window/2
        response_t1 = response_t[unit] + response_window

        # Calculate response statistics
        response_mean.loc[unit] = df[unit][int(response_t0):int(response_t1)].mean()
        response_std.loc[unit] = df[unit][int(response_t0):int(response_t1)].std()
    
    
    
    # Calculate average FR in *control_window* when drug has been WASHED OUT
    wash_mean = df[int(wash_t0):int(wash_t1)].mean()

    
    
    
    #### Store in data frame
    # Store max response and time of response in Data Frame
    # May be worth making an amplitude response or normalised response
    fr_response = pd.DataFrame(index=max_res.index,
                               columns=['ZT', 'Pre-drug avg', 'Spont. Active', 
                                        'App-avg', 'Response Amp.',
                                        'Response Time', 'Wash-avg', 'STD Response', 'Type'])
    
    fr_response["ZT"] = convert_2_ZT(wb.loc[experiment]['TOR'])
    fr_response["Pre-drug avg"] = control_mean
    fr_response["App-avg"]          = response_mean
    fr_response["Response Amp."]    = response_mean - control_mean
    fr_response["Response Time"]    = response_t - wb.loc[experiment]['Drug 1 time']
    fr_response["Wash-avg"]          = wash_mean
    
    # Define spontaneously active units as units with a control_mean firing
    # rate greater than some threshold
    true_cond = fr_response["Pre-drug avg"]>=spont_act_th
    fr_response.loc[true_cond, 'Spont. Active']=True
    fr_response.loc[~true_cond, 'Spont. Active']=False

    # Calculate an EXCITATORY response as a positive deviation from the mean
    # control firing rate
    excited_units = fr_response[fr_response["Response Amp."]
                                > (m * control_std)].index

    # Calculate an INHIBITORY response as a negative deviation from the mean
    # control firing rate
    inhibited_units = fr_response[fr_response["Response Amp."]
                                  < -(m * control_std)].index

    # Neurtral/No response does not differ from control mean
    neutral_units = fr_response.drop(inhibited_units).drop(excited_units).index

    # Detect units which are activated by the drug application.
    # ACTIVATED units are defined as excited units that are not spontaneously active 
    activated_units = control_mean[excited_units][control_mean.loc[excited_units] < spont_act_th].index

    # Include the Ex, In, Ne and Ac data into the fr_response Data Frame
    fr_response.loc[excited_units,   'Type'] = 'Excited'
    fr_response.loc[inhibited_units, 'Type'] = 'Inhibited'
    fr_response.loc[neutral_units,   'Type'] = 'Neutral'
    fr_response.loc[activated_units, 'Type'] = 'Activated'
    
    # add boolean column for signifiacnt response
    fr_response.loc[abs(fr_response.loc[:, 'Response Amp.'])
                    >(m*control_std), 'STD Response'] = True
    fr_response.loc[abs(fr_response.loc[:, 'Response Amp.'])
                    <=(m*control_std), 'STD Response'] = False


    return fr_response





def RI_drug_response(df, experiment, fourier_window=300, 
                     response_window_delay=300, spont_act_th=0.08, ri_th=0.02):
    """
    

    Parameters
    ----------
    df : data frame
        DF of rhythmic units, TIDA or otherwise.
    experiment : str
        Name of experiment.
    fourier_window : int, optional
        Length of data (in samples/seconds) to perform the control vs, application
        rhythmicity test on Problems occur if too small. The default is 300.
    response_window_delay : int, optional
        Delay, in seconds, after the drug application before the application 
        Fourier transform window starts. The default is 300.

    Returns
    -------
    ri_response : Data Frame
        DF containing pre and during-application RI's.

    """

    # Load the summary data for all ARC experiments
    wb = workbook_df(output_df=True)

    # Time window to take a control FR average of
    control_t0 = wb.loc[experiment]['Drug 1 time'] - fourier_window
    control_t1 = wb.loc[experiment]['Drug 1 time']

    # Time window to search for a response in, e.g. 10 mins after application
    response_t0 = wb.loc[experiment]['Drug 1 time'] + response_window_delay
    response_t1 = response_t0 + fourier_window
    
    # Time window to calculate the WASH firing rates in
    # end the wash window at the application time of the next drug
    if pd.isna(wb.loc[experiment,'Drug 2 time'])==False:
        wash_t1 = wb.loc[experiment,'Drug 2 time']
    elif pd.isna(wb.loc[experiment,'Drug 3 time'])==False:
        wash_t1 = wb.loc[experiment,'Drug 3 time'] 
    else:
        wash_t1 = wb.loc[experiment,'TTX']
    # start of wash window to ensure equal window length between control and wash
    wash_t0 = wash_t1 - fourier_window


    # Calulate the RI BEFORE drug application
    ri_pre, amp_spec, phs_spec = Fourier_Fisher_RI(df.iloc[int(control_t0):int(control_t1), :])

    # Calculate the RI DURING the application at the frequency of the control measurment
    # by performing a Fourier transform and finding the power at the pre-app frequnecy
    df_app = df.iloc[int(response_t0):int(response_t1),:]
    app_ri=[]
    T = df_app.shape[0]; N = df_app.shape[0]; dt = T/N; scale = 2*dt**2/T
    f = fftfreq(N, dt)
    for i in range(df_app.shape[1]):
        x = df_app.iloc[:,i]
        xf = scipy.fftpack.fft(np.array(x - x.mean()))
        Sxx = np.real(scale * (xf * xf.conj()))
        # Find the spectral power at the pre-application dominant frequency 
        pre_ri_power = Sxx[f==ri_pre['Dom. Freq.'][i]]
        # Convert to RI
        app_ri.append(pre_ri_power[0]/sum(Sxx))
        
    # Calculate the RI at WASH at the frequency of the control measurment
    # by performing a Fourier transform and finding the power at the pre-app frequnecy
    df_wash = df.iloc[int(wash_t0):int(wash_t1),:]
    wash_ri=[]
    T=df_wash.shape[0]; N=df_wash.shape[0]; dt=T/N; scale=2*dt**2/T
    f = fftfreq(N, dt)
    for i in range(df_wash.shape[1]):
        x = df_wash.iloc[:,i]
        xf = scipy.fftpack.fft(np.array(x - x.mean()))
        Sxx = np.real(scale * (xf * xf.conj()))
        # Find the spectral power at the pre-application dominant frequency 
        pre_ri_power = Sxx[f==ri_pre['Dom. Freq.'][i]]
        # Convert to RI
        wash_ri.append(pre_ri_power[0]/sum(Sxx))



    # Build a rhythmicity index-response data frame for a single slice
    ri_response = pd.DataFrame(index=df.columns, columns=['ZT', 'pre-freq',
                                                          'pre-RI', 'app-RI',
                                                          'Response Time', 'wash-RI',
                                                          'Sig. Response'])

    ri_response["ZT"] = convert_2_ZT(wb.loc[experiment]['TOR'])
    ri_response["pre-freq"] = ri_pre["Dom. Freq."]
    ri_response["pre-RI"] = ri_pre["RI"]
    ri_response["app-RI"] = app_ri
    ri_response["Response Time"] = tonic_switch_detection(df, experiment)
    ri_response["wash-RI"] = wash_ri
    
    # significant response test: drop below rhythmicity threshold
    cond = ri_response["app-RI"]<=ri_th
    ri_response.loc[cond, "Sig. Response"] = True
    ri_response.loc[~cond, "Sig. Response"] = False
        
    return ri_response





# def tida_tonic_response(tida_df, experiment, epoch=1, dfdt_th=0.1,
#                         median_filter_th=0.1, rolling_med_window=5,
#                         return_ddomf=False, heatmap=False, ax=None):
#     """
#     Function to detect the exsistance of a TIDA unit phasic-to-tonic response
#     (think typical TRPC activation). If a response occours within the time
#     window defined by EPOCH (e.g. "1" means drug 1), then the response time,
#     returned in the data frame RESPONSE_TIMES will be a non-zero number. A
#     response time of 0 means no response

#     Inputs:
#         - TIDA_DF     : Data frame of TIDA unit firing rates
#         - EXPERIMEANT : Name of the experiment
#         - EPOC        : Which event epoch to search for a response in
#         - DFDT_TH     : df/dt threshold to record as a valid response
#         - MEDIAN_FILTER_TH : Defines what an outlier is
#         - ROLLING_MED_WINDOW : Window used to calculate the rolling mean

#     Outputs:
#         - RESPONSE_TIMES : Series, indexed by TIDA unit name, of response times
#           for each unit. response_time=0 means there is no response

#     """

#     # Instantly raise an error if the data frame contains no TIDA units
#     if len(tida_df.columns) == 0:
#         raise ValueError(
#             'TIDA unit data frame is empty - no analysis can be performed')

#     # Load in the workbook to get relevent summary data
#     wb = workbook_df()

#     # Define the window to seach for a response
#     search_t0 = wb.loc[experiment]["Drug 1 time"]
#     search_t1 = tida_df.shape[0]

#     # Calculate the dominant frequency time series using the STFT method
#     dom_f = firing_rate_stft(tida_df, nperseg=100, single_plot=False)

#     # Filter outliers from dom_f using a median filter
#     # Calculate the rolling median
#     median = dom_f.rolling(rolling_med_window, center=True).median()

#     # Define outliers as elements that are more tha TH away the local median
#     diff = abs(dom_f - median)

#     # Filter the dom_f by replacing outliers with a suitable value
#     filtered = dom_f.where(diff < median_filter_th, median)

#     # Calculate |df_{dom}/dt|,
#     ddomf = filtered.diff().abs()

#     # Calculate the response time as the time that |df_{dom}/dt|>dfdt_threshold
#     # is crossed within the search window
#     search_idx0 = ddomf.index[pd.Series(ddomf.index > search_t0).idxmax()-1]
#     search_idx1 = ddomf.index[pd.Series(ddomf.index > search_t1).idxmax()-1]
#     response_times = (ddomf.loc[search_idx0:search_idx1] > dfdt_th).idxmax()

#     if return_ddomf == False:
#         return response_times

#     if return_ddomf == True:
#         return response_times, ddomf


def period_drug_response(df, experiment, m=3, control_window=100, response_window=100,
                     equal_control_response=True, search_window=600, spont_act_th=0.08, wash_after_search=False):
    """
    A function that determines the response of a firing rate time series to an
    application of a drug. First, the response is detected as the maxium FR 
    amplitude in a SEARCH_WINDOW length of time subsequent to drug application.
    Next, a RESPONSE_WINDOW average around the caculated response is taken as
    the FR response to the drug. Th response amplitude is compared with the
    CONTROL_WINDOW averaged FR immediatly prior to drug application. If the
    response amplitude (reponse - control) is M times larger than the STD of the 
    control FR, then a response is detected.

    Notes:
        - Need to change to look at Drug 2 responses too
        - Could include 2 dicts as additional output, each giving the 30s of FR
        data used to calculate the summaries in the data frame? 
        
    
    Parameters
    ----------
    df : data frame
        data frame of firing rates.
    experiment : str
        name of the experiment.
    m : int, optional
        Number of STD's to use for significance detection. The default is 3.
    control_window : int, optional
        Length of the time window [s] to calculate the average control firing rate.
        The default is 30.
    response_window : int, optional
        Length of the time window [s] to calculate the average control firing rate.
        The default is 30.
    search_window : int, optional
        Length of the time window [s] in which to search for a maximum firing 
        rate response. The default is 600.
    spont_act_th : float, optional
        Threshold, in Hz, to define a spontaneously active unit. Default is 0.5
    
    Returns
    -------
    fr_response : Data frame
        A summary sheet of the response to the applied drug.

    """

    # Load the summary data for all ARC experiments
    wb = workbook_df(output_df=True)

    #### Define the three windows
    # Time window for BASELINE
    control_t0 = wb.loc[experiment]['Drug 1 time'] - control_window
    control_t1 = wb.loc[experiment]['Drug 1 time']

    # Time window to SEARCH for a response in, e.g. 10 mins after application
    search_t0 = wb.loc[experiment]['Drug 1 time']
    search_t1 = wb.loc[experiment]['Drug 1 time'] + search_window
    
    # Time window for WASH
    # end the wash window at the application time of the next drug
    if pd.isna(wb.loc[experiment,'Drug 2 time'])==False: 
        wash_t1 = wb.loc[experiment,'Drug 2 time']
    elif pd.isna(wb.loc[experiment,'Drug 3 time'])==False:
        wash_t1 = wb.loc[experiment,'Drug 3 time'] 
    else:
        wash_t1 = wb.loc[experiment,'TTX']
    wash_t0 = wash_t1 - control_window
    
    if wash_after_search==True:
        wash_t0 = search_t1
        wash_t1 = wash_t0 + control_window
    
    
    
    
    #### Caclulate averages in the three set windows
    # Calculate average in a *control_window* length window: BASELINE
    control_mean = df[int(control_t0):int(control_t1)].mean()
    control_std  = df[int(control_t0):int(control_t1)].std()
    
    # Calculate the maxium response during the search window
    max_res     = abs(df[int(search_t0):int(search_t1)]-control_mean).max()
    response_t  = abs(df[int(search_t0):int(search_t1)]-control_mean).idxmax()
    
    # Assess whether mresponse occours at the end of the window
    boo = (response_t == search_t1)
    if boo.any() == True:
        print(' A response has been found at the end of the search window, '
              ' this may indicate a monotonic firing rate', response_t[boo])

    # Calculate average in a *drug_window* length window around
    # the peak response
    if equal_control_response==True:
        response_window = control_window
    response_mean = pd.Series([], dtype='float64')
    response_std = pd.Series([], dtype='float64')
    for unit in df.columns:
        # Define the time for the average calculation
        response_t0 = response_t[unit] 
        response_t1 = response_t[unit] + response_window

        # Calculate response statistics
        response_mean.loc[unit] = df[unit][int(response_t0):int(response_t1)].mean()
        response_std.loc[unit] = df[unit][int(response_t0):int(response_t1)].std()
    
    # Calculate average period in *control_window* when drug has been WASHED OUT
    wash_mean = df[int(wash_t0):int(wash_t1)].mean()

    
    
    #### Store three averages and other data in a data frame
    # Store max response and time of response in Data Frame
    # May be worth making an amplitude response or normalised response
    response = pd.DataFrame(index=max_res.index,
                               columns=['ZT', 'BL-P','App-P', 'Response Amp.',
                                        'Response Time', 'Wash-P', 'Type', 'STD Response'])
    
    response["ZT"] = convert_2_ZT(wb.loc[experiment]['TOR'])
    response["BL-P"] = control_mean
    response["App-P"]          = response_mean
    response["Response Amp."]    = response_mean - control_mean
    response["Response Time"]    = response_t - wb.loc[experiment]['Drug 1 time']
    response["Wash-P"]          = wash_mean
    

    # Calculate an EXCITATORY response as a positive deviation from the mean
    # control firing rate
    increase_units = response[response["Response Amp."] > (m * control_std)].index

    # Calculate an INHIBITORY response as a negative deviation from the mean
    # control firing rate
    decrease_units = response[response["Response Amp."] < -(m * control_std)].index

    # Neurtral/No response does not differ from control mean
    neutral_units = response.drop(decrease_units).drop(increase_units).index

    # Include the Ex, In, Ne and Ac data into the fr_response Data Frame
    response.loc[increase_units,   'Type'] = 'Increased'
    response.loc[decrease_units, 'Type'] = 'Decreased'
    response.loc[neutral_units,   'Type'] = 'Neutral'
    
    # add boolean column for signifiacnt response
    response.loc[abs(response.loc[:, 'Response Amp.'])
                    >(m*control_std), 'STD Response'] = True
    response.loc[abs(response.loc[:, 'Response Amp.'])
                    <=(m*control_std), 'STD Response'] = False

    return response




def tonic_switch_detection(df, experiment, search_window=600, dfdt_th=0.05, 
                           median_filter_th=0.01, median_filter_window=5):
    """

    Parameters
    ----------
    df : Data frame
        Data frame of TIDA unit firing rates.
    experiment : str
        Name of the experiment.
    search_window : int, optional
        Window length over which to search for a response . The default is 600.
    dfdt_th : float, optional
        df/dt threshold to record as a valid response. The default is 0.05.
    median_filter_th : float, optional
        Defines what an outlier is. The default is 0.01.
    median_filter_window : int, optional
        Window used to calculate the rolling median. The default is 5.
        
    Raises
    ------
    an
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    response_times : Series
        Series indexed by TIDA unit name, of response times
        #           for each unit. response_time=0 means there is no response.

    """
    
    # Instantly raise an error if the data frame contains no units
    if len(df.columns) == 0:
        raise ValueError(
            'Data frame is empty - no analysis can be performed!')
        
    # Load in the workbook to get relevent summary data
    wb = workbook_df()

    # Define the window to seach for a response
    search_t0 = wb.loc[experiment]["Drug 1 time"]
    search_t1 = search_t0 + search_window
    
    # Calculate the dominant frequency time series using the STFT method
    dom_f = firing_rate_stft(df, nperseg=100, single_plot=False)
    
    # median filter the dom_f time series' 
    dom_f = rolling_median_filt(dom_f, window_size=median_filter_window, threshold=median_filter_th, double=True)
    
    # Calculate |df_{dom}/dt|,
    ddomf = dom_f.diff().abs()
    
    # Calculate the response time as the time that |df_{dom}/dt|>dfdt_threshold
    # is crossed within the search window
    search_idx0 = ddomf.index[pd.Series(ddomf.index > search_t0).idxmax()-1]
    search_idx1 = ddomf.index[pd.Series(ddomf.index > search_t1).idxmax()-1]
    response_times = (ddomf.loc[search_idx0:search_idx1] > dfdt_th).idxmax()
    
    return response_times
    
    
    
    
    
    


def plot_FR_response_pie(fr_response, fig, experiment, plot_traces=False, num_traces=1, time_window=1000):
    """
    A simple function that produces a pie chart of the resposnes observed from
    a simple FR response function. Responses can either be inhibited, excited 
    or neutral (neither)
    """

    # Load the summary data for all ARC experiments
    wb = workbook_df(output_df=True)

    # Time points to plot over
    t0 = wb.loc[experiment]['Drug 1 time']-60
    t1 = t0 + time_window

    # Numbers in each group
    num_excited = fr_response[fr_response['Type'] == 'Excited'].shape[0]
    num_inhibited = fr_response[fr_response['Type'] == 'Inhibited'].shape[0]
    num_neutral = fr_response[fr_response['Type'] == 'Neutral'].shape[0]
    num_activated = fr_response[fr_response['Type'] == 'Activated'].shape[0]
    total = fr_response.shape[0]

    # Make dummy lists. These aid in only plotting responses with n>0
    num_list = [num_excited, num_inhibited, num_neutral, num_activated]
    lab_list = ['Excited', 'Inhibited', 'Neutral', 'Activated']
    col_list = ['r', 'b', 'C2', 'C4']

    # Check everything adds up
    if num_excited + num_inhibited + num_neutral + num_activated != total:
        print('Total number of neurons do not match')

        # Make labels based upon the responses
    labels = []
    numbers = []
    colors = []
    for i in range(len(num_list)):
        if num_list[i] > 0:
            labels.append(lab_list[i])
            numbers.append(num_list[i])
            colors.append(col_list[i])

    # Calculate how many example plots there should be
    rows = len(labels)

    if plot_traces == True:
        gs = GridSpec(rows, 3, fig)
        ax = fig.add_subplot(gs[:, 0])
    else:
        ax = fig.add_subplot(111)

    # Pie chart text function

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)
        return my_autopct

    # plot the pie chart
    ax.margins(x=0.1)
    ax.pie(x=numbers, labels=labels, colors=colors,
           wedgeprops={'linewidth': 1},
           autopct=make_autopct(numbers),
           shadow=True,
           startangle=0,
           explode=[0.1]*len(labels))
    ax.axis('equal')
    ax.axis('off')
    ax.text(0.5, -1.2, "n = "+str(total)+" units")

    if plot_traces == True:
        # Import firing rate data from pickle object
        path = r"C:\Users\kh19883\OneDrive - University of Bristol\Documents"\
            "\PhD Neural Dynamics\ARC-Bursting\MEA Data\OFS Sorted Spikes"\
            "\TDistEM_DOM10_I5\Pandas Objects\\"
        path = path+'TTX Filtered Units\\'+experiment+'.pkl'
        df = pd.read_pickle(path)

        n = 0
        for i in range(len(numbers)):
            units = fr_response[fr_response["Type"] == labels[i]].index

            # Skip out plotting any groups with zero members
            if len(units) == 0:
                continue
            else:
                ax = fig.add_subplot(gs[n, 1:])
                df.loc[t0:t1, units].sample(n=num_traces, axis=1).rolling(10,
                                                                          center=True).mean().plot(ax=ax, legend=True, color=colors[i])
                ax.legend(bbox_to_anchor=(1.05, 1))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                n = n+1
                if i != len(numbers)-1:
                    # ax.spines['bottom'].set_visible(False)
                    ax.set_xticklabels([])

        # Resturn the number of units in each response catagory
        return [num_excited, num_inhibited, num_activated, num_neutral]


def plot_ein_response_heatmap(df, fr_response, experiment, ax, cmap="bwr",
                              yticklabels=True):
    """
    Function to plot the Excited, Inhibited and Neutral responses to a drug
    (incuding Activated)

    Inputs:
        - DF : Full DF of firing rates

    """

    # Load the summary data for all ARC experiments
    wb = workbook_df(output_df=True)

    # Set the segment to detect rythmicity and TIDAs
    baseline_end = wb.loc[experiment, 'Drug 1 time']

    # Find units which respond
    ex_idx = fr_response[fr_response["Type"] == 'Excited'].index
    ac_idx = fr_response[fr_response["Type"] == 'Activated'].index
    in_idx = fr_response[fr_response["Type"] == 'Inhibited'].index
    resposive_units = ex_idx.union(ac_idx).union(in_idx)

    # Define the firing rate DF of responsive units (RU)
    ru = df[resposive_units]

    # Subtractive normaise by basline mean
    ru_norm = ru - ru[0:baseline_end].mean()

    # Reorder the units depemding on FR amplitude
    ru_norm = ru_norm.loc[:, ru_norm.max().sort_values(ascending=False).index]

    # Time across y axis
    ru_norm = ru_norm.transpose()

    # Smooth the responses
    ru_norm = ru_norm.rolling(5, center=True, axis=1).mean()

    # Plot the heat map
    hm = sns.heatmap(ru_norm, ax=ax, cmap=cmap, xticklabels=False,
                     yticklabels=yticklabels,
                     cbar_kws={'label': 'Spike Frequency Amplitude'},
                     vmin=-5, vmax=5)

    ticks = np.arange(0, ru_norm.shape[1], 200)
    ax.set_ylim([ax.get_ylim()[0]+0.5, ax.get_ylim()[1]-0.5])
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks], rotation=45)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_ylabel('Unit', fontsize=15)

    return hm


# %% STATISTICS ---------------------------------------------------------------

def normal_test(x, alpha=0.05):
    """
     Function to run multiple normality test for the data, x. 
     
     The null-hypothesis for all the tests is that the data x is drawn from a 
     normal distribution. p>alpha => accept H0 (distribution is normal)


    Parameters
    ----------
    x : array_lik 
        Array of data.
    alpha : float, optional
        p-value threshold The default is 0.05.

    Returns
    -------
    None.

    """
    

    # The null-hypothessi for all the tests below is that the data x is drawn from a random
    # distribution. p>alpha => accept H0 (distribution is normal)

    # Apply the Shapiro-Wilk test
    k2, p = stats.shapiro(x)
    if p > alpha: print("Shapiro-Wilk test:           NORMAL, p={:g}".format(p))
    else: print("Shapiro-Wilk test:           Not normal, p={:g}".format(p))

    # Apply the D’Agostino’s K-squared test
    k2, p = stats.normaltest(x)
    if p > alpha: print("D’Agostino’s K-squared test: NORMAL, p={:g}".format(p))
    else: print("D’Agostino’s K-squared test: Not normal, p={:g}".format(p))

    # Apply the Chi-Square Normality test
    k2, p = stats.chisquare(x)
    if p > alpha: print("Chi-Square Normality test:   NORMAL, p={:g}".format(p))
    else: print("Chi-Square Normalitytest:    Not normal, p={:g}".format(p))

    # Apply Lilliefors Normality test
    from statsmodels.stats.diagnostic import lilliefors
    k2, p = lilliefors(x)
    if p > alpha: print("Lilliefors Normality test:   NORMAL, p={:g}".format(p))
    else: print("Lilliefors Normality test:   Not normal, p={:g}".format(p))

    # Apply Kolmogorov-Smirnov test
    k2, p = stats.kstest(x, 'norm')
    if p > alpha: print("Kolmogorov-Smirnov test:     NORMAL, p={:g}".format(p))
    else: print("Kolmogorov-Smirnov test:     Not normal, p={:g}".format(p))



# %% TTX FILTERS -------------------------------------------------------------

def ttx_filter(df, ttx_fr_threshold, smooth_window, ttx_on):
    # Function to filter out the single units that are unresponsive to TTX
    # (tetrodotoxin) application. TTX should abolish all action potentials, so
    # any activity remaining post-TTX is presumably noise and noisy units can
    # be disregarded.
    # Input:
    #   - DF : DataFrame of single units
    #   - TTX_FR_THRESHOLD : freqency cutoff during TTX application
    #   - SMOOTH_WINDOW : see **peak_drug_response**
    #   - TTX_ON : time of TTX application
    # Output:
    #   - ACTIVE_UNITS : DataFrame of units that pass the TTX filter

    # Find the drug response of TTX
    ttx_res = peak_drug_response(df, smooth_window, 'TTX', ttx_on)

    # Define the active units as units with suffciently low firing rate after
    # TTX application
    active_idx = ttx_res[ttx_res["Peak FR"] < ttx_fr_threshold].index
    active_units = df[active_idx]

    return active_units


def ttx_filter2(df, ttx_fr_threshold, ttx_fr_window):
    # Function to filter out the single channels that are unresponsive to TTX
    # (tetrodotoxin) application. TTX should abolish all action potentials, so
    # any activity remaining post-TTX is presumably noise and noisy units can
    # be disregarded. Responsivness calculated as average firing rate over
    # ttx_fr_window seconds before the experiment ends.
    #
    # This algorithm assumes TTX application, without wash, at the end of an
    # experiment and further assumes no noise-incurring disturbances.
    #
    # Input:
    #   - DF : DataFrame of single channels
    #   - TTX_FR_THRESHOLD : freqency cutoff during TTX application
    # Output:
    #   - ACTIVE_UNITS : DataFrame of channels that pass the TTX filter

    # Calculate the average firing rate for [ttx_fr_window] seconds before the end of expeiment
    ttx_fr = df.iloc[-ttx_fr_window:, :].mean()

    # Make a df of identified neuronal chennels
    neuron_ind = ttx_fr[ttx_fr < ttx_fr_threshold].index
    neuron_chns = df[neuron_ind]

    # Make df of noise chennels --- comment this out for faster algorithm
    noise_ind = ttx_fr[ttx_fr >= ttx_fr_threshold].index
    noise_chns = df[noise_ind]

    return neuron_chns, noise_chns


# %% BURST ANALYSIS -----------------------------------------------------------


def read_burstExcel(data_path_burst, sheet_name, rhythmic_channels):
    # Function that imports Burst Analysis data from excel (orginally from
    # NeuroExplorer) and stores the data in a DataFrame (FD) in which the
    # Burst Paramters form the top column names.
    # All channels from are burst analyised, so this function selects only
    # the rhythmic channels defined by RHYTHMIC_CHANNELS.
    # The output UNITS is the list of units that have been analysed, useful
    # for a further function
    #
    # DataFrame organized as:
    #
    #               Duaration | End | Start | Start | MeanISI | PeakFreq | #Spikes
    #            ch_a,...,ch_n
    #              ------------------------------------------------------------
    # burst_index |
    #
    #
    #
    #
    #

    df = pd.read_excel(data_path_burst, sheet_name, header=0)

    # First column names (the units)
    units = []
    for i in range(0, int((len(df.columns)+1)/6)):
        units.append(df.columns[6*i][0:7].strip())

    # Second column names (burst paramters)
    burst_params = []
    for i in range(0, 6):
        burst_params.append(df.columns[i][7:].strip())

    # Form MultiIndex column headers
    df.columns = pd.MultiIndex.from_product([units, burst_params])

    # Filter out the non-rythmic channels
    df = df[rhythmic_channels.columns.tolist()]
    # Replave the units definition
    units = rhythmic_channels.columns.tolist()

    # Swap the column levels so that top column level is the paramters
    fd = df.swaplevel(axis=1)
    fd = fd.sort_index(axis=1)

    return fd, units


def read_burstNex(df, channel_str_length=7, num_burst_props=8):
    """
    Function that imports Burst Analysis data from excel (orginally from
    NeuroExplorer) and stores the data in a DataFrame (FD) in which the
    Burst Paramters form the top column names.

    All channels from are burst analyised, so this function selects only
    the rhythmic channels defined by RHYTHMIC_CHANNELS.
    The output UNITS is the list of units that have been analysed, useful
    for a further function
    #
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


def baseline_burst_raster(spike_times, burst_params, units, end_time):

    # Function to visulaise the spike events along with the burst they belong to
    # Input:
    #   - SPIKE_TIME: DataFrame of spike times, different channels in different
    #                 columns
    #   - BURST_PARAMS: DataFrame of burst paramters calculated by NeuroExplorer
    #   - UNITS: List of unit/channel names to plot. They should be a subset of
    #            the units in BURST_PARMSS
    #   - END_TIME: End time of visualization.
    # Output:
    #   - a figure
    #
    # Notes: This plots rasters from t=0. For more robust usage, and interval
    # type calculation for plotting in [t_start, t_end] should be made.

    # A BASELINE raster visualization that starts from t=0.
    start_time = 0

    # Find index of each row closest to t=600s within a tolerance
    # Loop ensures a tolerance such that all indices are found
    num_zeros = 1
    tolerance = 1
    while num_zeros > 0:
        idx = spike_times[units].sub(end_time).abs().lt(tolerance).idxmax()
        tolerance = tolerance + 1
        num_zeros = (idx == 0).sum()

    # Make list of series
    raster_times = []
    for chn in units:
        raster_times.append(spike_times[chn][0:idx[chn]])

    # Create axis
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111)

    # Plot the raster diagram
    ax.eventplot(raster_times, linelengths=0.6)

    #

    # Plot bursts
    height = 0.8

    # Sequentially plot each burst
    for i in range(len(units)):
        rect = []
        num_bursts = len(burst_params['BurstStart'][units[i]])
        for j in range(num_bursts):
            if burst_params['BurstStart'][units].iloc[j, i] < end_time:
                rect.append(patches.Rectangle([burst_params['BurstStart'][units].iloc[j, i], i-height/2],
                                              burst_params['BurstDuration'][units].iloc[j, i],
                                              height)
                            )
        ax.add_collection(PatchCollection(rect, color=[1, 0.816, 0.663]))
        #ax.hlines([i]*num_bursts, burst_start_times[units[i]], burst_end_times[units[i]], color='C1')

    # Set axis paramters
    ax.set_xlabel('time [s]', fontsize=15)
    ax.set_ylabel('channel ', fontsize=15)
    ax.set_yticks(np.arange(len(units)))
    ax.set_yticklabels(units, fontsize=15)
    ax.set_xlim([start_time-1, 100])
    plt.show()
    
    
    
    
    
def run_burst_analysis(doc, pars='default', template="burst_interval_01", 
                       channel_str_length=13, tspan=None):
    """
    Function to run a NeuroExplorer burst analysis and store both the full burst
    analysis data and the summary in a data frame.

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
        pars = {"Max Int. (sec)": 2.5,
                "Max End Int.": 3,
                "Min Interburst Int.": 1,
                "Min Burst Duration": 0.1,
                "Min Num. Spikes": 1
                }
                   

    # set algorithm parameters by modifying the exsisting template parameters
    for par_name in pars:
        nex.ModifyTemplate(doc, template, par_name, str(pars[par_name]))
        
    # set the time to analyize over
    #if tspan!=None:
    # nex.ModifyTemplate(doc, template, "Select Data From (sec)", str(tspan[0]))
    # nex.ModifyTemplate(doc, template, "Select Data To (sec)", str(tspan[1]))

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
    burst_res = read_burstNex(results_df, channel_str_length=channel_str_length)

    # data frame for summary file
    summary = np.transpose(summary)
    burst_sum = pd.DataFrame(summary[:,1:], columns=sum_col_names[1:],
                             index=summary[:,0], dtype=float)
    
    return burst_res, burst_sum    






#%% DVC SIMULATIONS




def solve_anv01t01(icds, f_a, f_v, f_n=2*np.pi/22.5, c1=0, c2=0, s0n=1, s0v=1,
                   K_an=0.031, K_na=0.041, gamma_n=0.77, K_av=-0.045, K_va=-0.007,
                   tspan=(0,200)):
    """
    Function that solves the anv01t1 model. '01' refers to reciprical AP-NTS and 
    AP-4Vep coupling. 't01' refers to linear decay of coupling strengths.

    Parameters
    ----------
    icds : TYPE
        DESCRIPTION.
    f_a : TYPE
        DESCRIPTION.
    f_v : TYPE
        DESCRIPTION.
    f_n : TYPE, optional
        DESCRIPTION. The default is 2*np.pi/22.5.
    c1 : TYPE, optional
        DESCRIPTION. The default is 0.
    c2 : TYPE, optional
        DESCRIPTION. The default is 0.
    s0n : TYPE, optional
        DESCRIPTION. The default is 1.
    s0v : TYPE, optional
        DESCRIPTION. The default is 1.
    K_an : TYPE, optional
        DESCRIPTION. The default is 0.031.
    K_na : TYPE, optional
        DESCRIPTION. The default is 0.041.
    gamma_n : TYPE, optional
        DESCRIPTION. The default is 0.77.
    K_av : TYPE, optional
        DESCRIPTION. The default is -0.045.
    K_va : TYPE, optional
        DESCRIPTION. The default is -0.007.
    tspan : TYPE, optional
        DESCRIPTION. The default is (0,200).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    def sn_fun(t):
        s = s0n - c1*t
        if s<0: s=0
        return s

    def sv_fun(t):
        s = s0v - c2*t
        if s<0: s=0
        return s

    def anv_sync(t, x):
        th_a, th_n, th_v = x
        sn = sn_fun(t)
        sv = sv_fun(t)
        dth_a = f_a + sn*K_an*np.sin(th_n-th_a + gamma_n) + sv*K_av*np.sin(th_v-th_a)
        dth_n = f_n + sn*K_na*np.sin(th_a-th_n - gamma_n)
        dth_v = f_v + sv*K_va*np.sin(th_a-th_v)
        return [dth_a, dth_n, dth_v]
    
    
    sol = solve_ivp(anv_sync, tspan, icds, t_eval=np.arange(tspan[0], tspan[1]), rtol=1e-5, etol=1e-8)

    return sol
    
    
    
    
    
    
    
    
    
    
    
    
    
    


#%% CODE EXPLAINATIONS

def PLOT_FR_drug_response(df, experiment, m=3, control_window=60, response_window=60,
                     search_window=600, example_calc=None, ax=None):
    """
    A function that determines the response of a firing rate time series to an
    application of a drug. First, the response is detected as the maxium FR 
    amplitude in a SEARCH_WINDOW length of time subsequent to drug application.
    Next, a RESPONSE_WINDOW average around the caculated response is taken as
    the FR response to the drug. This response is compared with the
    CONTROL_WINDOW averaged FR immediatly prior to drug application. If the
    response amplitude (reponse - control) is M times larger than the STDof the 
    control FR, then a response is detected.

    Notes:
        - Need to change to look at Drug 2 responses too
    """

    # Load the summary data for all ARC experiments
    wb = workbook_df(output_df=True)

    # 60s window to take a control FR average of
    control_t0 = wb.loc[experiment]['Drug 1 time'] - control_window
    control_t1 = wb.loc[experiment]['Drug 1 time']

    # Find maxium FR in a post-drug window
    search_t0 = wb.loc[experiment]['Drug 1 time']
    search_t1 = wb.loc[experiment]['Drug 1 time'] + search_window

    # Calculate the maxium FR respose during the search window
    smooth_para = 30
    smooth_fr = df[search_t0:search_t1].rolling(
        smooth_para+1, center=True).mean()
    max_fr = smooth_fr.max()
    max_t = smooth_fr.idxmax()

    # Assess whether max response occours at the end of the window
    # This may only be useful when max_fr is calculated on a smoothed
    # FR time series
    boo = (max_t == search_t1)
    if boo.any() == True:
        print(' A maxium has been found at the end of the search window, '
              ' this may indicate a monotonic firing rate', max_t[boo])

    # Calculate average FR in a *control_window* length window
    control_mean = df[control_t0:control_t1].mean()
    control_std  = df[control_t0:control_t1].std()

    # Calculate average FR in a *drug_window* length window around
    # the maxium
    response_window = control_window
    response_mean = pd.Series([])
    response_std = pd.Series([])
    for unit in df.columns:
        # Define the time for the average calculation
        response_t0 = max_t[unit] - response_window/2
        response_t1 = max_t[unit] + response_window/2

        # Calculate response statistics
        response_mean.loc[unit] = df[unit][response_t0:response_t1].mean()
        response_std.loc[unit] = df[unit][response_t0:response_t1].std()

    # Store max response and time of response in Data Frame
    # May be worth making an amplitude response or normalised response
    fr_response = pd.DataFrame(index=max_fr.index,
                               columns=['Mean FR Response', 'Response Amp.',
                                        'Response Time', 'Type'])
    fr_response["Mean FR Response"] = response_mean
    fr_response["Response Amp."] = response_mean - control_mean
    fr_response["Response Time"] = max_t

    # Calculate an EXCITATORY response as a positive deviation from the mean
    # control firing rate
    excited_units = fr_response[fr_response["Response Amp."]
                                > m * control_std].index

    # Calculate an INHIBITORY response as a negative deviation from the mean
    # control firing rate
    inhibited_units = fr_response[fr_response["Response Amp."]
                                  < -m * control_std].index

    # Neurtral/No response does not differ from control mean
    neutral_units = fr_response.drop(inhibited_units).drop(excited_units).index

    # Detect units which are activated by the drug application. Can only detect
    # units which are depolarised / excited
    activated_units = control_mean[excited_units][control_mean.loc[excited_units] == 0].index

    # Include the Ex, In, Ne and Ac data into the fr_response Data Frame
    fr_response.loc[excited_units,   'Type'] = 'Excited'
    fr_response.loc[inhibited_units, 'Type'] = 'Inhibited'
    fr_response.loc[neutral_units,   'Type'] = 'Neutral'
    fr_response.loc[activated_units, 'Type'] = 'Activated'

    # % The rest of this function is concerned with plotting examples to
    # illustrate the method.
    # Good example is 'ch_106b' of '220113_s1'

    if example_calc != None:

        if ax == None:
            raise ValueError(
                'Provide an axis to plot the example calculation on')

        for unit in example_calc:

            # Plot the FR time series
            df[unit][control_t0:search_t1 +
                     50].rolling(3, center=True).mean().plot(ax=ax)

            # Plot the cotrol statistics
            ax.hlines(control_mean[unit], control_t0, control_t1, 'r', ls='--')
            ax.hlines(control_mean[unit]+control_std[unit],
                      control_t0, control_t1, 'r')
            ax.hlines(control_mean[unit]-control_std[unit],
                      control_t0, control_t1, 'r')
            ylim = [0, 7]
            ax.set_ylim(ylim)

            # plot the response statistics
            response_t0 = max_t[unit] - response_window/2
            response_t1 = max_t[unit] + response_window/2
            ax.hlines(response_mean[unit], response_t0,
                      response_t1, 'r', ls='--')

            # Add some useful annitations

            # Control STD arrow
            ax.annotate("", xy=(control_t1, control_mean[unit]+control_std[unit]),
                        xytext=(control_t1, control_mean[unit]),
                        arrowprops=dict(arrowstyle="<|-|>", color='r'),
                        color='r')
            ax.annotate(r"$\sigma_c$", xy=(0.5*(control_t0+control_t1), control_mean[unit]+control_std[unit]),
                        xytext=(0.5*(control_t0+control_t1),
                                control_mean[unit]+control_std[unit]*1.2),
                        fontsize=15,
                        color='r')

            # Control m*STD arrow
            ax.annotate("", xy=(control_t1+10, control_mean[unit] + m*control_std[unit]),
                        xytext=(control_t1+10, control_mean[unit]),
                        arrowprops=dict(arrowstyle="-|>", color='r'),
                        color='r')

            # Response threshold
            ax.annotate(str(m)+r"$\sigma_c$ = response threshold",
                        xy=(control_t1+10,
                            control_mean[unit] + m*control_std[unit]*1.2),
                        xytext=(control_t1+10,
                                control_mean[unit] + m*control_std[unit]*1.2),
                        fontsize=15,
                        color='r')
            ax.hlines(control_mean[unit] + m*control_std[unit],
                      control_t1+10, search_t1, color='r')

            # Search window
            rect = patches.Rectangle((search_t0, ylim[0]+1), search_window, ylim[1]-ylim[0]-2,
                                     facecolor='k', alpha=0.1)
            ax.add_patch(rect)

            # Response Window
            rect = patches.Rectangle((response_t0, ylim[0]+1), response_window, ylim[1]-ylim[0]-2,
                                     facecolor='k', alpha=0.1)
            ax.add_patch(rect)

    return fr_response
