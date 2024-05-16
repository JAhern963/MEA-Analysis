# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:08:44 2022

@author: kh19883

This module is for scripts which cal upon NeuroExplorer functions and processes.

Note that this module must be used alongside the license key for the
Neuroexplorer software.


"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('C:\\ProgramData\\Nex Technologies\\NeuroExplorer 5 x64')
import nex
sys.path.append('C:\\Users\\kh19883\\OneDrive - University of Bristol\\Documents\\PhD Neural Dynamics')
#from myModules import rhythm as rh


def spike_times(experiment, exp_path, save_path, source='MCD'):
    """
    SPIKE_TIMES is a function that accepts the plexon spike file EXPERIMENT and
    returns a data frame containing the spike times for every unit as series
    
    INPUT:
        - Experiment: A string in the form 'yymmdd_sx' to describe the
        experiment
        - Exp_Path: A string describing the directory of the experiment .plx file
        - Save_path:A string to identify the directory where the output pickle
        object will be saved.
    
    OUTPUT:
        - A pickled data frame saved to a directory indicated by save_path.
    
    """
    
    
    # Construct the path where .plx is saved
    path = exp_path+'\\'+experiment+'_spks.plx'
    # Open the PLX file
    doc = nex.OpenDocument(path)
    
    # Genreate list of channel names
    num_channels = len(nex.NeuronVars(doc))
    channels=[]
    for i in range(num_channels):
        channels.append( nex.GetVarName(doc, i+1, "neuron") )
        
    # Genrate and save a DataFrame of TimeStamps
    time_stamps = pd.DataFrame(columns=channels)
    for chn in channels:
        time_stamps[chn] = pd.Series( doc[chn].Timestamps() )

    # Convert new MCE colunm names to old MCD names 
    if source == 'MCE':
        
        # Remove the REF electrode from the data frame - need to remove all REF units, not just idx=-1
        time_stamps = time_stamps.drop('SeSt-Label_Ref_ID_14_Str_Spike a', axis=1)
        
        # Convert the column names to old format
        time_stamps.columns = convert_MCE_columnidx_to_MCD(time_stamps.columns)   
    
    # Save as a pickled data frame
    save_path = save_path+'\\'+experiment+'.pkl'
    pd.to_pickle(time_stamps, save_path)
    
    
    
    
def firing_rate(experiment, exp_path, save_path, source='MCD'):
    """
    FIRING_RATE is a function that accepts the plexon spike file EXPERIMENT and
    returns a data frame containing the 1s binned histogrma of spikes
    (resulting in a 1s sampled firing rate) for every unit as series. 
    
    INPUT:
        - Experiment: A string in the form 'yymmdd_sx' to describe the
        experiment
        - Exp_Path: A string describing the directory of the experiment .plx file.
          Use the form: r"C:\dir0\dir1\...\dirN"
        - Save_path:A string to identify the directory where the output pickle
        object will be saved.
    
    OUTPUT:
        - A pickled data frame saved to a directory indicated by save_path.
    
    """
        
    # Construct the path where .plx is saved
    path = exp_path+'\\'+experiment+'_spks.plx'
    # Open the PLX file
    doc = nex.OpenDocument(path)
    
    # Execute 'Rate Histogram' Analysis of ALL chnanels
    nex.ApplyTemplate(doc, "RateHistograms")
    RateResults = doc.GetAllNumericalResults()
    
    # Generate and Save a DataFrame for Rate Histograms
    num_channels = len(nex.NeuronVars(doc))
    Rates = pd.DataFrame(index=RateResults[0])
    for i in range(num_channels):
        col_name = nex.GetNumResColumnName(doc,i+2)
        Rates[col_name] = RateResults[i+1]
        
        
    # Convert new MCE colunm names to old MCD names 
    if source == 'MCE':
        
        # Remove the REF electrode from the data frame - need to remove all REF units, not just idx=-1
        Rates = Rates.drop('SeSt-Label_Ref_ID_14_Str_Spike a', axis=1)
        
        # Convert the column names to old format
        Rates.columns = convert_MCE_columnidx_to_MCD(Rates.columns) 
        
    # Save as a pickled data frame
    save_path = save_path+'\\'+experiment+'.pkl'
    pd.to_pickle(Rates, save_path)
        
    
    
    
    


def convert_MCE_columnidx_to_MCD(old_columns):
    '''
    Function to convert the MCE/Nex/OFS sorter file names into the old MCD file names.
    Specifically, the old names for individual units were 'ch_CRi',
    where C=column, R=row and i=spike sorted unit. The new files have the format 
    'SeSt-Label_CR_ID_X_Str_Spike i' or 'SeSt-Label_CR_ID_X_Str_Spike Di'.

    I have no idea what the X means or the D that occurs at idx=-2.

    This function converts from new to old 

    '''

    new_columns=[]
    for i in range(len(old_columns)):

        old_str = old_columns[i]
        new_str = convert_MCE_columnname_to_MCD(old_str)
        new_columns.append(new_str)

    return new_columns
    
    
    
def convert_MCE_columnname_to_MCD(old_str):
    '''
    Nested function for the above convesion function which actually does the 
    convestion of the old string into the new one
    '''
    
    # Retreive the C,R position of the unit 
    RC_str = old_str.partition("_")[2].partition("_")[0]
    
    # Retreive the unit indentifier
    unit_str = old_str[-1]
    
    # Combine RC and unit strings
    new_str = "ch_" + RC_str + unit_str
    
    return new_str
    
    
        
    
    
    
    
    
    
    
    