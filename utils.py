'''
    CSE583 PRML Term Project
    Few Utils functions
    
    Author: Anurag Pendyala
    PSU Email: app5997@psu.edu
    Description: File containing various utility functions
    
'''

import numpy as np
import librosa
import matplotlib.pyplot as plt

def combine_npy_dicts(file_list):
    '''
        Function that takes file list and creates a cummulative dataset
    '''
    combined_dict = {}

    for file in file_list:
        npy_dict = np.load(file, allow_pickle=True).item()

        for key in npy_dict.keys():
            if key not in combined_dict:
                combined_dict[key] = []
            combined_dict[key].extend(npy_dict[key])
    
    return combined_dict

def text_to_class(y):
    '''
        Function that converts string classes into integers for the model
    '''
    # change this label map according to the application
    map = {
        'american': 0,
        'australian': 1,
        'british': 2,
        'indian': 3
    }

    return np.array([map[x] for x in y])

def class_to_text(y):
    '''
        Function that reconverts integer classes to strings
    '''
    map = {
        0: 'american',
        1: 'australian',
        2: 'british',
        3: 'indian'
    }

    return np.array([map[x] for x in y])

def save_mfcc_from_audio(audio_path, out_name = 'mfcc', max_len = 1.8):
    '''
        Function that saves the MFCC Spectrum_plot for a given audio file path
    '''
    y, sr = librosa.load(audio_path, sr = 22050)

    n = len(y)
    k = int(sr*max_len)

    start = (n-k)//2
    end = start + k

    # taking the middle 1.8s
    y = y[start:end]

    mfccs = librosa.feature.mfcc(y=y, sr = sr, n_mfcc = 13)

    fig, ax = plt.subplots(figsize=(15, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set(title='MFCC coefficients')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(out_name + '.png')
