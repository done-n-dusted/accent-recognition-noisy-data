'''
    CSE583 PRML Term Project
    Few Utils functions
    
    Author: Anurag Pendyala
    PSU Email: app5997@psu.edu
    Description: File containing functions to add noise to audio files
    
'''

import augly.audio as audaugs
import os
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import numpy as np

def add_noise(audio_file, noise_path, out_name, noise_level):
    '''
        Function that adds noise for a particular noise level for an audio file
    '''
    audio, sample_rate = sf.read(audio_file)
    audio = np.array(audio)
    aug_audio, _ = audaugs.add_background_noise(audio, background_audio=noise_path, snr_level_db=noise_level)
    aug_audio = np.mean(aug_audio, axis = 0)
    sf.write(out_name, aug_audio, sample_rate)