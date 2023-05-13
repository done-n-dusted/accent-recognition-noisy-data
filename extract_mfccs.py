'''
    CSE583 PRML Term Project
    Code to convert audio files in folders to npy files
    
    Author: Anurag Pendyala
    PSU Email: app5997@psu.edu
    Description: File containing code to test a particular model.
    Run the programming in the following way:
    python extract_mfccs.py path/to/data.csv path/for/npy.npy
'''

import pandas as pd
import numpy as np
import librosa
import sys
from tqdm import tqdm

def extract_mfccs(fpath):
    audio, sr = librosa.load(fpath)
    mfccs = librosa.feature.mfcc(audio, sr = sr, n_mfcc = 13)
    return np.transpose(mfccs)

def create_npy(csv_path, max_len = 1.8):

    df = pd.read_csv(csv_path)
    fpaths = df['location']
    labels = df['label']
    res = {'labels': labels}

    data = []

    print("Loading files...")
    for fpath in tqdm(fpaths):
        audio, sr = librosa.load(fpath)
        n = len(audio)
        k = int(sr*max_len)
        start = (n - k) // 2
        end = start + k
        # print(list(audio), sr)
        audio = audio[start:end]
        mfccs = librosa.feature.mfcc(y=audio, sr = sr, n_mfcc = 13)
        # mfccs = np.transpose(mfccs)
        data.append(mfccs)
    
    res['features'] = data

    return res

def main():
    if len(sys.argv) != 3:
        print("Retry.")
        return
    
    csv_name, npy_name = sys.argv[1:]
    feature_dict = create_npy(csv_name)
    np.save(npy_name + '.npy', feature_dict)

main()

    
