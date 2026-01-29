import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import librosa
import mne
import scipy.io

# given a folder full of wav files\n",

ss = 16384
hop = 8192
osr = 500
nsr = 64

def standardize(signal):
    mean = np.mean(signal)
    std_dev = np.std(signal)
    return (signal - mean) / std_dev if std_dev > 0 else signal

def normalize(signal):
    max_abs_value = np.max(np.abs(signal))
    return signal / max_abs_value if max_abs_value > 0 else signal

def standardize_and_normalize(signal):
    # Standardize the signal
    standardized_signal = standardize(signal)
    
    # Normalize the signal
    normalized_signal = normalize(standardized_signal)
    
    return normalized_signal

def apply_highpass_filter(signal, cutoff, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

p = Path(r'./Dataset/UNMDataset/Jim_rest/').glob('**/*')
files = [x for x in p if x.is_file()]

fileName = []
chunks = []
array = []
label = []
pid = []

cont = []
PD = []

positivePID = ['801','802','804','805','806','807','808','809','810','811','813','814','815','816','817','818','819','820','821','822','823','824','825','826','827','828','829']
negativePID = ['894', '908', '906', '903', '7060', '893', '909', '911', '895', '913', '900', '896', '899', '914', '910', '890',
    '891', '912', '905', '904', '892', '902', '901', '898', '897', '7070', '907', '8060', '8070']

# Frontal Region
frontal_channels = [
    'Fp1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1',
    'AFz', 'Fz',
    'F2', 'F4', 'F6', 'F8', 'AF4', 'AF8', 'Fp2'
]


# Fronto-Central Region
fronto_central_channels = [
    'FT7', 'FT9',
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'FT8', 'FT10'
]


# Central Region
central_channels = [
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'
]

# Centro-Parietal Region
centro_parietal_channels = [
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
]

# Parietal Region
parietal_channels = [
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'
]

# Temporal Region
temporal_channels_left = [
    'T7', 'TP7', 'TP9'
]

temporal_channels_right = [
    'T8', 'TP8', 'TP10'
]

# Parieto-Occipital Region
parieto_occipital_channels = [
    'PO7', 'PO3', 'POz', 'PO4', 'PO8'
]


# Occipital Region
occipital_channels = [
    'O1', 'Oz', 'Iz', 'O2'
]


# Combining all channels into a single list
channel_organized = (
    frontal_channels +
    fronto_central_channels +
    central_channels +
    centro_parietal_channels +
    parietal_channels +
    temporal_channels_left +
    temporal_channels_right +
    parieto_occipital_channels +
    occipital_channels
)

for file in files:
    is_present = False
    for positive_id in positivePID:
        if positive_id in str(file) and '8060' not in str(file) and '8070' not in str(file):
            is_present = True
            pid.append(positive_id)
        else:
            for negative_ID in negativePID:
                if negative_ID in str(file):
                    pid.append(negative_ID)

    if (pid[-1] + '_2') in str(file) and is_present:
        print('skip')
        continue

    mat = scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
    EEG = mat['EEG']

    chanlocs = EEG.chanlocs
    eeg_data = EEG.data
    n_samples = eeg_data.shape[1]

    # Build lookup: channel label -> index in EEG.data
    chan_to_idx = {chan.labels: i for i, chan in enumerate(chanlocs)}

    # Initialize canonical-aligned array
    aligned_eeg = np.zeros((len(channel_organized), n_samples), dtype=np.float32)

    # Insert available channels, zero-fill missing
    for i, chan_name in enumerate(channel_organized):
        if chan_name in chan_to_idx:
            src_idx = chan_to_idx[chan_name]
            aligned_eeg[i, :] = eeg_data[src_idx, :]
        # else: leave zeros explicitly

    reordered_eeg_data = aligned_eeg

    resampled = np.array([librosa.resample(channel, orig_sr = osr, target_sr = nsr) for channel in reordered_eeg_data])

    for i in range(len(resampled)):
        resampled[i, :] = standardize_and_normalize(resampled[i, :])

    fileName.append(str(file))
    array.append(resampled)

    if is_present:
        print(f"One of the positive IDs is present in the file name: {str(file)}")
        label.append(1)
    else:
        print(f"No positive IDs found in the file name: {str(file)}")
        label.append(0)

pddf = pd.DataFrame(data={'fileName':fileName, 'array':array, 'label':label})

pickle.dump(pddf, open('./dataframes/eegUNMPDVC.pkl', 'wb'))

