import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import mne

# -----------------------------
# Utility functions
# -----------------------------
def standardize(signal):
    mean = np.mean(signal)
    std_dev = np.std(signal)
    return (signal - mean) / std_dev if std_dev > 0 else signal

def normalize(signal):
    max_abs_value = np.max(np.abs(signal))
    return signal / max_abs_value if max_abs_value > 0 else signal

def standardize_and_normalize(signal):
    return normalize(standardize(signal))

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

# -----------------------------
# Loader for one BDF file
# -----------------------------
def load_and_align_bdf(bdf_file, channel_organized, target_sr=64):
    # Load raw EEG
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
    raw.resample(target_sr)
    data = raw.get_data()
    chanlocs = raw.info['ch_names']
    n_samples = data.shape[1]

    # Initialize aligned matrix
    aligned = np.zeros((len(channel_organized), n_samples))

    # Fill in available channels, zero-fill missing
    for i, chan in enumerate(channel_organized):
        if chan in chanlocs:
            idx = chanlocs.index(chan)
            sig = data[idx, :]
            sig = standardize_and_normalize(sig)
            aligned[i, :] = sig
        else:
            aligned[i, :] = np.zeros(n_samples)

    return aligned

# -----------------------------
# Main processing
# -----------------------------
def main():
    dataset_root = Path("./Dataset/ds002778/")
    participants = pd.read_csv(dataset_root / "participants.tsv", sep="\t")

    files = dataset_root.glob("sub-*/ses-*/eeg/*.bdf")

    arrays, labels, subjects, sessions, fileNames = [], [], [], [], []

    for file in files:
        subj = file.parts[-4]   # e.g. "sub-pd22" or "sub-hc07"
        sess = file.parts[-3]   # e.g. "ses-off" or "ses-hc"

        if sess == 'ses-off':
            continue

        # Label directly from subject name
        if "pd" in subj.lower():
            label = 1
        elif "hc" in subj.lower():
            label = 0
        else:
            raise ValueError(f"Unknown subject type in {subj}")

        arr = load_and_align_bdf(str(file), channel_organized, target_sr=64)

        arrays.append(arr)
        labels.append(label)
        subjects.append(subj)
        sessions.append(sess)
        fileNames.append(str(file))

        print(f"Processed {file} -> shape {arr.shape}, label {label}")


    # Build DataFrame
    df = pd.DataFrame({
        "subject": subjects,
        "session": sessions,
        "fileName": fileNames,
        "array": arrays,
        "label": labels
    })

    # Save pickle
    out_path = Path("./dataframes")
    out_path.mkdir(exist_ok=True)
    pickle.dump(df, open(out_path / "eegSanDiegoAll.pkl", "wb"))

    print(f"\nSaved DataFrame with {len(df)} entries to {out_path/'eegSanDiegoAll.pkl'}")

if __name__ == "__main__":
    main()
