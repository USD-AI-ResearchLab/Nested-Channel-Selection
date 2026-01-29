# !/usr/bin/env python
# coding: utf-8

#from modSpec import create_mod_spectrogram
import matplotlib.ticker as ticker
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.constraints import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import sklearn
from itertools import cycle
import ast
import time
import os
import warnings
import copy
import gc
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, StratifiedGroupKFold
from multiprocessing import Process, Manager
from collections import defaultdict
import multiprocessing as mp
from trainUt import *
import math
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import cv2

reg = None

#code found online for reinitializing the weights of a model in a method
def reset_weights(model):
  for layer in model.layers:
    if isinstance(layer, tensorflow.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))

init =  HeNormal(seed=0)

def cnn(input_shape,
        dropout_rate=0.20,
        dense_units=256,
        filters=[64, 64, 128, 128, 128, 256],
        kernel_sizes=[(7,7), (5,5), (3,3), (3,3), (3,3), (3,3)],
        reg=None,
        final_activation="relu"):
    """
    CNN model with tunable hyperparameters.
    - dropout_rate: applied to Dropout/SpatialDropout2D layers
    - dense_units: size of the dense layer before output
    - filters: list of filter counts for each Conv2D block
    - kernel_sizes: list of kernel sizes for each Conv2D block
    - reg: kernel_regularizer (e.g. l2)
    - final_activation: 'relu' (default) or 'sigmoid'
    """

    X_input = Input(input_shape)

    X = GaussianNoise(0.025)(X_input)
    X = RandomFlip(mode="horizontal")(X)

    # First conv block
    X = Conv2D(filters[0], kernel_sizes[0], strides=(1,1),
               padding="same", kernel_regularizer=reg)(X)
    X = BatchNormalization()(X)
    X = SpatialDropout2D(dropout_rate, data_format='channels_last')(X)
    X = Activation('relu')(X)

    # Second conv block
    X = Conv2D(filters[1], kernel_sizes[1], strides=(1,1),
               padding="same", kernel_regularizer=reg)(X)
    X = BatchNormalization()(X)
    X = SpatialDropout2D(dropout_rate, data_format='channels_last')(X)
    X = MaxPooling2D((2,4), strides=(2,4), padding="same")(X)
    X = Activation('relu')(X)

    # Remaining conv blocks except the last
    for f, k in zip(filters[2:-1], kernel_sizes[2:-1]):
        X = Conv2D(f, k, padding="same", kernel_regularizer=reg)(X)
        X = BatchNormalization()(X)
        X = Dropout(rate=dropout_rate)(X)
        X = MaxPooling2D((2,2), strides=(2,2), padding="same")(X)
        X = Activation('relu')(X)

    # Final conv block
    X = Conv2D(filters[-1], kernel_sizes[-1], padding="same",
               kernel_regularizer=reg, name="conv_last")(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=dropout_rate)(X)

    if final_activation == "sigmoid":
        X = Activation('sigmoid')(X)
    else:
        X = Activation('relu')(X)

    X = Flatten()(X)

    # Dense block (always present)
    X = Dense(dense_units)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=dropout_rate)(X)
    if final_activation == "sigmoid":
        X = Activation('sigmoid')(X)
    else:
        X = Activation('relu')(X)

    # Output
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X, name='cnn')
    return model

def runChan(L, params, train = False):

    physical_devices = tensorflow.config.list_physical_devices('gpu') 
    for gpu_instance in physical_devices: 
        tensorflow.config.experimental.set_memory_growth(gpu_instance, True)

    resultsloss = L[0]
    resultsacc  = L[1]
    X_train     = L[2]   # not used in ablation, but still passed
    y_train     = L[3]
    X_test      = L[4]
    y_test      = L[5]
    count       = L[6]
    resultsauc  = L[7]
    resultsrec  = L[8]
    cfg_id      = L[13]   # you’ll control this in the outer loop
    innerCount  = L[14]
    if train:
        train_patients = L[19]
        test_patients = L[20]
        train_channels = L[21]
        test_channels = L[22]
        X_train = L[15]
        y_train = L[16]
        X_test = L[17]
        y_test = L[18]
    else:
        train_patients = L[9]
        test_patients = L[10]
        train_channels = L[11]
        test_channels = L[12]
        X_train = L[2]
        y_train = L[3]
        X_test = L[4]
        y_test = L[5]

    input_shape = (X_test.shape[1], X_test.shape[2], 1)
    model = cnn(
        input_shape,
        dropout_rate=params["dropout"],
        dense_units=params["dense_units"],
        filters=params["filters"],
        kernel_sizes=params["kernel_sizes"],
        reg=reg,
        final_activation=params["final_activation"]
    )

    if train:
        # loading the best checkpoint for the train/validation
        model.load_weights('./checks/eegCNNGeneralChan' + str(cfg_id) + str(count) + str(innerCount) +  ".weights.h5")
    else:
        # loading the best checkpoint for the train/validation
        model.load_weights('./checks/eegCNNGeneralChan' + str(cfg_id) + str(count) + ".weights.h5")

    # get raw predictions
    y_pred_probs = model.predict(X_test, batch_size=1).flatten()

    # --- group by channel, then by patient ---
    chan_patient_probs = {}
    chan_patient_true  = {}

    for ch, pid, prob, true in zip(test_channels, test_patients, y_pred_probs, y_test):
        if ch not in chan_patient_probs:
            chan_patient_probs[ch] = {}
            chan_patient_true[ch]  = {}
        if pid not in chan_patient_probs[ch]:
            chan_patient_probs[ch][pid] = []
            chan_patient_true[ch][pid]  = true
        chan_patient_probs[ch][pid].append(prob)

    # --- aggregate per channel ---
    channel_metrics = {"Loss": {}, "Acc": {}, "AUC": {}, "Rec": {}}

    for ch in chan_patient_probs:
        patient_preds = []
        patient_true  = []
        patient_probs = []

        for pid in chan_patient_probs[ch]:
            probs = np.array(chan_patient_probs[ch][pid])
            true  = chan_patient_true[ch][pid]

            # average probability for this patient on this channel
            avg_prob = probs.mean()
            pred = 1 if avg_prob >= 0.5 else 0

            patient_preds.append(pred)
            patient_true.append(true)
            patient_probs.append(avg_prob)

        patient_preds = np.array(patient_preds)
        patient_true  = np.array(patient_true)
        patient_probs = np.array(patient_probs)
        


        # metrics for this channel (across patients)
        acc  = np.mean(patient_preds == patient_true)

        # NOTE: using hard predictions for loss/AUC, not probabilities
        loss = tensorflow.keras.losses.binary_crossentropy(
            patient_true.astype("float32"),
            patient_probs.astype("float32")
        ).numpy().mean()
        auc = tensorflow.keras.metrics.AUC()(patient_true, patient_probs).numpy()
        rec  = tensorflow.keras.metrics.Recall()(patient_true, patient_preds).numpy()


        channel_metrics["Loss"][ch] = float(loss)
        channel_metrics["Acc"][ch]  = float(acc)
        channel_metrics["AUC"][ch]  = float(auc)
        channel_metrics["Rec"][ch]  = float(rec)

    L[23] = channel_metrics

    return

def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            # Binary sigmoid model
            class_channel = predictions[:, 0]
        else:
            # Softmax model
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]


    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = heatmap / (tf.reduce_max(tf.abs(heatmap)) + 1e-8)
    return heatmap.numpy()

def save_gradcam(img, heatmap, out_path, alpha=0.4):
    plt.figure(figsize=(4,4))
    plt.imshow(img.squeeze(), cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def runEval(L, params):
    # GPU setup
    physical_devices = tensorflow.config.list_physical_devices('gpu')
    for gpu_instance in physical_devices:
        tensorflow.config.experimental.set_memory_growth(gpu_instance, True)

    count   = L[6]
    cfg_id  = L[13]
    X_test  = L[4]
    y_test  = L[5]
    test_patients  = L[10]
    test_channels  = L[12]
    good_channels  = L[28]
    nChan = L[31]

    # clear session & build model
    tensorflow.keras.backend.clear_session()
    input_shape = (X_test.shape[1], X_test.shape[2], 1)
    model = cnn(input_shape, dropout_rate=params["dropout"], dense_units=params["dense_units"], filters=params["filters"], kernel_sizes=params["kernel_sizes"], reg=reg, final_activation=params["final_activation"])

    # load weights
    weight_path = f'./checks/eegCNNGeneralChan{cfg_id}{count}.weights.h5'
    model.load_weights(weight_path)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------
    y_pred_probs = model.predict(X_test, batch_size=1).flatten()

    good_channels = [int(c) for c in good_channels]  # ranked preference list

    gradcam_dir = "./results/gradcams_goodChan"
    os.makedirs(gradcam_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Collect entries for TRUE‑POSITIVE patients only
    # ------------------------------------------------------------
    patient_entries = defaultdict(list)

    for idx, (pid, ch, prob, y) in enumerate(
        zip(test_patients, test_channels, y_pred_probs, y_test)
    ):
        if y != 1:
            continue  # patient-level positivity only

        patient_entries[pid].append((idx, int(ch), prob))

    # ------------------------------------------------------------
    # Select EXACTLY nChan channels per patient
    # ------------------------------------------------------------
    viz_entries = []

    for pid, entries in patient_entries.items():
        patient_channels = {ch for _, ch, _ in entries}

        # Top-nChan globally preferred channels this patient actually has
        selected_channels = [
            ch for ch in good_channels
            if ch in patient_channels
        ][:nChan]

        # Enforce exactly nChan
        if len(selected_channels) < nChan:
            continue

        selected_channels = set(selected_channels)

        for idx, ch, prob in entries:
            if ch in selected_channels:
                viz_entries.append((pid, ch, idx, prob))

    # ------------------------------------------------------------
    # Visualization setup
    # ------------------------------------------------------------
    colors = [(0, "yellow"), (1, "green")]
    custom_cmap = LinearSegmentedColormap.from_list("spec_cmap", colors, N=100)

    last_conv_layer_name = "conv_last"

    # ------------------------------------------------------------
    # Grad‑CAM + Spectrogram Overlay
    # ------------------------------------------------------------
    for i, (pid, ch, idx, prob) in enumerate(viz_entries):
        x = X_test[idx]

        img_array = np.expand_dims(x[..., np.newaxis], axis=0)
        heatmap = make_gradcam_heatmap(
            model, img_array, last_conv_layer_name, 0)

        # Flip so low frequencies are at the bottom
        spec = np.flipud(x.squeeze())
        heatmap_resized = cv2.resize(heatmap, (spec.shape[1], spec.shape[0]))
        heatmap_resized = np.flipud(heatmap_resized)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Spectrogram only
        axes[0].imshow(spec, cmap=custom_cmap, aspect="auto", interpolation="none")
        axes[0].set_title(f"Patient {pid}, Ch {ch}, p={prob:.2f}")
        axes[0].axis("off")

        # Spectrogram + Grad‑CAM
        axes[1].imshow(spec, cmap=custom_cmap, aspect="auto", interpolation="none")
        axes[1].imshow(
            heatmap_resized,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            alpha=0.5,
            aspect="auto",
            interpolation="none",
        )

        axes[1].set_title("Grad‑CAM Overlay")
        axes[1].axis("off")

        out_path = os.path.join(
            gradcam_dir,
            f"patient{pid}_chan{ch}_p{prob:.2f}_idx{i}_overlay.png",
        )

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


    return

if __name__ == "__main__":

    #augment = SpecFrequencyMask(p=1)

    infile = open('./dataframes/eegUNMPDVC.pkl','rb')
    data = pickle.load(infile)
    data['data'] = 1
    infile.close()

    infile = open('./dataframes/eegIowaPubNPDVCPD.pkl','rb')
    data2 = pickle.load(infile)
    data2['data'] = 2
    infile.close()

    infile = open('./dataframes/eegSanDiegoAll.pkl','rb')
    data3 = pickle.load(infile)
    data3['data'] = 3
    infile.close()

    data = pd.concat([data, data2, data3], ignore_index=True)

    data = data.sample(frac = 1, random_state=42)


    dataset = [] # Train Dataset
    vdataset =[]
    #iterating over valid train/validation data
    count = 0
    countn = 0

    count = 0
    # --- control parameters ---
    desired_length = 16384        # signal window length
    window_hop     = desired_length // 2   # sliding hop for long signals
    n_fft          = 256           # STFT FFT size
    stft_hop       = 64            # STFT hop length


    dataset   = []
    patient_id = 0
    count      = 0

    # Build dataset with patient and channel separation
    for row in data.itertuples():
        y, label = row.array, np.array(row.label)
        #if count > 30:
        #    continue
        # Split each channel into windows (pad if short, slide if long)
        for i, channel in enumerate(y):
            if np.all(y[i] == 0) or np.any(np.isnan(y[i])):
                #print(i)
                continue

            if len(channel) <= desired_length:
                padded = np.zeros(desired_length)
                padded[:len(channel)] = channel
                windows = [padded]
            else:
                windows = []
                for start in range(0, len(channel) - desired_length + 1, window_hop):
                    windows.append(channel[start:start+desired_length])
                # Ensure tail coverage if not aligned to hop
                if (len(channel) - desired_length) % window_hop != 0:
                    windows.append(channel[-desired_length:])

            # Spectrogram per window, per channel (kept separate)
            for w in windows:
                stft = np.abs(librosa.stft(y=w, n_fft=n_fft, hop_length=stft_hop))**2
                ms2 = stft / stft.max()
                ms_DB = librosa.power_to_db(S=ms2, ref=0)
                ms_DB = ms_DB - 20
                ms_DB = ms_DB / ms_DB.max()

                # Trim first row/col if desired
                ms_DB = ms_DB[1:, 1:]

                # Dynamically determine target shape
                freq_bins, time_bins = ms_DB.shape
                # For consistency, you can still crop/pad to a fixed size if needed
                #target_freq  = min(128, freq_bins)
                #target_time  = min(256, time_bins)
                #ms_DB = ms_DB[:target_freq, :target_time]

                # Reshape with channel dimension
                ms_DB = np.reshape(ms_DB, (freq_bins, time_bins, 1))

                dataset.append((ms_DB, label, patient_id, i, row.data))

        patient_id += 1
        count += 1


    # Unpack dataset
    data_X, data_y, patient_ids, channel_ids , dataset_ids = zip(*dataset)
    data_X, data_y = np.array(data_X), np.array(data_y)
    patient_ids, channel_ids, dataset_ids = np.array(patient_ids), np.array(channel_ids), np.array(dataset_ids)
    unique_datasets = np.unique(dataset_ids)

    # Define hyperparameter grid
    param_grid = [
        {"lr":0.01,"decay_rate":0.5,"batch_size":32,"dropout":0.15,"dense_units":128,"filters":[48,48,96,96,128,128],"kernel_sizes":[(7,7),(5,5),(7,7),(5,5),(3,3),(3,3)],"final_activation":"sigmoid"},
    ]

    df = pd.read_csv("./results/eegCNN_foldsCV.csv") 
    best_row = df.loc[df["GCAcc"].idxmax()]
    best_fold = int(best_row["Fold"])

    all_results = []       # averages per config
    all_folds   = []       # every fold’s raw metrics

    manager = Manager()

    for cfg_id, params in enumerate(param_grid, start=1):

        # your logic here


        print(f"Running config {cfg_id}: {params}")

        # shared list for this config
        lst = manager.list()
        lst.append(manager.list())   # 0 resultsloss
        lst.append(manager.list())   # 1 resultsacc
        lst.append([])               # 2 trainX
        lst.append([])               # 3 trainY
        lst.append([])               # 4 testX
        lst.append([])               # 5 testY
        lst.append(0)                # 6 count
        lst.append(manager.list())   # 7 resultsauc
        lst.append(manager.list())   # 8 resultsrec
        lst.append(None)             # 9 train_patients
        lst.append(None)             # 10 test_patients
        lst.append(None)             # 11 train_channels
        lst.append(None)             # 12 test_channels
        lst.append(cfg_id)           # 13 cfg_id
        lst.append(0)                # 14 innerCount
        lst.append([])               # 15 trainloss
        lst.append([])               # 16 trainacc
        lst.append([])               # 17 trainauc
        lst.append([])               # 18 trainrec
        lst.append(None)             # 19 inner_train_patients
        lst.append(None)             # 20 inner_test_patients
        lst.append(None)             # 21 inner_train_channels
        lst.append(None)             # 22 inner_test_channels
        lst.append(manager.dict())   # index 23, for channel_metrics
        lst.append(manager.list())   # 24 goodChanLoss
        lst.append(manager.list())   # 25 goodChanAcc
        lst.append(manager.list())   # 26 goodChanAuc
        lst.append(manager.list())   # 27 goodChanRec
        lst.append([])               # 28 goodChannels
        lst.append(0)                # 29 nGram level
        lst.append(0)                # 30 train combo id
        lst.append(4)                # 31 Int of Number of Channels


        # cross‑validation loop
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (trainIdx, testIdx) in enumerate(skf.split(data_X, data_y, groups=patient_ids)):

        #for fold, test_ds in enumerate(unique_datasets):
            if fold != best_fold:
                continue
            lst[6] = fold
            lst[14] += (lst[6] * 5)
            #testIdx  = np.where(dataset_ids == test_ds)[0]
            #trainIdx = np.where(dataset_ids != test_ds)[0]
            X_train = data_X[trainIdx]
            X_test  = data_X[testIdx]
            y_train = np.array(data_y[trainIdx])
            y_test  = np.array(data_y[testIdx])

            train_patients = patient_ids[trainIdx]
            test_patients  = patient_ids[testIdx]
            train_channels = channel_ids[trainIdx]
            test_channels  = channel_ids[testIdx]

            X_train = np.array([x.reshape((int(n_fft/2), int(desired_length/stft_hop), 1)) for x in X_train])
            X_test  = np.array([x.reshape((int(n_fft/2), int(desired_length/stft_hop), 1)) for x in X_test])

            # update list
            lst[2], lst[3], lst[4], lst[5] = X_train, y_train, X_test, y_test
            lst[9], lst[10], lst[11], lst[12] = train_patients, test_patients, train_channels, test_channels

            chan_scores = defaultdict(list)

            inner_skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            for inner_fold, (inner_trainIdx, inner_valIdx) in enumerate(
                    inner_skf.split(X_train, y_train, groups=train_patients)):
                #if inner_fold < 4:
                #    continue

                # --- slice inner train/val sets ---
                inner_X_train = X_train[inner_trainIdx]
                inner_y_train = y_train[inner_trainIdx]
                inner_X_val   = X_train[inner_valIdx]
                inner_y_val   = y_train[inner_valIdx]

                inner_train_patients = train_patients[inner_trainIdx]
                inner_val_patients   = train_patients[inner_valIdx]
                inner_train_channels = train_channels[inner_trainIdx]
                inner_val_channels   = train_channels[inner_valIdx]

                # --- reshape into CNN input format ---
                inner_X_train = np.array([
                    x.reshape((int(n_fft/2), int(desired_length/stft_hop), 1))
                    for x in inner_X_train
                ])
                inner_X_val = np.array([
                    x.reshape((int(n_fft/2), int(desired_length/stft_hop), 1))
                    for x in inner_X_val
                ])

                # --- put into shared memory slots for inner loop ---
                lst[15] = inner_X_train
                lst[16] = inner_y_train
                lst[17] = inner_X_val
                lst[18] = inner_y_val
                lst[19] = inner_train_patients
                lst[20] = inner_val_patients
                lst[21] = inner_train_channels
                lst[22] = inner_val_channels

                # run in a separate process
                #p = Process(target=runTrain, args=(lst, params, True))
                #p.start()
                #p.join()

                # --- run channel analysis in return mode ---
                p = Process(target=runChan, args=(lst, params, True))
                p.start()
                p.join()

                # collect metrics from slot 23
                metrics = lst[23]   # copy out of manager.dict
                for ch, acc in metrics["Acc"].items():
                    chan_scores[ch].append(acc)

                lst[14] += 1


            # after inner CV, average across folds
            avg_acc = {ch: np.mean(scores) for ch, scores in chan_scores.items()}
            best_channels = sorted(avg_acc, key=avg_acc.get, reverse=True)

            lst[28] = best_channels

            #print(np.unique(train_patients))
            #print(np.unique(test_patients))

            # run in a separate process
            p = Process(target=runEval, args=(lst, params))
            p.start()
            p.join()




    




