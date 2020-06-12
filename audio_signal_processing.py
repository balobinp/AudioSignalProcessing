# The file contains supporting functionsfor the project

from os.path import join, isdir, isfile
import os
import shutil
from tqdm.auto import tqdm
import librosa
import json
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from scipy.io import wavfile
import soundcard as sc
import python_speech_features
import tensorflow as tf
from sklearn.model_selection import train_test_split
import IPython.display as ipd


def add_noice(sample, sr=8000, noice_vol=0.05):
    '''Add noice by adding a random values to the original measures.
    Inputs:
        sample (numpy.ndarray): wav file array;
        sr=8000 (int): incoming signal sample rate;
        noice_vol=0.05 (float): the noice coefficient.
    Return:
        numpy.ndarray: reprocessed sample wav file array.
    Usage:
asp.sample_filtered = add_noice(sample, sr=8000, noice_vol=0.05)'''
    min_noice = min(sample) * noice_vol
    max_noice = max(sample) * noice_vol
    sample_noice = np.random.uniform(min_noice, max_noice, sr)
    return sample + sample_noice


def remove_silence(sample, sr=8000, threshold=0.05):
    '''Remove values less then defined threshold. Threshold is defined between 0 and 1.
    Moving average is used to prepare the sample for analysis.
    Inputs:
        sample (numpy.ndarray): wav file array;
        sr=8000 (int): incoming signal sample rate;
        threshold=0.05 (float): threshold to remove silence.
    Return:
        numpy.ndarray: reprocessed sample wav file array;
        int: samples in the reprocessed sample wav file array.
    Usage:
target_sr = 8000
sample_filtered, sr_filtered = asp.remove_silence(sample, sr=target_sr, threshold=0.25)
sample_filtered = librosa.resample(sample_filtered, orig_sr=sr_filtered, target_sr=target_sr)[:target_sr]'''
    mask = []
    sample_abs = pd.Series(sample).apply(np.abs)
    sample_mean = sample_abs.rolling(window=int(sr/10), min_periods=1, center=True).mean()
    threshold = sample_mean.max() * threshold
    for mean in sample_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return sample[mask], sum(mask)


def spoil_audio(sample, sr=8000, rem_coef=0.1):
    """Spoil the wav audio file by removing measures.
    Inputs:
        sample (numpy.ndarray): wav file array;
        sr=8000 (int): incoming sample rate;
        rem_coef=0.1 (float): removing coefficient
    Return:
        numpy.ndarray: reprocessed sample wav file array"""

    new_sample_rate = sr + sr * rem_coef
    sample_filtered = librosa.resample(sample, orig_sr=sr, target_sr=new_sample_rate)

    # Получить Series из wav numpy.ndarray
    s1 = Series(sample_filtered)

    # Выбрать случайно индексы
    rand_ind = np.random.choice(s1.index.values, sr, replace=False)

    # Выбрать из исходного Series только полученные случайные индексы
    s1.loc[rand_ind].sort_index().values

    return s1.loc[rand_ind].sort_index().values


def two_samples_analyzer(first_sample, second_sample, sr=8000):
    """Draw plots and show parameters for two wav audio samples.
    Inputs:
        first_sample (numpy.ndarray): wav file array;
        second_sample(numpy.ndarray): wav file array;
        sr=8000 (int): the sample rate of both samples;
    Return: None"""

    for sample in [first_sample, second_sample]:
        print(f"""Shape: {sample.shape}""")
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Raw wave of the sample')
        ax1.set_xlabel('time, sec.')
        ax1.set_ylabel('Amplitude')
        ax1.plot(sample)
        plt.show()
        display(ipd.Audio(sample, rate=sr))


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, sample_length=22050):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :param sample_length (int): Audio sample length
    :return:
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {"mapping": [], "labels": [], "MFCCs": [], "files": []}

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we're at sub-folder level
        if dirpath is not dataset_path:
            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("\\")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))
            # process all audio files in sub-dir and store MFCCs
            for file in tqdm(filenames):
                file_path = join(dirpath, file)
                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)
                # drop audio files with less than pre-decided number of samples
                if len(signal) >= sample_length:
                    # ensure consistency of the length of the signal
                    signal = signal[:sample_length]
                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                 hop_length=hop_length)
                    # store data for analysed track
                    data["MFCCs"].append(MFCCs.T[..., np.newaxis].tolist())
                    data["labels"].append(i - 1)
                    data["files"].append(file_path)

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    :return mapping (list): Targets mapping
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    mapping = data['mapping']
    return X, y, mapping


def gen_feat(classes=5, samples_in_class=100, dim=2):
    """Generates the test dataset as 1 second mfcc audio feature.
    Input
        classes (int): the number of the classes to generate.
        samples_in_class (int): samples in each class.
        dim (int): dimensions in the output mfcc data.
            1 dimention for MLP
            2 dimentions for 1D convolutional NN
            3 dimentions for 2D convolutional NN
    return
        X (np.ndarray): like 1 second mfcc audio feature
        y (np.ndarray): classes
    """
    X = []
    y = []
    if dim == 3:
        for class_ind in range(classes):
            for _ in range(samples_in_class):
                X.append(np.array([np.random.normal(loc=class_ind, scale=2, size=13) for _ in range(44)]))
                y.append(class_ind)
        return np.array(X)[..., np.newaxis], np.array(y)
    elif dim == 2:
        for class_ind in range(classes):
            for _ in range(samples_in_class):
                X.append(np.random.normal(loc=class_ind, scale=2, size=8000))
                y.append(class_ind)
        return np.array(X)[..., np.newaxis], np.array(y)
    elif dim == 1:
        for class_ind in range(classes):
            for _ in range(samples_in_class):
                X.append(np.random.normal(loc=class_ind, scale=2, size=8000))
                y.append(class_ind)
        return np.array(X), np.array(y)


def prepare_dataset(X, y, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """
    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model_conv2d(input_shape, output_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param output_shape (int): The number of classes
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """
    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.5)

    # softmax output layer
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def build_model_mlp(learning_rate=0.0001):
    """Build neural network using keras.
    :return model: TensorFlow model
    """
    # build model with 3 layers: 8000 -> 5 -> 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, input_dim=8000, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # choose optimiser
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # compilie model
    model.compile(optimizer=optimizer, loss='mse', metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)
    # train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """
    fig, axs = plt.subplots(2)
    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")
    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")
    plt.show()