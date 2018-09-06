"""
We are reading in the data, normalizing and preprocessing it.
"""

import numpy as np
from scipy.io import loadmat
from glob import glob
from scipy.signal import butter, sosfiltfilt


def discard_irrelevant_measurements(train_x, sfreq):
    """
    We discard the first 0.5 seconds of each trial as it does not contain any information.
    :param train_x: The training data of a single subject.
    :param sfreq: The sampling frequency.
    :return train_x: The training data of a single subject without the measurements at the start.
    """
    tmin = 0
    tmax = 1
    tmin_original = -0.5

    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    train_x = train_x[:, :, beginning:end]

    return train_x


def normalize_data(train_x):
    """
    We normalize the data.
    :param train_x: The training data of a single subject.
    :return train_x: The standardized training data of a single subject.
    """

    #  For each trial, we subtract the mean of the given trial.
    for current_trial in range(0, train_x.shape[0]):
        current_median = np.nanmedian(train_x[current_trial, :, :])
        train_x[current_trial, :, :] -= current_median

    #  We standardize the training data.
    train_x_vector = np.reshape(train_x, [train_x.shape[0] * train_x.shape[1] * train_x.shape[2], 1]).flatten()
    current_std = np.nanstd(train_x_vector)

    for current_trial in range(0, train_x.shape[0]):
        train_x[current_trial, :, :] /= current_std

    return train_x


def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        #  b, a = butter(order, [low, high], btype='band')

        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        sos = butter_bandpass(lowcut, highcut, fs, order)
        #  y = filtfilt(b, a, data)
        y = sosfiltfilt(sos, data)
        return y

def apply_filter(train_x):
    '''
    We are applying a butterworth filter.
    :param train_x: The training data of a single subject.
    :return train_x_filtered: Filtered training data of a single subject.
    '''

    #  We are applying the filter on each sensor of each trial.
    data_points_to_interpolate = 250

    lowest_frequency = 1
    highest_frequency = 100
    butterworth_filter = 6

    filtered_data = train_x.copy()

    for current_trial_nr in range(0, train_x.shape[0]):

        filtered_trial = butter_bandpass_filter(train_x[current_trial_nr, :, :],
                                                       lowest_frequency, highest_frequency,
                                                       data_points_to_interpolate, butterworth_filter)

        filtered_data[current_trial_nr, :, :] = filtered_trial

    print('Filters applied')
    return filtered_data



def preprocess_data():
    '''
    We are loading the dataset and we are preprocessing it. Datafiles are huge, therefore I did not
    split the loading and preprocessing into separate function.
    :param glob_data: Directory of training files.
    :return preprocessed_data: Dictionary holding the training and testing data.
    '''

    data_path = '../data/*.mat'

    preprocessed_data = dict()
    for file_nr, glob_file in enumerate(glob(data_path)):

        #  We are reading in the data from the mat file.
        data = loadmat(glob_file, squeeze_me=True)
        train_x = data['X']
        sfreq = data['sfreq']
        yy = data['y']

        #  We delete the first 0.5 seconds from the trial as the visual stimuli are only presented 0.5 seconds
        #  into the trial.
        train_x = discard_irrelevant_measurements(train_x, sfreq)
        train_x = normalize_data(train_x)

        #  We bandpass filter the data as we are only interested in a certain range of frequencies.
        train_x_filtered = apply_filter(train_x)

        #  We are saving the data of all subjects.
        preprocessed_data['subject' + str(file_nr) + '_train_x'] = train_x
        preprocessed_data['subject' + str(file_nr) + '_train_x_filtered'] = train_x_filtered

        preprocessed_data['subject' + str(file_nr) + '_train_y'] = yy
        print('Subject' + str(file_nr + 1) + ' preprocessed.')
        print(' ')

        np.save('../data/subject' + str(file_nr) + '_train_x_filtered.npy', preprocessed_data['subject' + str(file_nr) + '_train_x_filtered'])
        np.save('../data/subject' + str(file_nr) + '_train_y.npy', preprocessed_data['subject' + str(file_nr) + '_train_y'])

    return preprocessed_data


if __name__ == '__main__':
    preprocess_data()
