import yaml
import subprocess
import sys
import os
import time
import random
import json
import numpy as np
from .CNN.run_cnn import train_CNN
from .CNN.nets import nets
from .data_formatting import spdz_format_cnn
from clear_code import pers
from keras.models import Model

from .networking import client, server
from .run_full_test import run as run_full
from .run_forward_test import run as run_forward

###############################################
## IMPORTANT - Must update when adding new tests
run_types = {"run_full_tests": run_full, "run_forward_test": run_forward}


## IMPORTANT - Must update when adding new tests
###############################################


def run(setting_map_path):
    print("parsing settings map")
    # retrieves the settings of the yaml file the user passed in
    settings_map = _parse_settings(setting_map_path)

    print("loading data")
    # load data from source. Source data is separated by participants
    source_data, target_data = _load_data(settings_map)

    print("pre-processing data")
    # normalize and window the data (no longer separated by participants)
    source_data, target_data, fake_norm_target_data, distribution = \
        _pre_process_data(settings_map, source_data, target_data)

    print("splitting data of target user into known/unknown subsets for our k-shot classifier")
    # randomly select 'k' instances from the target data for our k-shot classifier
    collect_subset = False
    if settings_map["test_subset_size"].lower() != "none":
        collect_subset = True

    target_test_data, target_kshot_data, history = _partition_data(settings_map, target_data,
                                                                   collect_subset=collect_subset)

    target_test_fake_norm_data, target_kshot_fake_norm_data, _ = _partition_data(settings_map, target_data,
                                                                                 collect_subset=collect_subset,
                                                                                 holdout_indices_history=history)

    _, source_kshot_data, _ = _partition_data(settings_map, source_data)

    # Run Correct Tests
    run_types[settings_map["run_type"]](settings_map, source_data, target_data,
                                        target_test_data, target_kshot_data, source_kshot_data,
                                        target_test_fake_norm_data, target_kshot_fake_norm_data, distribution)


def _partition_data(settings_map, data, collect_subset=False, holdout_indices_history=None):
    has_history = True
    if holdout_indices_history is None:
        has_history = False
        holdout_indices_history = {}
    features = data[0]
    labels = data[1]

    kshot = int(settings_map["kshot"])

    # Since data is already ohe'd, it is more efficient just to use a dict where the keys are ohe'd vectors
    data_sorted_by_label = {}

    rows_of_data = len(data[0])

    testing_features = []
    testing_labels = []
    kshot_features = []
    kshot_labels = []

    for i in range(rows_of_data):
        sample = features[i]
        # tuple makes it hashable
        ohe_label = tuple(labels[i].tolist())
        if ohe_label not in data_sorted_by_label.keys():
            data_sorted_by_label[ohe_label] = []
        data_sorted_by_label[ohe_label].append(sample)

    for key in data_sorted_by_label.keys():
        rows_of_subset = len(data_sorted_by_label[key])

        # In this context, the holdout refers to the values that should be saved for our k-shot classifier
        holdout_indices = None

        if has_history:
            holdout_indices = holdout_indices_history[key]
        else:
            holdout_indices = np.random.choice(rows_of_subset, size=kshot, replace=False)

        holdout_indices_history[key] = holdout_indices
        remaining_indices = [i for i in range(rows_of_subset) if i not in holdout_indices]
        random.shuffle(remaining_indices)

        # If we are only to collect a subset of data, just grab first n% of values
        if collect_subset:
            remaining_indices = \
                remaining_indices[:int(float(settings_map["test_subset_size"]) * len(remaining_indices))]

        features_of_subset = np.array(data_sorted_by_label[key])

        key_as_np = np.array(list(key))

        # Note to self, probably not the cleanest way of doing this, but it works
        testing_labels_subset = np.array([key_as_np for i in remaining_indices])
        kshot_labels_subset = np.array([key_as_np for i in holdout_indices])

        kshot_features.extend(features_of_subset[np.array(holdout_indices)])
        kshot_labels.extend(kshot_labels_subset)
        testing_features.extend(features_of_subset[np.array(remaining_indices)])
        testing_labels.extend(testing_labels_subset)

    testing_features = np.array(testing_features)
    testing_labels = np.array(testing_labels)
    kshot_features = np.array(kshot_features)
    kshot_labels = np.array(kshot_labels)

    # print(testing_features.shape)
    # print(testing_labels.shape)
    # print(kshot_features.shape)
    # print(kshot_labels.shape)

    test = []
    holdout = []

    test.append(testing_features)
    test.append(testing_labels)
    holdout.append(kshot_features)
    holdout.append(kshot_labels)

    return test, holdout, holdout_indices_history


def __ohe(labels):
    # Assumption: labels range from 0 to n (sequentially), resulting in n + 1 total labels
    different_labels = max(labels).astype(int) + 1
    return np.array([[int(i == num.astype(int)) for i in range(different_labels)] for num in labels])


def _pre_process_data(settings_map, source_data, target_data):
    source_data_norm, source_labels, mean, std = None, None, None, None
    target_data_norm, target_labels = None, None
    fake_target_data_norm, fake_target_labels = None, None
    if settings_map["normalize"].lower() == "self":
        print("normalizing according to self")
        source_data_norm, source_labels, _, _ = __normalize(source_data)
        target_data_norm, target_labels, _, _ = __normalize([target_data])
    elif settings_map["normalize"].lower() == "source":
        print("normalizing according to source")
        source_data_norm, source_labels, mean, std = __normalize(source_data)
        target_data_norm, target_labels, _, _ = __normalize([target_data], mean, std)
        fake_target_data_norm, fake_target_labels, _, _ = __normalize([target_data],
                                                                      mean=[0] * len(mean), std=[1] * len(std))
    else:
        raise Exception('Invalid normalize param')

    return __window_data(settings_map, source_data_norm, source_labels), \
           __window_data(settings_map, target_data_norm, target_labels), \
           __window_data(settings_map, fake_target_data_norm, fake_target_labels), \
           (mean, std)


def __window_data(settings_map, data, labels):
    if data is None:
        return [None, None]

    window_size = int(settings_map["time_slice_len"])

    windowed_data = [[], []]

    row_quantity = len(data)

    for i in range(row_quantity // window_size):
        start_index = i * window_size
        end_index = (i + 1) * window_size

        # Otherwise we'd be out of bounds
        if end_index < row_quantity:
            start_label = labels[start_index]
            end_label = labels[end_index]
            # If the labels match, then there is enough data to make a full window of data
            if start_label == end_label:
                windowed_data[0].append(data[start_index:end_index])
                windowed_data[1].append(start_label)

    windowed_data[0] = np.array(windowed_data[0])
    # one hot encodes each label, making this vector into a matrix
    windowed_data[1] = __ohe(windowed_data[1])

    return windowed_data


def __normalize(data, mean=None, std=None):
    """
    :param data: Data from participants.
    :param mean: mean used to calculate norm. If None, calculates mean
    :param std: std used to calculate norm. If None, calculates std
    :return: Normalized data stacked as single matrix, and the labels.
             Also returns the mean and std used to normalize the data.
    """
    stacked_data = data[0]

    for i in range(len(data) - 1):
        stacked_data = np.vstack((stacked_data, data[i + 1]))

    labels = stacked_data[:, -1]  # for last column
    stacked_data = stacked_data[:, :-1]  # for all but last column

    calculate_stats = False
    if mean is None and std is None:
        calculate_stats = True

    if calculate_stats:
        mean = stacked_data.mean(axis=0)
        std = stacked_data.std(axis=0)

    col_wise_stacked_data = stacked_data.T
    col_wise_stacked_data_normalized = []

    for i in range(len(col_wise_stacked_data)):
        col = col_wise_stacked_data[i]
        col_wise_stacked_data_normalized.append((col - mean[i]) / std[i])

    col_wise_stacked_data_normalized = np.array(col_wise_stacked_data_normalized)

    return col_wise_stacked_data_normalized.T, labels, mean, std


def _load_data(settings_map):
    path_to_data = settings_map["path_to_public_data_dir"]
    target_id = int(settings_map["target_id"])

    participants = 0
    participant_files = []

    for filename in os.listdir(path_to_data):
        participants += 1
        p_id = ""
        # id substring needs to be located at the end of the filename for this to work
        for i in range(len(filename)):
            # -5 because we assume files have a .csv extension
            ch = filename[-5 - i]
            if ch.isnumeric():
                p_id = ch + p_id
            else:
                break
        participant_files.append((filename, int(p_id)))

    source_data = []
    target_data = None

    # does this work for tuples?
    for file, p_id in participant_files:
        file_path = path_to_data + "/" + file
        if p_id != target_id:
            source_data.append(np.genfromtxt(file_path, delimiter=','))
        else:
            target_data = np.genfromtxt(file_path, delimiter=',')

    return source_data, target_data


def _parse_settings(setting_map_path):
    settings_map = None

    with open(setting_map_path, 'r') as stream:
        try:
            settings_map = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return settings_map
