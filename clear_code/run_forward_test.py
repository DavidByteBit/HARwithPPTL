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
from . import utils

from .networking import client, server


def run(settings_map, source_data, target_data, target_test_data, target_kshot_data, source_kshot_data):
    print("training (or collecting) CNN")
    # Train CNN where source data is the training data, and target data is the test data: NOTE - also stores CNN
    # in a format that will make it easy to export for in-the-clear use, and MP-SPDZ use
    cnn_acc_res = utils._train(settings_map, source_data, target_test_data)

    print("cnn acc: " + str(cnn_acc_res))

    if settings_map["run_spdz"].lower() == "true":

        if settings_map["test_range"].lower() != "none":
            test_range = settings_map["test_range"].replace("(", "").replace(")", "") \
                .replace("[", "").replace("]", "").split(",")

            lower_bound = int(test_range[0])
            upper_bound = int(test_range[1])

            target_test_data[0] = target_test_data[0][lower_bound:upper_bound]
            target_test_data[1] = target_test_data[1][lower_bound:upper_bound]

        print("storing params in MP-SPDZ files")
        # store local params in private files
        utils._store_secure_params(settings_map, source_kshot_data, target_kshot_data, target_test_data)

        print("distributing metadata")
        # send online params (custom networking)
        metadata = utils._distribute_Data(settings_map)

        print("editing secure code")
        print(metadata)
        # prep MP-SPDZ code
        utils._edit_source_code(settings_map, metadata, target_test_data, run_personalizor="false")

        print("transferring files to MP-SPDZ library")
        # Write our secure mpc files to the MP-SPDZ library
        utils._populate_spdz_files(settings_map)

        print("compiling secure code")
        # compile MP-SPDZ code
        utils._compile_spdz(settings_map, compile_program="test_forwarding")

        print("running secure code... This may take a while")
        # run MP-SPDZ code
        utils._run_mpSPDZ(settings_map, run_program="test_forwarding")

        # For convenience, we just have party 0 store this information
        if settings_map["party"] == "0":
            print("validating results")
            # validate results
            utils._validate_results(settings_map)

            print("Determining the accuracy of the MP-SPDZ protocol")
            # Take the predicted labels of the spdz protocol and comapre them against the ground truth
            mpc_accuracy = utils._compute_spdz_accuracy(settings_map, target_test_data)

            print("Saving MP-SPDZ accuracy results")
            # Save spdz accuracy results
            utils._store_mpc_results(settings_map, mpc_accuracy)


