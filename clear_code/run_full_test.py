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


def run(settings_map, source_data, target_data, target_test_data, target_kshot_data, source_kshot_data,
        target_test_fake_norm_data, target_kshot_fake_norm_data, distribution):

    print("training (or collecting) CNN")
    # Train CNN where source data is the training data, and target data is the test data: NOTE - also stores CNN
    # in a format that will make it easy to export for in-the-clear use, and MP-SPDZ use
    cnn_acc_res = utils.train(settings_map, source_data, target_test_data)

    print("personalizing model")
    # personalize model/classify
    pers_result = utils.personalize_classifier(settings_map, source_kshot_data, target_test_data, target_kshot_data)

    print("cnn acc: " + str(cnn_acc_res))
    print("pers acc: " + str(pers_result))

    print("storing ITC results")
    # Stores our in-the-clear results
    utils.store_itc_results(settings_map, cnn_acc_res, pers_result)

    if settings_map["run_spdz"].lower() == "true":

        print("storing params in MP-SPDZ files")
        # store local params in private files
        utils.store_secure_params(settings_map, source_kshot_data, target_kshot_data, target_test_data,
                                  target_test_fake_norm_data, target_kshot_fake_norm_data, distribution)

        print("distributing metadata")
        # send online params (custom networking)
        metadata = utils.distribute_Data(settings_map)

        print("editing secure code")
        print(metadata)
        # prep MP-SPDZ code
        utils.edit_source_code(settings_map, metadata, target_test_data, run_personalizor="true")

        print("transferring files to MP-SPDZ library")
        # Write our secure mpc files to the MP-SPDZ library
        utils.populate_spdz_files(settings_map)

        if settings_map["compile"].lower() == "true":
            print("compiling secure code")
            # compile MP-SPDZ code
            utils.compile_spdz(settings_map)

        print("running secure code... This may take a while")
        # run MP-SPDZ code
        utils.run_mpSPDZ(settings_map, run_program="run")

        # For convenience, we just have party 0 store this information
        if settings_map["party"] == "0":
            print("validating results")
            # validate results
            utils.validate_results(settings_map)

            print("Determining the accuracy of the MP-SPDZ protocol")
            # Take the predicted labels of the spdz protocol and compare them against the ground truth
            mpc_accuracy = utils.compute_spdz_accuracy(settings_map, target_test_data)

            print("Saving MP-SPDZ accuracy results")
            # Save spdz accuracy results
            utils.store_mpc_results(settings_map, mpc_accuracy)

            utils.write_stats(settings_map, mpc_accuracy, cnn_acc_res, pers_result, len(target_test_data[0]))
