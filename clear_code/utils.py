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


def _pre_process_source_data(settings_map, data):
    features = data[0]
    labels = data[1]

    model = _load_cnn(settings_map, data)

    n_outputs = model.layers[-1].output_shape[-1]
    dense_output = model.layers[-2].output_shape[-1]

    feature_Extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    extracted_features = feature_Extractor(features)

    weight_matrix_intermediate = [[0 for _ in range(dense_output)] for _ in range(n_outputs)]

    for i in range(len(labels)):
        label = int(np.argmax(labels[i]))
        print(int(label))
        reduced_feat = np.array(extracted_features[i] / float(2 * int(settings_map["kshot"]))).tolist()
        for j in range(dense_output):
            weight_matrix_intermediate[int(label)][j] += reduced_feat[j]


    return weight_matrix_intermediate


def store_mpc_results(settings_map, mpc_accuracy):
    class_path = settings_map["path_to_this_repo"] + "/storage/results/mpc/accuracy.save"

    payload = "target: {a}, k: {b}, random_seed: {d}, accuracy: {e}\n".format(
        a=settings_map["target_id"],
        b=settings_map["kshot"],
        d=settings_map["random_seed"],
        e=mpc_accuracy,
    )

    with open(class_path, 'a+') as stream:
        stream.write(payload)


def write_stats(settings_map, mpc_accuracy, cnn_acc_res, pers_result, test_size):
    stats_path = settings_map["stats_output_file"]
    save_file_times = settings_map["path_to_this_repo"] + "/storage/results/mpc/times.save"

    times_array = []
    with open(save_file_times, 'r') as stream:
        for line in stream:
            if "Stopped" in line:
                times_array.append(line.strip())

    valid_times = times_array[-2:]
    for i in range(len(valid_times)):
        new_str = []
        for ch in valid_times[i][::-1]:
            if not (ch.isdigit() or ch == "."):
                break
            new_str.append(ch)
        valid_times[i] = "".join(new_str)[::-1]
    valid_times = [float(el) for el in valid_times]

    print(valid_times)

    total_class_time = valid_times[1] + valid_times[0]
    avg_time = total_class_time / test_size

    results = "{k}-shot, {target}, {cnn_acc}, {clear_pers}, {mpc_pers}, {pers_time}, " \
              "{avg_class_time}, {total_class_time}, {total_time}".format(k=settings_map["kshot"],
                                                                          target=settings_map["target_id"],
                                                                          cnn_acc=cnn_acc_res, clear_pers=pers_result,
                                                                          mpc_pers=mpc_accuracy,
                                                                          pers_time=valid_times[0],
                                                                          avg_class_time=avg_time,
                                                                          total_class_time=total_class_time,
                                                                          total_time=valid_times[1])

    with open(stats_path, 'a+') as stream:
        stream.write(results + "\n")


def compute_spdz_accuracy(settings_map, target_test_data):
    class_path = settings_map["path_to_this_repo"] + "/storage/results/mpc/classifications.save"
    mpc_r = settings_map["path_to_this_repo"] + "/storage/results/mpc/classifications.save"

    classifications = ""

    with open(class_path, 'r') as stream:
        # should be one line
        for line in stream:
            classifications += line

    classifications = json.loads(classifications.replace("\n", "").replace("\'", ""))
    print(len(classifications))
    correct = 0

    for i in range(len(classifications)):
        c = np.argmax(target_test_data[1][i])
        correct += int(int(classifications[i]) == c)

    accuracy = float(correct) / float(len(classifications))

    print(accuracy)

    return accuracy


def validate_results(settings_map):
    tolerance = float(settings_map["validation_threshold"]) / 100.0
    path_to_this_repo = settings_map["path_to_this_repo"]

    import json

    itc_wm_path = path_to_this_repo + "/storage/results/itc/weight_matrix.save"

    itc_wm = ""
    with open(itc_wm_path, "r") as stream:
        for line in stream:
            itc_wm += line.replace("\'", "").replace("\"", "")

    itc_wm = json.loads(itc_wm)

    itc_fp_path = path_to_this_repo + "/storage/results/itc/forward_pass.save"

    itc_fp = ""
    with open(itc_fp_path, "r") as stream:
        for line in stream:
            itc_fp += line.replace("\'", "").replace("\"", "")

    itc_fp = json.loads(itc_fp)

    mpc_fp_path = path_to_this_repo + "/storage/results/mpc/results.save"
    mpc_results = []
    with open(mpc_fp_path, 'r') as stream:
        for line in stream:
            mpc_results.append(line)

    mpc_results = "".join(mpc_results).replace("\n", "").split("@end")

    mpc_wm = str(mpc_results[-1]).replace("\n", "").replace("\'", "")
    mpc_fp = str(mpc_results[:-1]).replace("\n", "").replace("\'", "")

    mpc_wm = json.loads(mpc_wm)
    mpc_fp = json.loads(mpc_fp)

    for i in range(len(itc_fp)):
        valid = _compare_within_range(itc_fp[i], mpc_fp[i], tolerance, base=0.1)
        if not valid:
            print("WARNING, NON-VALID RESULT FOR {a} and tolerance {b}"
                  "\nCorrect result {c}\nMPC result {d}".format(a=i, b=tolerance, c=itc_fp[i], d=mpc_fp[i]))

    for i in range(len(itc_wm)):
        valid = _compare_within_range(itc_wm[i], mpc_wm[i], tolerance, base=0.1)
        if not valid:
            print("WARNING, NON-VALID RESULT FOR {a} and tolerance {b}"
                  "\nCorrect result {c}\nMPC result {d}".format(a=i, b=tolerance, c=itc_wm[i], d=mpc_wm[i]))


def _compare_within_range(a, b, tolerance, base=0.1):
    valid = True

    assert len(a) == len(b)

    for i in range(len(a)):
        c = a[i]
        d = b[i]

        #
        r = np.abs(c - d)
        # Add base so we don't compare a value of 0 against something like 0.0001. Results would say these
        # values are too different, but in practice, this kind of difference should be fine
        m = np.mean([c, d]) + base

        percent_diff = r / m

        if percent_diff - tolerance > 0:
            valid = False
            break

    return valid


def run_mpSPDZ(settings_map, run_program="test_forwarding"):
    runner = settings_map["VM"]
    is_online = settings_map["online"].lower() == "true"
    path_to_spdz = settings_map['path_to_top_of_mpspdz']
    path_to_this_repo = settings_map["path_to_this_repo"]

    intermediate_results_file = path_to_this_repo + "/tmp.save"

    if settings_map["party"] == "0":
        with open(intermediate_results_file, 'w') as stream:
            stream.write("")
        if is_online:
            run_cmd = "cd {a} && ./{b} {file} -pn {c} -h {d} >> {e}".format(a=path_to_spdz, b=runner, file=run_program,
                                                                            c=settings_map["host_port"],
                                                                            d=settings_map["host_ip"],
                                                                            e=intermediate_results_file
                                                                            )
        else:
            run_cmd = "cd {a} && ./{b} {file} >> {e}".format(a=path_to_spdz, b=runner, file=run_program,
                                                             e=intermediate_results_file)

    else:
        if is_online:
            run_cmd = "cd {a} && ./{b} {file} -pn {c} -h {d}".format(a=path_to_spdz, b=runner, file=run_program,
                                                                     c=settings_map["host_port"],
                                                                     d=settings_map["host_ip"],
                                                                     )
        else:
            run_cmd = "cd {a} && ./{b} {file}".format(a=path_to_spdz, file=run_program, b=runner)

    print("Starting secure program with command: {a}".format(a=run_cmd))

    try_again = True
    allowed_attempts = 10
    total_attempts = 0
    while try_again:
        try:
            subprocess.check_call(run_cmd, shell=True)
            try_again = False
        except Exception:
            if total_attempts >= allowed_attempts:
                try_again = False
            total_attempts += 1

    # TODO: these conditions are too restrictive, need to re-work this area
    if settings_map["party"] == "0" and run_program == "run":
        save_file_times = settings_map["path_to_this_repo"] + "/storage/results/mpc/times.save"
        save_file_intermediate = settings_map["path_to_this_repo"] + "/storage/results/mpc/results.save"
        save_file_classifications = settings_map["path_to_this_repo"] + "/storage/results/mpc/classifications.save"

        save_results = ""

        with open(intermediate_results_file, 'r') as stream:
            for line in stream:
                save_results += line

        save_results = save_results.split("@results")

        for line in save_results:
            if "Stopped timer" in line:
                with open(save_file_times, 'a+') as stream:
                    stream.write(line)
            else:
                if "class" in line:
                    with open(save_file_classifications, 'w') as stream:
                        stream.write(line.replace("class", ""))
                elif "weights" in line:
                    with open(save_file_intermediate, 'w') as stream:
                        stream.write(line.replace("weights", ""))


def compile_spdz(settings_map, compile_program="run"):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0" and settings_map["online"].lower() != "true":
        return

    # Compile .mpc program
    c = settings_map["compiler"]
    online = settings_map["online"]

    script_to_compile = compile_program + ".mpc"

    subprocess.check_call("cp {a}/mpc_code/{b} {c}/Programs/Source/{b}".
                          format(a=settings_map['path_to_this_repo'], b=script_to_compile,
                                 c=settings_map["path_to_top_of_mpspdz"]),
                          shell=True)

    subprocess.check_call("python3 {a}/compile.py {b} {c}".format(a=settings_map["path_to_top_of_mpspdz"], b=c,
                                                                  c=compile_program), shell=True)

    print("cp {a}/mpc_code/{b} {c}/Programs/Source/{b}".format(a=settings_map['path_to_this_repo'],
                                                               b=script_to_compile,
                                                               c=settings_map["path_to_top_of_mpspdz"]))

    print("python3 {a}/compile.py {b} {c}".format(a=settings_map["path_to_top_of_mpspdz"], b=c,
                                                  c=compile_program))

    # if not online.lower() == "true":
    #     subprocess.check_call("rm tmp.txt", shell=True)


def populate_spdz_files(settings_map):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0" and settings_map["online"].lower() != "true":
        return

    def getListOfFiles(dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)

            # run.mpc should not go to the Compiler directory
            if ".mpc" in fullPath or "init" in fullPath:
                continue

            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append((fullPath, entry))

        return allFiles

    dir = settings_map["path_to_this_repo"] + "/mpc_code"

    allFiles = getListOfFiles(dir)

    # puts all mpc files into the MP-SPDZ compiler - this excludes run.mpc
    for path_data in allFiles:
        subprocess.check_call("cp {a} {b}/Compiler/{c}".
                              format(a=path_data[0], b=settings_map["path_to_top_of_mpspdz"], c=path_data[1]),
                              shell=True)

    # Now take care of run.mpc
    subprocess.check_call("cp {a}/mpc_code/run.mpc {b}/Programs/Source/run.mpc".
                          format(a=settings_map['path_to_this_repo'], b=settings_map["path_to_top_of_mpspdz"]),
                          shell=True)

    # Now take care of run.mpc
    subprocess.check_call("cp {a}/mpc_code/test_forwarding.mpc {b}/Programs/Source/run.mpc".
                          format(a=settings_map['path_to_this_repo'], b=settings_map["path_to_top_of_mpspdz"]),
                          shell=True)


def edit_source_code(settings_map, all_metadata, data, run_personalizor="false"):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0" and settings_map["online"].lower() != "true":
        return

    repo_file_path = settings_map["path_to_this_repo"] + "/mpc_code/run.mpc"

    kshot = settings_map["kshot"]
    shapes = all_metadata
    n_timesteps = data[0].shape[1]
    n_features = data[0].shape[2]
    n_outputs = data[1].shape[1]
    test_samples = data[0].shape[0]
    norm = settings_map["normalize"]

    print("test size is -- {a}".format(a=test_samples))

    file = []
    found_delim = False
    start_of_delim = 0

    i = 0
    with open(repo_file_path, 'r') as stream:
        for line in stream:
            if not found_delim and "@args" in line:
                start_of_delim = i
                found_delim = True
            i += 1
            file.append(line)

    compile_args = _format_args(normalize=norm, test_data_len=test_samples, kshot=kshot, window_size=n_timesteps,
                                shapes=shapes, n_features=n_features, n_outputs=n_outputs,
                                run_personalizor=run_personalizor)

    file[start_of_delim + 1] = "settings_map = {n}\n".format(n=compile_args)

    # file as a string
    file = ''.join([s for s in file])

    with open(repo_file_path, 'w') as stream:
        stream.write(file)


def _format_args(**kwargs):
    res = "{"

    # # shapes is a special case (list of strings), so we can populate that explicitly
    # shapes = kwargs.pop("shapes")
    # res += "\'{key}\': {value},".format(key="metrics", value=shapes)

    for key in kwargs:
        res += "\'{key}\': \'{value}\',".format(key=key, value=kwargs[key])

    # Omit last comma
    res = res[:-1] + "}"

    return res


def distribute_Data(settings_map):
    if settings_map["ignore_custom_networking"].lower() == "true":
        # Note this has to be manually maintained. It's really just for testing purposes
        # return "[[16, 12, 2],[16],[128, 16, 8],[128],[50, 256],[50]]"
        return "[[8, 9, 2],[8],[128, 8, 5],[128],[50, 128],[50]]"

    is_model_owner = bool(settings_map["party"] == "0")

    metadata = None

    if is_model_owner:
        metadata = _read_shapes(settings_map)

    all_metadata = None

    if is_model_owner:
        all_metadata = _distribute_as_client(settings_map, metadata)
    else:
        all_metadata = _distribute_as_host(settings_map, metadata)

    all_metadata = str(all_metadata)
    all_metadata = all_metadata.replace("\'", "").replace("\"", "")

    # print("all metadata: {a}".format(a=all_metadata))

    return str(all_metadata)

    # ./storage/spdz_compatible/save_model.txt


def _read_shapes(settings_map):
    path_to_this_repo = settings_map["path_to_this_repo"]
    shape_path = path_to_this_repo + "/storage/spdz_compatible/spdz_shapes.save"

    metadata = []

    with open(shape_path, 'r') as f:
        for line in f:
            metadata.append(line.replace("(", "[").replace(")", "]").replace("\n", "")
                            .replace(" ", "").replace(",]", "]"))

    metadata = "[" + ",".join(metadata) + "]"
    print(metadata)
    return metadata


def _distribute_as_host(settings_map, metadata=None):
    # TODO: Need an exit cond. if not online

    if settings_map["online"].lower() != "true":
        return metadata

    data = server.run(settings_map, introduce=False)  # receive data

    return data.split("@seperate")


def _distribute_as_client(settings_map, metadata):
    # print(metadata)

    if settings_map["online"].lower() == "false":
        return metadata

    client.run(settings_map, metadata, introduce=False)

    return metadata


def store_secure_params(settings_map, kshot_source_data, kshot_target_data, target_test_data,
                        target_test_fake_norm_data, target_kshot_fake_norm_data, distribution):
    # TODO: These tasks should, ideally, be split up between the parties
    if settings_map["party"] != "0":
        return

    # loads params into intermediate files to be sent to MP-SPDZ files
    spdz_format_cnn.load_payload(settings_map)
    model_params_path = "./storage/spdz_compatible/spdz_cnn.save"

    all_data = []

    if settings_map["normalize"].lower() == "source":
        # Commenting out for a test: TODO: Fix things, and uncomment this
        # kshot_target_data = target_kshot_fake_norm_data
        # target_test_data = target_test_fake_norm_data

        dist_mean = str(distribution[0]).replace("[", '').replace("]", '').replace(",", '')
        dist_std = str(distribution[1]).replace("[", '').replace("]", '').replace(",", '')

        all_data.append(dist_mean)
        all_data.append(dist_std)

    with open(model_params_path, 'r') as stream:
        # Note, should only be one, really long line
        for line in stream:
            all_data.append(line)

    # upload data
    # for matrix in kshot_source_data[0]:
    #     matrix = str(matrix.T.tolist())
    #     matrix = matrix.replace("[", '').replace("]", '').replace(",", '')
    #     all_data.append(matrix)
    #
    # for ohe_label in kshot_source_data[1]:
    #     all_data.append(str(int(np.argmax(ohe_label))))

    wm = _pre_process_source_data(settings_map, kshot_source_data)
    matrix = str(wm)
    matrix = matrix.replace("[", '').replace("]", '').replace(",", '')
    all_data.append(matrix)

    for matrix in kshot_target_data[0]:
        matrix = str(matrix.T.tolist())
        matrix = matrix.replace("[", '').replace("]", '').replace(",", '')
        all_data.append(matrix)

    for ohe_label in kshot_target_data[1]:
        all_data.append(str(int(np.argmax(ohe_label))))

    for matrix in target_test_data[0]:
        matrix = str(matrix.T.tolist())
        matrix = matrix.replace("[", '').replace("]", '').replace(",", '')
        all_data.append(matrix)

    ' '.join(all_data)

    all_data = str(all_data).replace("]", '').replace("[", '').replace(",", '').replace("\'", "").replace("\\n", "")

    with open(settings_map["path_to_private_data"], 'w') as stream:
        stream.write(all_data)


def store_itc_results(settings_map, cnn_acc_res, pers_result):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0":
        return

    results_filepath = "./storage/results/itc/accuracy.csv"

    result = "-----@start-----\n"
    result += "cnn_acc_res = " + str(cnn_acc_res) + "\n"
    result += "pers_result = " + str(pers_result) + "\n"
    result += "target_id = " + settings_map["target_id"] + "\n"
    result += "net = " + settings_map["net"] + "\n"
    result += "epochs = " + settings_map["epochs"] + "\n"
    result += "kshot = " + settings_map["kshot"] + "\n"
    result += "-----@end-----\n"

    with open(results_filepath, 'a+') as f:
        f.write(result)


def _load_cnn(settings_map, data):
    n_timesteps = data[0].shape[1]
    n_features = data[0].shape[2]
    n_outputs = data[1].shape[1]

    net_string = settings_map["net"]
    model = nets().models[net_string](n_timesteps, n_features, n_outputs)

    cnn_to_load_path = settings_map["cnn_path"]

    model.load_weights(cnn_to_load_path)

    return model


def personalize_classifier(settings_map, source_data, target_test_data, target_kshot_data):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0":
        return

    n_outputs = source_data[1].shape[1]

    model = _load_cnn(settings_map, source_data)

    model_feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    label_space = [i for i in range(n_outputs)]

    personalizer = pers.personalizer(label_space=label_space, feature_extractor=model_feature_extractor)
    personalizer.initialize_weight_matrix(settings_map, source_data, target_kshot_data)

    _save_weight_matrix(settings_map, personalizer)

    results = []
    correct_results = []

    for i in range(len(target_test_data[0])):
        data = target_test_data[0][i]
        new_data = np.expand_dims(data, axis=0)
        results.append(personalizer.classify(new_data))
        correct_results.append(np.dot(label_space, target_test_data[1][i]))

    # results = personalizer.classify(target_test_data[0])
    # correct_results = []
    #
    # for i in range(len(target_test_data[0])):
    #     correct_results.append(np.dot(label_space, target_test_data[1][i]))
    #
    # print(results)
    # print(correct_results)

    return float(sum([int(results[i] == correct_results[i]) for i in range(len(results))])) / len(target_test_data[0])


def _save_weight_matrix(settings_map, personalizer):
    weight_path = settings_map["path_to_this_repo"] + "/storage/results/itc/weight_matrix.save"

    matrix = str([['{:.7f}'.format(b) for b in a] for a in personalizer.weight_matrix.tolist()])

    # print(matrix)

    with open(weight_path, 'w') as f:
        f.write(matrix)


def partition_data(settings_map, data, collect_subset=False):
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
        holdout_indices = np.random.choice(rows_of_subset, size=kshot, replace=False)
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

    return test, holdout


def train(settings_map, source_data, target_test_data):
    # If this is offline, then just let party 0 do this step
    if settings_map["party"] != "0":
        return

    seed = settings_map["random_seed"]

    if seed.lower() != "none":
        seed = int(seed)
        import tensorflow as tf
        tf.random.set_seed(seed)

    accuracy = None

    # if we are not training, then simply classify target_data using the pre-built model
    if settings_map["train_cnn"].lower() != "true":
        # TODO: Upload correct weights...
        model = _load_cnn(settings_map, source_data)

        results = model.evaluate(target_test_data[0], target_test_data[1], verbose=0)
        accuracy = results[1]
    else:
        epochs = int(settings_map["epochs"])

        # Note that training the CNN using this function will save model parameters for later use by in-the-clear code
        accuracy = train_CNN(source_data[0], source_data[1],
                             target_test_data[0], target_test_data[1]).run_experiment(settings_map, repeats=1,
                                                                                      epochs=epochs)

    # Store the model in a different file format than it is currently saved for MP-SPDZ
    _store_cnn_SPDZ_format(settings_map, source_data)

    return accuracy / 100.0


def _store_cnn_SPDZ_format(settings_map, data):
    model = _load_cnn(settings_map, data)

    # print(model.summary())

    layers_w = model.get_weights()
    layer_str = ""
    layer_wstr = ""

    for layer in layers_w:
        layer_str += str(layer.tolist()) + "\n@end\n"
        layer_wstr += str(layer.shape) + "\n"
        # print(layer.shape)

    with open("./storage/spdz_compatible/save_model.txt", 'w') as f:
        # f.write(layer_wstr)
        # f.write("@end")
        f.write(layer_str)
