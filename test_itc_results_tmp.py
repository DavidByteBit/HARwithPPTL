import main
import sys

settings_path = sys.argv[1]

target_id_loc = 0
train_cnn_loc = 0
kshot_loc = 0

original_file_contents = []
with open(settings_path, 'r') as in_stream:

    counter = 0

    for line in in_stream:
        original_file_contents.append(line)

        if "target_id" in line:
            target_id_loc = counter
        elif "train_cnn" in line:
            train_cnn_loc = counter
        elif "kshot" in line:
            kshot_loc = counter

        counter += 1


def alter_settings(target_id_val, train_cnn_val, kshot_val):
    new_file_contents = original_file_contents[:]

    new_file_contents[target_id_loc] = str(target_id_val)
    new_file_contents[train_cnn_loc] = str(train_cnn_val)
    new_file_contents[kshot_loc] = str(kshot_val)

    with open(settings_path, 'w') as out_stream:
        out_stream.write("".join(new_file_contents))


num_of_participants = 10
num_of_tests = 5
kshot_vals = [1,5,10]

for k in kshot_vals:
    new_kshot_val = k
    for i in range(num_of_participants):
        new_target_id_val = i + 1
        new_train_cnn = "true"
        for j in range(num_of_tests):
            alter_settings(new_target_id_val, new_train_cnn, new_kshot_val)
            main.run_main(settings_path)

        new_train_cnn = "false"
        for j in range(num_of_tests):
            alter_settings(new_target_id_val, new_train_cnn, new_kshot_val)
            main.run_main(settings_path)
