import main
import sys

settings_path = sys.argv[1]

target_start = 1

if len(sys.argv) >= 3:
    target_start = int(sys.argv[2])

target_id_loc = 0
test_range_loc = 0
train_cnn_loc = 0
kshot_loc = 0

original_file_contents = []
with open(settings_path, 'r') as in_stream:

    counter = 0

    for line in in_stream:
        original_file_contents.append(line)

        if "target_id" in line:
            target_id_loc = counter
        if "test_range" in line:
            test_range_loc = counter
        if "train_cnn" in line:
            train_cnn_loc = counter
        if "kshot" in line:
            kshot_loc = counter

        counter += 1

print("original file contents \n\n{a}\n".format(a="".join(original_file_contents)))


def alter_settings(target_id_val, train_cnn_val, train_kshot_val):
    new_file_contents = original_file_contents[:]

    new_file_contents[target_id_loc] = str(" target_id: \"{a}\"\n".format(a=target_id_val))
    new_file_contents[train_cnn_loc] = str(" train_cnn: \"{a}\"\n".format(a=train_cnn_val))
    new_file_contents[kshot_loc] = str(" kshot: \"{a}\"\n".format(a=train_kshot_val))

    with open(settings_path, 'w') as out_stream:
        out_stream.write("".join(new_file_contents))
        print("new file contents \n\n{a}\n".format(a="".join(new_file_contents)))


num_of_participants = 10

num_of_tests = 5
kshot_vals = [1, 5, 10]

for k in kshot_vals:
    new_kshot_val = k
    for i in range(num_of_participants - (target_start - 1)):
        for _ in range(num_of_tests):
            new_target_id_val = i + target_start
            new_train_cnn = "true"

            alter_settings(new_target_id_val, new_train_cnn, new_kshot_val)
            main.run_main(settings_path)
