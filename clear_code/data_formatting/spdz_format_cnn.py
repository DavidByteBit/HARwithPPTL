import numpy as np
import json


def load_payload(settings_map):
    path_to_this_repo = settings_map["path_to_this_repo"]

    file = path_to_this_repo + "/storage/spdz_compatible/save_model.txt"

    file_as_str = []

    with open(file, 'r') as f:
        for line in f:
            file_as_str.append(line)

    file_as_str = "".join(file_as_str)

    parameters = file_as_str.split("@end")[:-1]
    new_p = []

    for line in parameters:
        new_p.append(json.loads(line.replace("\n", "")))

    shapes = []

    for line in new_p:
        shapes.append(np.array(line).shape)

    # print(shapes)

    # TODO make dynamic
    new_p[0] = np.array(new_p[0]).T.tolist()
    new_p[2] = np.array(new_p[2]).T.tolist()
    new_p[4] = np.array(new_p[4]).T.tolist()

    # print(len(new_p))
    new_p = new_p[:-2]
    # print(len(new_p))

    shapes = []

    for line in new_p:
        shapes.append(np.array(line).shape)

    # print(shapes)

    # recursively flattens list
    def flatten(S):
        if not S:
            return S
        if isinstance(S[0], list):
            return flatten(S[0]) + flatten(S[1:])
        return S[:1] + flatten(S[1:])

    new_p = flatten(new_p)

    print("CHECKPOINT 1 ")
    print(shapes)

    with open(path_to_this_repo + "/storage/spdz_compatible/spdz_shapes.save", 'w') as f:
        for line in shapes:
            f.write(str(line) + "\n")

    with open(path_to_this_repo + "/storage/spdz_compatible/spdz_cnn.save", 'w') as f:
        f.write(str(new_p).replace("]", '').replace("[", '').replace(",", ''))

    test_val = ""

# with open("test_val.save", 'r') as f:
#     for line in f:
#         test_val = json.loads(line)
#         test_val = str(np.array(test_val).T.tolist())
#         test_val = test_val.replace("[", '').replace("]", '').replace(",", '')
#         break
#
# print(test_val)
#
# with open("formatted_test_val", 'w') as f:
#     f.write(test_val)
