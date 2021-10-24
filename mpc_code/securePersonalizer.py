from Compiler import mpc_math
from Compiler import ml

from Compiler.types import *
from Compiler.library import *


def dot_product(a, b):
    assert (len(a) == len(b))
    return sum(a * b)


def Euclid(x):
    ret_x = sint(0)

    for k in range(len(x)):
        ret_x += x[k] * x[k]

    ret_x = mpc_math.sqrt(ret_x)

    return ret_x


def personalization(layers, source, target, total_amount_of_data, output_dim, label_space):

    print_ln("\n\n\n\n%s", label_space.reveal())

    source_size = len(source[0])
    target_size = len(target[0])

    window_size = len(source[0][0])

    feat_size = len(source[0][0][0])
    print(feat_size)

    data_size = total_amount_of_data

    # Data and labels run parallel to each other
    data = MultiArray([data_size, window_size, feat_size], sfix)
    labels = sint.Array(data_size)

    # Line 1
    @for_range(window_size)
    def _(j):
        @for_range(feat_size)
        def _(k):
            @for_range(source_size)
            def _(i):
                data[i][j][k] = source[0][i][j][k]
            @for_range(target_size)
            def _(i):
                data[i + source_size][j][k] = target[0][i][j][k]


    @for_range(target_size)
    def _(i):
        labels[i] = source[1][i]
        labels[i + source_size] = target[1][i]

    print_ln("\n%s", labels.reveal_nested())
    # print_ln("\n%s", data.reveal_nested())


    projected_data = sfix.Matrix(data_size, output_dim)

    @for_range(data_size)
    def _(i):
        projected_data[i] = layers.forward(data[i])  # Line 5 prep-work

    weight_matrix = sfix.Matrix(len(label_space), output_dim)

    @for_range(len(label_space))  # Line 2
    def _(j):
        num = sfix.Array(output_dim)  # Length may need to be dynamic.
        num.assign_all(0)
        dem = sfix.Array(1)
        dem[0] = sfix(0)
        @for_range(data_size)  # Line 3
        def _(i):
            eq_res = (sint(j) == labels[i])  # Line 4

            print_ln("\neq result: %s", eq_res.reveal())

            feat_res = projected_data[i]  # Line 5

            print_ln("\nfeat result: %s", feat_res.reveal())

            scalar = sfix.Array(output_dim)
            @for_range(output_dim)
            def _(k):
                scalar[k] = eq_res

            num_intermediate = sfix.Array(output_dim)

            num_intermediate.assign(scalar * feat_res)  # line 6

            print_ln("\nnum_int result: %s", num_intermediate.reveal_nested())

            # @for_range(output_dim)
            # def _(k):
            #     num_intermediate[k] = scalar[k] * feat_res[k]

            dem[0] += eq_res  # Line 7
            @for_range(output_dim)
            def _(k):
                num[k] += num_intermediate[k]  # line 8

            print_ln("\nnum result: %s", num.reveal_nested())

        dem_extended = sfix.Array(output_dim)
        dem_extended.assign_all(dem[0])  # Line 9

        print_ln("\ndem_ext result: %s", dem_extended.reveal_nested())

        W_intermediate_1 = sfix.Array(output_dim)

        # W_intermediate_1.assign(num / dem_extended)  # line 10

        @for_range(output_dim)  # Line 10
        def _(k):
            W_intermediate_1[k] = num[k] / dem_extended[k]

        print_ln("\nW_inter result: %s", W_intermediate_1.reveal_nested())

        W_intermediate_2 = Euclid(W_intermediate_1)  # Line 11

        print_ln("\neuclid result: %s", W_intermediate_2.reveal())

        for k in range(output_dim):  # Line 12
            weight_matrix[j][k] = W_intermediate_1[k] / W_intermediate_2

        print_ln("\nweight_matrix result: %s", weight_matrix.reveal_nested())

    return weight_matrix  # Line 13


def infer(layers, weight_matrix, unlabled_data):
    data_feature = layers.forward(unlabled_data)  # Line 1

    label_space_size = len(weight_matrix)

    rankings = sfix.Array(label_space_size)

    @for_range_opt(label_space_size)  # Line 2
    def _(j):
        rankings[j] = dot_product(weight_matrix[j], data_feature)  # Line 3

    return ml.argmax(rankings)  # Line 4,5


#####################################################################################

# # CONSTANTS TO MAKE SURE THINGS WORK
# def feature_extractor(x):
#     a = Array(len(x), sint)
#     return a
#
#
# source = (Matrix(3, 2, sfix), Array(3, sint))
# source[0][0][0] = 0
# source[0][1][0] = 1
# source[0][0][1] = 2
# source[0][1][1] = 3
# source[0][2][0] = 4
# source[0][2][1] = 5
#
# source[1][0] = sint(0)
# source[1][1] = sint(1)
# source[1][2] = sint(0)
#
# target = (Matrix(2, 2, sfix), Array(2, sint))
# target[0][0][0] = 0
# target[0][1][0] = 1
# target[0][0][1] = 2
# target[0][1][1] = 3
#
# target[1][0] = sint(1)
# target[1][1] = sint(0)
#
# target_unlabled = Array(2, sfix)
# target_unlabled[0] = 0
# target_unlabled[1] = 1
#
# label_space = [sint(0), sint(1)]
#
# W = personalization(feature_extractor, source, target, label_space)
#
# predicted_label = infer(feature_extractor, W, target_unlabled)
#
# print_ln(str(predicted_label.reveal()))
