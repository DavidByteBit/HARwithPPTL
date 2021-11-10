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

    projected_data = sfix.Matrix(data_size, output_dim)

    @for_range_parallel(data_size//8, data_size)
    def _(i):
        projected_data[i] = layers.forward(data[i])  # Line 5 prep-work

    for i in range(data_size):
        print_ln("%s@end", projected_data[i].reveal_nested())

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

            feat_res = projected_data[i]  # Line 5

            scalar = sfix.Array(output_dim)
            scalar.assign_all(eq_res)

            num_intermediate = sfix.Array(output_dim)
            num_intermediate.assign(scalar * feat_res)  # line 6

            dem[0] += eq_res  # Line 7
            @for_range(output_dim)
            def _(k):
                num[k] += num_intermediate[k]  # line 8

        dem_extended = sfix.Array(output_dim)
        dem_extended.assign_all(1/dem[0])  # Line 9

        W_intermediate_1 = sfix.Array(output_dim)

        # W_intermediate_1.assign(num / dem_extended)  # line 10

        # @for_range(output_dim)  # Line 10
        # def _(k):
        #     W_intermediate_1[k] = num[k] * dem_extended[k]

        W_intermediate_1.assign(num * dem_extended)

        W_intermediate_2 = 1/Euclid(W_intermediate_1)  # Line 11

        for k in range(output_dim):  # Line 12
            weight_matrix[j][k] = W_intermediate_1[k] * W_intermediate_2

    print_ln("%s", weight_matrix.reveal_nested())

    return weight_matrix  # Line 13


def infer(layers, weight_matrix, unlabled_data, output_dim):
    data_size = len(unlabled_data)

    projected_data = sfix.Matrix(data_size, output_dim)

    @for_range_parallel(data_size//16, data_size)
    def _(i):
        projected_data[i] = layers.forward(unlabled_data[i])  # line1
        # print_ln("%s@end", projected_data[i].reveal_nested())

    label_space_size = len(weight_matrix)

    rankings = sfix.Array(label_space_size)

    classifications = sfix.Array(data_size)

    @for_range_opt(data_size)  # Line 2
    def _(i):
        @for_range_opt(label_space_size)  # Line 2
        def _(j):
            rankings[j] = dot_product(weight_matrix[j], projected_data[i])  # Line 3
        classifications[i] = ml.argmax(rankings)

    return classifications  # Line 4,5
