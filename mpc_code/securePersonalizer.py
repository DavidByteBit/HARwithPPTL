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


def personalization(layers, matrix_to_populate, target, total_amount_of_data, output_dim, n_outputs, kshot):

    # Data and labels run parallel to each other
    data = target[0]
    labels = target[1]

    projected_data = sfix.Matrix(total_amount_of_data, output_dim)

    @for_range_parallel(total_amount_of_data//8, total_amount_of_data)
    def _(i):
        projected_data[i] = layers.forward(data[i])  # Line 5 prep-work

    for i in range(total_amount_of_data):
        print_ln("%s@end", projected_data[i].reveal_nested())

    @for_range(n_outputs)  # Line 2
    def _(j):
        num = sfix.Array(output_dim)  # Length may need to be dynamic.
        num.assign_all(0)
        dem = sfix.Array(1)
        dem[0] = sfix(0)
        @for_range(total_amount_of_data)  # Line 3
        def _(i):
            eq_res = (sint(j) == labels[i])  # Line 4

            # print_ln("j = %s, i = %s, eq = %s", j, i, eq_res.reveal())

            feat_res = projected_data[i]  # Line 5

            scalar = sfix.Array(output_dim)
            scalar.assign_all(eq_res)

            num_intermediate = sfix.Array(output_dim)
            num_intermediate.assign(scalar * feat_res)  # line 6

            matrix_to_populate[j] += num_intermediate * (1.0/(2 * kshot))

        W_intermediate_1 = matrix_to_populate[j]

        # W_intermediate_1.assign(num / dem_extended)  # line 10

        # @for_range(output_dim)  # Line 10
        # def _(k):
        #     W_intermediate_1[k] = num[k] * dem_extended[k]

        W_intermediate_2 = 1/Euclid(W_intermediate_1)  # Line 11

        for k in range(output_dim):  # Line 12
            matrix_to_populate[j][k] = W_intermediate_1[k] * W_intermediate_2

    print_ln("weights%s", matrix_to_populate.reveal_nested())

    return matrix_to_populate  # Line 13


def infer(layers, weight_matrix, unlabled_data, output_dim, batch):
    data_size = len(unlabled_data)

    projected_data = sfix.Matrix(data_size, output_dim)

    classifications = sfix.Array(data_size)

    if batch:
        @for_range_parallel(15, data_size)
        def _(i):
            projected_data[i] = layers.forward(unlabled_data[i])  # line1
            # print_ln("%s@end", projected_data[i].reveal_nested())

            label_space_size = len(weight_matrix)

            # rankings = sfix.Array(label_space_size)

            @for_range_opt(data_size)  # Line 2
            def _(i):
                rankings = sfix.Array(label_space_size)
                rankings.assign_vector((weight_matrix.dot(projected_data[i])).get_vector())
                # @for_range_opt(label_space_size)  # Line 2
                # def _(j):
                #     rankings[j] = sfix.dot_product(weight_matrix[j], projected_data[i])  # Line 3
                classifications[i] = ml.argmax(rankings)

    else:

        @for_range(data_size)
        def _(i):
            projected_data[i] = layers.forward(unlabled_data[i])  # line1

            label_space_size = len(weight_matrix)

            # rankings = sfix.Array(label_space_size)

            @for_range(data_size)  # Line 2
            def _(i):
                rankings = sfix.Array(label_space_size)
                rankings.assign_vector((weight_matrix.dot(projected_data[i])).get_vector())
                # @for_range_opt(label_space_size)  # Line 2
                # def _(j):
                #     rankings[j] = sfix.dot_product(weight_matrix[j], projected_data[i])  # Line 3
                classifications[i] = ml.argmax(rankings)


    return classifications  # Line 4,5
