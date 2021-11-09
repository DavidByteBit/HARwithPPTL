import numpy as np

from Compiler.mpc_math import log_fx
from Compiler.mpc_math import cos
from Compiler.mpc_math import sin
from Compiler.mpc_math import sqrt
from Compiler.types import *
from Compiler.library import *
from Compiler import util

threads = 16

class Layers:

    def __init__(self, time_forward_pass=False):
        self.time_forward_pass = time_forward_pass
        self.layers = []
        # Have to start with a high, safe number so that it does not mess with any other functions
        self.timer = 1000

    def add_layer(self, l):
        self.layers.append(l)

    def forward(self, input):
        timer = self.timer

        processed_input = input

        for l in self.layers:
            if self.time_forward_pass:
                start_timer(timer_id=timer)
                processed_input = l.compute(processed_input)
                if l.flatten_after:
                    processed_input = flatten(processed_input)
                stop_timer(timer_id=timer)
                timer += 1
            else:
                processed_input = l.compute(processed_input)
                if l.flatten_after:
                    processed_input = flatten(processed_input)

        self.timer = timer

        return processed_input

    def get_final_dim(self):
        return self.layers[-1].output_shape


class Layer:

    def __init__(self, input_shape, output_shape, flatten_after=False):
        self.flatten_after = flatten_after
        self.input_shape = input_shape
        self.output_shape = output_shape


class Dense(Layer):

    def __init__(self, input_shape, output_shape, w, b, activation, flatten_after=False):
        super(Dense, self).__init__(input_shape, output_shape, flatten_after)
        # TODO should be shape, but Arrays have no shape...
        self.w_shape0 = len(w)
        self.w_shape1 = len(w[0])
        self.w = w

        self.b_shape = len(b)
        self.b = b

        self.activation = activation

    def compute(self, input_vec):
        print(input_vec)

        # TODO currently assumes 1d input/output
        w = self.w
        b = self.b

        w_shape0 = self.w_shape0

        output = sfix.Array(self.output_shape)

        weighted_inputs = sfix.Array(w_shape0)

        weighted_inputs.assign_vector((self.w.dot(input_vec)).get_vector())

        @for_range_parallel(w_shape0, w_shape0)
        def _(i):
            output[i] = self.activation(weighted_inputs[i] + b[i])

        print("dense")

        print(output)

        return output


class MaxPooling1D(Layer):

    def __init__(self, input_shape, output_shape, width, filter_dim, flatten_after=False):
        super(MaxPooling1D, self).__init__(input_shape, output_shape, flatten_after)
        self.width = width
        self.filter_dim = filter_dim
        # TODO: padding, stride

    def compute(self, input):

        width = self.width
        filter_dim = self.filter_dim
        output_width = len(input[0]) // width
        left_out_elements = len(input[0]) % width

        # assert filter_dim, output_width == self.output_shape

        output = sfix.Tensor((filter_dim, output_width))

        @for_range_opt((filter_dim, output_width - 1))
        def _(i, j):
            val = sfix.Array(width)

            @for_range(width)
            def _(k):
                val[k] = input[i][j * width + k]

            output[i][j] = util.max(val)

        @for_range(filter_dim)
        def _(i):
            val = sfix.Array(width + left_out_elements)

            @for_range(width + left_out_elements)
            def _(k):
                val[k] = input[i][(output_width - 1) * width + k]

            output[i][(output_width - 1)] = util.max(val)

        # print("maxpool")
        print(output)

        return output


class Conv1D(Layer):

    def __init__(self, input_shape, output_shape, kernels, kernel_bias, activation, stride=None, flatten_after=False):
        super(Conv1D, self).__init__(input_shape, output_shape, flatten_after)
        self.activation = activation
        self.kernel_bias = kernel_bias
        self.kernels = kernels  # multi dimensioned because of multiple filters
        self.filters = len(kernels)  # size
        self.kernel_w = len(kernels[0][0])  # prev filters_dim or input height
        self.kernel_h = len(kernels[0])

        # TODO: padding, stride

    def compute(self, input):
        # print(input)
        kernels = self.kernels
        kernels_bias = self.kernel_bias
        k_width = self.kernel_w
        # print(k_width)
        output_width = len(input[0]) - k_width + 1

        # assert self.filters, output_width == self.output_shape

        output = sfix.Tensor((self.filters, output_width))

        cross_section = MultiArray([output_width, self.kernel_h, self.kernel_w], sfix)

        for o in range(output_width):
            for k in range(self.kernel_h):
                for e in range(self.kernel_w):
                    cross_section[o][k][e] = input[k][e + o]

        # print("first time")
        # print(output)
        @for_range_opt_multithread(2 * threads, (self.filters, output_width))
        def _(i, j):
            # val = sfix.Matrix(self.kernel_h, self.kernel_w)

            # @for_range(self.kernel_h)
            # def _(k):
            #     @for_range(self.kernel_w)
            #     def _(e):
            #         val[k][e] = input[k][e + j]  # optimize by doing things in-place?

            # print(kernels[j])
            output[i][j] = self.activation(dot_2d(cross_section[j], kernels[i]) + kernels_bias[i])

        # print("conv")

        print(output)

        return output


# TODO optimize
def max(x):
    max_value = sfix.Array(1)
    max_value[0] = x[0]
    "MAX CHECKPOINT 3.5"
    @for_range(len(x) - 1)
    def _(i):
        cmp = max_value[0] > x[i + 1]
        max_value[0] = cmp * max_value[0] + (1 - cmp) * x[i + 1]

    return max_value[0]


# TODO only works with 2d to 1
def flatten(x):
    w = len(x)
    h = len(x[0])

    new_array = sfix.Array(w * h)

    @for_range_opt((w, h))
    def _(i, j):
        new_array[i + j * w] = x[i][j]

    return new_array


def dot_1d(x, y):
    return sum(x * y)

    # res = sfix.Array(1)
    # res[0] = sfix(0)
    #
    # @for_range(len(x))
    # def _(i):
    #     res[0] += x[i] * y[i]
    #
    # return res[0]


def dot_2d(x, y):
    res = sfix.Array(1)
    res[0] = sfix(0)

    # print(x[0])
    # print(y[0])

    assert len(x) == len(y)
    assert len(x[0]) == len(y[0])

    # c = sfix.Array(len(x[0]))

    # WARNING: Consider removing parallelization if the results are looking incorrect
    @for_range_parallel(len(x), len(x))
    def _(i):
        c = sum(x[i] * y[i])
        res[0] += c

    return res[0]

    # @for_range(len(x))
    # def _(i):
    #     @for_range(len(x[0]))
    #     def _(j):
    #         prod = x[i][j] * y[i][j]
    #         res[0] += prod
    #
    # return res[0]
