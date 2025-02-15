# -*- coding: utf-8 -*-
"""conv2_update.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EvHhcnyRf0vNu82JeAekTnIhDJ9VA-9s
"""

import numpy as np

class Conv3x3_1_to_n_padding:
    def __init__(self, output=1, input=1, activation=None, dtype=np.float32):
        num_filters = output
        in_ch = input
        self.activation = activation
        self.dtype = dtype

        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3).astype(dtype) / 9
        self.biases = np.zeros(num_filters, dtype=dtype)

    def iterate_regions(self, image):
        image_padded = np.pad(image, 1, mode="constant", constant_values=0)
        h, w = image_padded.shape[:2]
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image_padded[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h, w, self.num_filters), dtype=self.dtype)
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        self.last_output = output
        if self.activation is not None:
            output = self.activation(output)
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape, dtype=self.dtype)
        d_L_d_input = np.zeros(self.last_input.shape, dtype=self.dtype)

        if self.activation is not None:
            d_L_d_out = self.activation.backward(self.last_output) * d_L_d_out

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                for out_ch in range(self.num_filters):
                    d_L_d_input[:, :] += d_L_d_out[:, :, out_ch] * self.filters[out_ch]

        self.filters -= learn_rate * d_L_d_filters
        self.biases -= learn_rate * np.sum(d_L_d_out, axis=(0, 1))

        return d_L_d_input


class Conv3x3_n_to_n_padding:
    def __init__(self, output=1, input=1, activation=None, dtype=np.float32):
        num_filters = output
        in_ch = input
        self.activation = activation
        self.dtype = dtype

        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3, in_ch).astype(dtype) / 9
        self.biases = np.zeros(num_filters, dtype=dtype)

    def iterate_regions(self, image):
        image_padded = np.pad(image, 1, mode="constant", constant_values=0)[:, :, 1:-1]
        h, w = image_padded.shape[:2]
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image_padded[i:(i + 3), j:(j + 3), :]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, c = input.shape
        output = np.zeros((h, w, self.num_filters), dtype=self.dtype)
        for im_region, i, j in self.iterate_regions(input):
            for filter in range(self.num_filters):
                output[i, j, filter] = np.sum(im_region * self.filters[filter, :, :, :], axis=(0, 1, 2))
        self.last_output = output
        if self.activation is not None:
            output = self.activation(output)
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape, dtype=self.dtype)
        d_L_d_input = np.zeros(self.last_input.shape, dtype=self.dtype)

        if self.activation is not None:
            d_L_d_out = self.activation.backward(self.last_output) * d_L_d_out

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        for im_region, i, j in self.iterate_regions(d_L_d_out):
            for in_ch in range(d_L_d_input.shape[-1]):
                d_L_d_input[i, j, in_ch] += np.sum(
                    np.matmul(im_region[:, :, :], np.transpose(self.filters[:, :, :, in_ch], axes=(2, 0, 1))),
                    axis=(0, 1, 2))

        self.filters -= learn_rate * d_L_d_filters
        self.biases -= learn_rate * np.sum(d_L_d_out, axis=(0, 1))

        return d_L_d_input