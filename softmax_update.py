# -*- coding: utf-8 -*-
"""softmax_update.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17K7axndjfO35JVjjktid8cMyGL-QjqdV
"""

import numpy as np

class Softmax:
    def __init__(self, dtype=np.float16):
        self.dtype = dtype
    
    def forward(self, input):
        self.last_totals = input
        shiftx = input - np.max(input)
        exp = np.exp(shiftx, dtype=self.dtype)
        out = exp / np.sum(exp, axis=0)
        return out
    
    def backprop(self, d_L_d_out):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals, dtype=self.dtype)
            S = np.sum(t_exp)
            if S == 0:
                S = 1
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            d_L_d_t = gradient * d_out_d_t
            return d_L_d_t