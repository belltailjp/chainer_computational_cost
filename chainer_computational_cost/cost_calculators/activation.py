#!/usr/bin/env python
# -*- coding: utf-8 -*-

def calc_activation(function, in_data, **kwargs):
    x, = in_data
    return (x.size, x.size, x.size)


def calc_prelu(function, in_data, **kwargs):
    x, W = in_data
    return (x.size, x.size + W.size, x.size)
