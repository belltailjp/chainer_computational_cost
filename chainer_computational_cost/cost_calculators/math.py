#!/usr/bin/env python
# -*- coding: utf-8 -*-


def calc_eltw_op(function, in_data, **kwargs):
    x = in_data[0]
    return (x.size, x.size * 2, x.size)
