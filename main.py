#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 2018

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

from __future__ import print_function
import sys
import numpy as np
import codecs as cs
from var import VectorAutoRegression

DATA_SOURCE = r"./data"
N = 1
SCALE = 1000


def read_data():
    with cs.open(DATA_SOURCE, 'r', 'utf-8') as src:
        stream = src.readlines()[1:]
        data = []
        for line in stream:
            sum_price, date = line.strip().split()
            data.append(float(sum_price))
        return data


def make_training_date(data, log_phaseP=1):
    x = [[] for p in range(log_phaseP)]
    y = []
    for i in range(len(data)):
        y.append(data[i])
        for j in range(log_phaseP):
            offset = i - (j + 1)
            if 0 <= offset and offset < len(data):
                x[j].append(data[offset])
            else:
                x[j].append(0.0)
    return x, y


def main(args):
    log_phaseP = int(args[1]) if len(args) >= 2 else 1
    data = read_data()
    x, y = make_training_date(data, log_phaseP)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape((len(data), N, 1))
    x = x.reshape((log_phaseP, len(data), N, 1))
    x = x / SCALE  # scale
    y = y / SCALE  # scale
    model = VectorAutoRegression(n=N, P=log_phaseP)
    model.fit(x, y, 1e-5, 1000, 1e-4)
    predict = model.predict(x)
    predict *= SCALE
    with cs.open("./pred.txt", 'w', 'utf-8') as out:
        for i in range(len(data)):
            out.write("%f\n" % (predict[i, 0, 0]))

if __name__ == '__main__':
    main(sys.argv)
