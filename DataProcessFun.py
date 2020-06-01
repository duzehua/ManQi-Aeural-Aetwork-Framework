#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 0:17
# @Author : ManQi
# @E-mail : zehuadu@126.com
# @Site : 
# @File : DataProcessFun.py
# @Software: PyCharm

import os
import struct
import numpy as np

def LoadData(path, kind='train'):
    """从路径 `path`中加载数据集"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def TurnToOneHotLabel(labelSet):
    n_class = 10  # np.max(labelSet)
    len_label = len(labelSet)
    oneHotMat = np.mat(np.zeros((len_label, n_class)))

    for i in range(len_label):
        oneHotMat[i, labelSet[i]] = 1

    return oneHotMat


def CountSameNum(list1, list2):
    '统计相同数据的个数'
    list1 = list(list1)
    list2 = list(list2)
    assert len(list1) == len(list2), 'CountSameNum函数中，输入数组长度不一致'

    count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            count += 1
    return count



def PreprocessDataSet(DataX, DataY):
    '对从二进制文件直接解析读取的数据集进行特征样本的归一化和标签的0-1化'
    DataX = DataX.astype(np.float32)
    DataX_shape = DataX.shape

    rows_max = DataX.max(axis=1).reshape([DataX_shape[0], 1])
    rows_max_mat = np.tile(rows_max, (1, DataX_shape[1]))

    rows_min = DataX.min(axis=1).reshape([DataX_shape[0], 1])
    rows_min_mat = np.tile(rows_min, (1, DataX_shape[1]))

    DataX_oned = (DataX - rows_min_mat) / (rows_max_mat - rows_min_mat)
    DataY_onehot = TurnToOneHotLabel(DataY)

    return np.array(DataX_oned, ndmin=2), np.array(DataY_onehot, ndmin=2)


def gen_batch(dataSet, labelSet, batch_size):
    row, col = dataSet.shape
    n = row // batch_size

    for i in range(n):
        x = dataSet[i * batch_size: (i + 1) * batch_size, :]
        y = labelSet[i * batch_size: (i + 1) * batch_size]
        yield (x, y)


def gen_epochs(n, dataSet, labelSet, batch_size):
    for _ in range(n):
        row, col = dataSet.shape
        index = np.arange(row)
        np.random.shuffle(index)
        dataSet = dataSet[index, :]
        labelSet = labelSet[index]
        yield gen_batch(dataSet, labelSet, batch_size)