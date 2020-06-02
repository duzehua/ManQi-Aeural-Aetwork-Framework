#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 21:26
# @Author : ManQi
# @E-mail : zehuadu@126.com
# @Site : 
# @File : mqnn_lay_cal.py
# @Software: PyCharm
"""
File description:
网络层计算，被 manqi_neural_networks.py 中同名功能函数所调用，当在 manqi_neural_networks.py 中添加新的功能函数 fun_new 时，
应同时在本文件中添加同名函数以确保程序正常运行，函数输入应包含两部分 input_ 和 argument ，input_ 为输入用于计算的输入，argument
包含当前功能函数的参数，以列表的形式传入。

manqi_neural_networks.py 中与该函数关联需对应的内容包括：
class ManQiNeuralNetwork:

    def __init__(self):
        self.opti_true_flag_l: 若参数可调，应在 manqi_neural_networks.py 同名函数中定义标识符

    def fun_new(self, input_, argument=[]): -> 同名函数，函数基本结构如下
        self.GRAPH.append(fun_new)
        self.ARGUMENT.append(['标识符', argument])
"""


import numpy as np
import scipy.special


def fun_full_connect_(input_, argument=[]):
    """
    全连接层运算
    :param input_: 需要计算的量
    :param argument: 权重与偏置
    :return: 计算结果
    """

    '此处实际上还需要检查各个变量是否为数值'

    return np.dot(argument, input_)  # hidden_lay1 = w * x


def fun_active_(input_, argument=[]):
    return scipy.special.expit(input_)