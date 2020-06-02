#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 21:23
# @Author : ManQi
# @E-mail : zehuadu@126.com
# @Site : 
# @File : mqnn_net_init.py
# @Software: PyCharm
"""
File description:
参数初始化操作，被 manqi_neural_networks.py 中 fun_init_graph 函数所调用，如在 mqnn_lay_cal.py 已添加新的网络计算函数 new_fun_cal
，且该函数的参数可更新，需首先在 manqi_neural_networks.py 的 self.init_switch 中，以键值对的方式分别添加标识符和函数名 fun_new_int
(如: 'cnn': init_cnn)，其中标识符与 mqnn_lay_cal.py 中 new_fun_cal 标识符一致，便于程序网络在初始化时能够匹配查找到 new_fun_cal
所需要进行的初始化操作 fun_new_int。fun_new_int 为该文件中同名函数。

如需对参数可更新的 new_fun_cal 添加初始化，manqi_neural_networks.py 中与该函数关联需对应的内容包括：
class ManQiNeuralNetwork:

    def __init__(self):

    self.init_switch = {...,
                        '标识符': fun_new_int，
                        'init_nn_lay': init_nn_lay_default}
    其中，self.init_switch 中 'init_nn_lay': init_nn_lay_default 为预留的默认操作

为匹配大部分初始化操作，fun_new_int 的形参以单个列表的形式传入函数，根据函数指定的数据进行初始化操作。
"""

import numpy as np


def init_full_connect_lay(input_):
    """
    实现全连接层的初始化
    :param input_: 输入，包含：[当前层的节点信息，前面层的节点信息]
    当前层的节点信息 cur_arg_data => [节点数, 权重均值, 权重标准差]
    前面层的节点信息 last_arg_data => 前一层的节点数
    :return:
    """
    cur_arg_data, last_arg_data = input_
    if cur_arg_data[0]:
        w = np.random.normal(cur_arg_data[1], cur_arg_data[2], (last_arg_data, cur_arg_data[0])).T
    else:
        w = 1.0
    return w


def init_nn_lay_default(input_):
    assert '网络初始化失败，缺少必要的网络结构参数，请检查网络！'