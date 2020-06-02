#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 21:25
# @Author : ManQi
# @E-mail : zehuadu@126.com
# @Site : 
# @File : mqnn_net_para_update.py
# @Software: PyCharm
"""
File description:
参数更新计算，被 manqi_neural_networks.py 中 fun_optimization 函数所调用，如需添加新的参数更新方式，则应在 manqi_neural_networks.py
的 self.parameter_update 中，以键值对的方式分别添加标识符和函数名 fun_new (如: 'Adam': parameter_update_adam)，其中标识符为
主函数脚本中作为实参标识更新方式的输入，parameter_update_adam 为该文件中同名函数。

如需添加新的参数更新方式，manqi_neural_networks.py 中与该函数关联需对应的内容包括：
class ManQiNeuralNetwork:

    def __init__(self):

    self.parameter_update = {...,
                            '标识符': fun_new，
                            'para_update': parameter_update_default}
    其中，self.parameter_update 中 'para_update': parameter_update_default 为预留的默认操作

为匹配大部分优化操作，fun_new 的形参输入按顺序包含以下内容：
    :param error_l: 每层网络的误差
    :param fp_result_l: 网络每层的前向计算结果
    :param update_flag_l: 参数更新标志位
    :param parameter_l: 待更新的参数列表
    :param nn_input: 神经网络的输入
    :param lr: 学习率

完成参数更新后，需将更新得到的参数列表 parameter_l 返回
"""

import numpy as np


def parameter_update_gradient_descent(error_l, fp_result_l, update_flag_l, parameter_l, nn_input, lr):
    """
    梯度下降法更新神经网络参数
    :param error_l: 每层网络的误差
    :param fp_result_l: 网络每层的前向计算结果
    :param update_flag_l: 参数更新标志位
    :param parameter_l: 待更新的参数列表
    :param nn_input: 神经网络的输入
    :param lr: 学习率
    :return: 更新后的网络参数
    """
    len_ARGL = len(parameter_l)
    Error_temp = error_l[-1]
    curt_lay_result = fp_result_l[-1]  # 当前层结果
    for i in range(len_ARGL - 1, -1, -1):
        # print(i)
        if update_flag_l[i]:
            if i == 0:
                last_result = nn_input  # 输入
            else:
                last_result = fp_result_l[i - 1]  # 前一层结果，需判断是否为第一层，第一层应将输入导入
            parameter_l[i] += lr * np.dot((Error_temp * curt_lay_result * (1.0 - curt_lay_result)),
                                          np.transpose(last_result))
            curt_lay_result = last_result
            Error_temp = error_l[i - 1]  # 后面的误差给前面用
        else:
            pass
    return parameter_l


def parameter_update_default(error_l, fp_result_l, update_flag_l, parameter_l, nn_input, lr):
    """
    当不指定参数更新方式时，理应报错
    :param error_l:
    :param fp_result_l:
    :param update_flag_l:
    :param parameter_l:
    :param nn_input:
    :param lr:
    :return:
    """
    assert '参数更新失败，缺少必要的参数优化信息，请检查输入！'