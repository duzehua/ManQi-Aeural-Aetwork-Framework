#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/05/11 15:35
# @Author : ManQi
# @Site :
# @File : manqi_neural_networks.py
# @Software: PyCharm
"""
File description:
自写神经网络的主体部分，调用各个模块共同实现网络的神经网络构建、训练与测试，主要使用步骤应包括：
1、声明网络 sess = ManQiNeuralNetwork()
2、定义输入数据的尺寸 sess.input_node
3、网络结构构建
    lay1 = sess.lay_fun_a(input)
    lay2 = sess.lay_fun_b(lay1)
    ...
    laym = sess.lay_fun_y(layl)
    out = sess.lay_fun_z(laym)
4、网络初始化 sess.fun_init_graph()
5、网络训练
    out = sess.train(input)
    loss = label - out
    sess.fun_optimization(loss, '参数更新标识符', 学习率)
6、网络预测
    out = sess.predict(input)
"""

import numpy as np
from mqnn_lay_cal import *
from mqnn_net_init import *
from mqnn_net_para_update import *


class ManQiNeuralNetwork:
    def __init__(self):
        self.GRAPH = []  # 网络结构
        self.ARGUMENT = []  # 存放每一层的结构信息，如当前层含义、节点数等内容
        self.ARGUMENT_L = []  # 根据ARGUMENT生成的张量参数
        self.input = []  # 输入数据
        self.input_node = []  # 输入层的结构信息
        self.loss = []  # one-hot损失
        self.learning_rate = 0.01  # 学习率，默认0.01
        self.FP_result = []  # 前向计算结果
        self.opti_flag_l = []  # 层参数可调标志位
        self.opti_true_flag_l = ['full_connect']
        self.init_switch = {'full_connect': init_full_connect_lay,
                            'init_nn_lay': init_nn_lay_default}
        self.parameter_update = {'GradientDescent': parameter_update_gradient_descent,
                                 'para_update': parameter_update_default}

    def ForwardPropagation(self, input_):
        """
        神经网络的训练过程，包括前向传播、反向传播(误差计算与优化)过程。返回当前步骤更新完成的网络。
        :param input_: 输入数据
        :return:
        """
        deep_nn = len(self.GRAPH)
        # 检查输入格式
        self.input = input_
        out = []
        try:
            out = self.GRAPH[0](input_, self.ARGUMENT_L[0])
        except:
            print('请检查网络是否成功初始化！！！')
        self.FP_result.append(out)
        if deep_nn > 1:
            for i in range(1, deep_nn):
                out = self.GRAPH[i](out, self.ARGUMENT_L[i])
                self.FP_result.append(out)
        return out

    def train(self, input_):
        """
        神经网络的训练过程
        :param input_: 输入的训练样本数据集
        :return:
        """
        input_ = input_.T
        out = self.ForwardPropagation(input_)
        out = out.T
        return out

    def predict(self, input_):
        """
        神经网络的预测过程
        :param input_: 输入的测试样本数据集
        :return:
        """
        input_ = input_.T
        out = self.ForwardPropagation(input_)
        out = out.T
        self.FP_result = []
        return out

    def fun_loss(self, input_):
        """
        检查输入的损失是否格式正确(数值)，同时，赋值给self.loss
        :param input_: 损失值 one-hot损失
        :return:
        """
        self.loss = input_

    def fun_optimization(self, input_, para_update_type, learning_rate):
        """
        根据输入的损失值，计算每一层的误差，同时应用指定的优化方法进行网络参数优化
        :param input_: 应包含：[损失，优化方法，...]，注意：此处损失不是一个数，是one-hot损失
        :param para_update_type: 参数更新类型
        :param learning_rate: 学习率
        :return:
        """
        self.learning_rate = learning_rate
        assert input_.any(), '损失值缺失'
        '误差计算'
        Error_L = [input_.T]
        Error_temp = Error_L
        len_ARGL = len(self.ARGUMENT_L)
        for i in range(len_ARGL - 1, 0, -1):
            if self.opti_flag_l[i]:
                curt_Error = np.dot(self.ARGUMENT_L[i].T, Error_temp[0])
                Error_L = [curt_Error] + Error_L
                Error_temp = [curt_Error]
            else:
                Error_L = [[]] + Error_L
        '参数更新'
        self.ARGUMENT_L = self.parameter_update.get(para_update_type, parameter_update_default)(Error_L, self.FP_result,
                                                                                                self.opti_flag_l,
                                                                                                self.ARGUMENT_L,
                                                                                                self.input,
                                                                                                learning_rate)
        self.FP_result = []

    def fun_init_graph(self):
        """
        对ARGUMENT数组进行初始化，构建初始权重参数等内容，存放于ARGUMENT_L中
        为识别ARGUMENT中初始化应进行的操作，应对每一个元素添加标志位以明确操作类型
        实现过程包括，对ARGUMENT数组进行迭代，分析每个元素的标志位并进行的操作
        :return:
        """
        len_ARG = len(self.ARGUMENT)

        for i in range(len_ARG):
            cur_arg = self.ARGUMENT[i]
            if cur_arg[0]:
                cur_arg_data = cur_arg[1]
                flag_ = cur_arg[0]
                last_arg_data = []
                for j in range(i - 1, -1, -1):
                    last_arg = self.ARGUMENT[j]
                    len_last_arg = len(last_arg)
                    if len_last_arg == 2:
                        if last_arg[1]:
                            last_arg_data = last_arg[1][0]
                            break
                        else:
                            pass
                    else:
                        pass
                if not last_arg_data:
                    last_arg_data = self.input_node
                cur_arg_data_ = self.init_switch.get(flag_, init_nn_lay_default)([cur_arg_data, last_arg_data])
            else:
                flag_ = []
                cur_arg_data_ = 1.0

            '确定是否为可调参数，修改标志位'
            if (flag_ in self.opti_true_flag_l) and (type(cur_arg_data_).__name__ != 'int'):
                self.opti_flag_l.append(1)
            else:
                self.opti_flag_l.append(0)

            self.ARGUMENT_L.append(cur_arg_data_)  # 此处去掉了[]，可能后面出错

    def input_layer(self, input_):
        """
        定义输入层信息，定义输入格式用于检查实际输入是否正确，同时配置参数用于初始化
        :param input_: 输入的尺寸，目前只支持一维，即数据长度
        :return:
        """
        self.input_node = input_

    def fun_full_connect(self, input_, argument=[]):
        """
        全连接层的实现，主要负责将全连接层运算和相关参数添加到会话中
        :param input_: 预留输入
        :param argument: 参数配置，默认为空[] 或 自定义时需输入[节点数, 权重均值, 权重标准差]
        :return:
        """
        # assert ((len(argument) == 0) or argument.any()), '输入的argument维度不匹配'
        self.GRAPH.append(fun_full_connect_)
        self.ARGUMENT.append(['full_connect', argument])

    def fun_active(self, input_, argument=[]):
        """
        激活层的实现，对输入进行非线性变换
        :param input_: 输入
        :param argument: 参数配置，暂定为空[]，后期扩展添加激活方法选择
        :return:
        """
        self.GRAPH.append(fun_active_)
        self.ARGUMENT.append([argument])
