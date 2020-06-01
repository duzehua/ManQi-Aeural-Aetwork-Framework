#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/0511 15:35
# @Author : ManQi
# @Site :
# @File : ManQiNeuralNetworks.py
# @Software: PyCharm

import numpy as np
import scipy.special

def fun_sum_(input_, argument=[]):
    return sum(input_)

def fun_mul_(input_, argument=[]):
    return 2 * input_

def fun_full_connect_(input_, argument=[]):
    '''
    全连接层运算
    :param input_: 需要计算的量
    :param argument: 权重与偏置
    :return: 计算结果
    '''

    # if len(argument) == 1:
    #     argument = 1
    # else:
    #     pass
    '此处实际上还需要检查各个变量是否为数值'
    # pass

    return np.dot(argument, input_)  # hidden_lay1 = w * x

def fun_active_(input_, argument=[]):
    return scipy.special.expit(input_)

def init_full_connect_lay(input_):
    '''
    实现全连接层的初始化
    :param input_: 输入，包含：[当前层的节点信息，前面层的节点信息]
    :return:
    '''
    cur_arg_data, last_arg_data = input_
    if cur_arg_data[0]:
        # 权值矩阵(输入层 到 隐藏层 条链)，分布中心为 0、标准差 1/sqrt（i_nodes）、大小 当前层节点数 * 上一层节点数
        '[全连接层标志位(full_connect)，节点数0，分布中心1，权重标准差2]'
        w = np.random.normal(cur_arg_data[1], cur_arg_data[2], (last_arg_data, cur_arg_data[0])).T
        # w = cur_arg_data[2].T
    else:
        w = 1.0
    return w
    # pass

def init_nn_lay_default(input_):
    pass

class Sess_NN:
    '''
    神经网络构建，训练与测试
    实现步骤：
    1、声明网络
    2、网络结构构建
    3、网络初始化
    4、网络训练
    5、网络测试
    '''
    def __init__(self):
        self.GRAPH = []  # 网络结构
        self.ARGUMENT = []  # 存放每一层的结构信息，如当前层含义、节点数等内容
        self.ARGUMENT_L = []  # 根据ARGUMENT生成的张量参数
        # self.struct_check_list = []
        # self.struct_check_flag = 0
        self.input = []  # 输入数据
        self.input_node = []  # 输入层的结构信息
        self.loss = []  # one-hot损失
        self.learning_rate = 0.01  # 学习率，默认0.01
        self.FP_result = []
        self.opti_flag_l = []  # 可调参标志位，用于误差计算和优化过程，参数可调置为1，否则置为0。目前，主要对全连接及卷积层等。初始化时，根据类型标识符进行定值。
        self.opti_true_flag_l = ['full_connect']
        # self.input_y = []
        self.init_switch = {'full_connect': init_full_connect_lay,
                            'init_nn_lay': init_nn_lay_default}  # 结构层对应的初始化方法

    def ForwardPropagation(self, input_):
        '''
        神经网络的训练过程，包括前向传播、反向传播(误差计算与优化)过程。返回当前步骤更新完成的网络。
        :param input_: 输入数据
        :return:
        '''
        deep_nn = len(self.GRAPH)
        # 检查输入格式
        self.input = input_
        try:
            out = self.GRAPH[0](input_, self.ARGUMENT_L[0])  # 后面此处的第二个输入应改为ARGUMENT_L
        except:
            print('请检查网络是否初始化，或检查权重参数是否正确！')
        # if not self.ARGUMENT_L[0] == 1:
        #     self.FP_result.append(out)
        # else:
        #     self.FP_result.append([])
        self.FP_result.append(out)
        if deep_nn > 1:
            for i in range(1, deep_nn):
                out = self.GRAPH[i](out, self.ARGUMENT_L[i])  # 后面此处的第二个输入应改为ARGUMENT_L
                self.FP_result.append(out)
                # if not self.ARGUMENT_L[i] == 1:
                #     self.FP_result.append(out)
                # else:
                #     self.FP_result.append([])


        return out

    def train(self, input_):
        '''
        神经网络的训练过程，包括前向传播、反向传播(误差计算与优化)过程。返回当前步骤更新完成的网络。
        :param input_: 输入

        :return:
        '''

        # deep_nn = len(self.GRAPH)

        # 检查输入格式
        # pass

        '正向传播'
        input_ = input_.T
        out = self.ForwardPropagation(input_)
        out = out.T
        return out

    def predict(self, input_):
        '''
        神经网络的训练过程，包括前向传播、反向传播(误差计算与优化)过程。返回当前步骤更新完成的网络。
        :param input_: 输入
        :return:
        '''

        # deep_nn = len(self.GRAPH)

        # 检查输入格式
        # pass

        '正向传播'
        input_ = input_.T
        out = self.ForwardPropagation(input_)
        out = out.T
        self.FP_result = []

        return out

    def fun_loss(self, input_):
        '''
        检查输入的损失是否格式正确(数值)，同时，赋值给self.loss
        :param input_: 损失值
        :return:
        '''
        # 检查输入格式，注意：此处损失不是一个数，是one-hot损失！！！！！！！！！！！！！！
        # pass

        # 赋值
        self.loss = input_

    def fun_optimization(self, input_, learning_rate):  # 后期更改为根据不同的优化函数，进行不同的优化，即把优化过程函数化
        '''
        根据输入的损失值，计算每一层的误差，同时应用指定的优化方法进行网络参数优化
        :param input_: 应包含：[损失，优化方法，...]，注意：此处损失不是一个数，是one-hot损失！！！！！！！！！！！！！！
        :param learning_rate: 学习率
        :return:
        '''

        self.learning_rate = learning_rate
        # 检查损失值是否存在
        assert input_.any(), '损失值缺失'
        # 计算每一层误差

        # '--反向传播计算误差'
        # hidden_lay2_errors = targets - hidden_lay2_active  # 计算输出层误差
        #
        # # 反向传播法计算隐层误差，隐层误差=输出层误差点乘权重E=W.E'
        # hidden_lay1_errors = np.dot(self.w_h_to_o.T, hidden_lay2_errors)

        '输出层误差为输入，从倒数第一个结构开始前推，若当前层张量参数为1(即表示为激活或池化等)，则将上一个误差结果复制到当前层继续前推到第一个张量参数为止，注意！！！！！！！！！！！有可能递推到第二个，具体看输出结果'
        '各层损失值计算'
        Error_L = [input_.T]  # 输出层误差
        Error_temp = Error_L  # 误差暂存，用于计算前一个误差
        len_ARGL = len(self.ARGUMENT_L)
        for i in range(len_ARGL - 1, 0, -1):
            # if self.ARGUMENT_L[i] == 1:
            #     Error_L.append(Error_L[-1])  # 注意，Error_L是正序排列的，不能搞混
            # else:
            #     Error_L.append(np.dot(self.ARGUMENT_L[i], Error_L[-1]))
            if self.opti_flag_l[i]:  # 由于以输出误差作为Error_L最后一位，因此前一层的误差与当前层索引一致，Error_L直接更新即可，但若报错，后期需检查！！！！！
                # Error_L = [np.dot(self.ARGUMENT_L[i], Error_L[-1]), Error_temp]  # 注意，不能搞混，
                curt_Error = np.dot(self.ARGUMENT_L[i].T, Error_temp[0])
                Error_L = [curt_Error] + Error_L  # 注意，不能搞混，
                Error_temp = [curt_Error]
            else:
                Error_L = [[]] + Error_L

        '梯度下降更新参数'
        Error_temp = Error_L[-1]
        curt_lay_result = self.FP_result[-1]  # 当前层结果
        for i in range(len_ARGL - 1, -1, -1):
            # print(i)
            if self.opti_flag_l[i]:
                if i == 0:
                    last_result = self.input  # 输入
                else:
                    last_result = self.FP_result[i - 1]# 前一层结果，需判断是否为第一层，第一层应将输入导入
                self.ARGUMENT_L[i] += self.learning_rate * np.dot((Error_temp * curt_lay_result * (1.0 - curt_lay_result)), np.transpose(last_result))
                curt_lay_result = last_result
                Error_temp = Error_L[i - 1]  # 后面的误差给前面用
            else:
                pass

        self.FP_result = []
        '层紧跟的激活层不单独作为一层，与前一项作为同一层'
        '!!!!!!!!!!!需要全局注意的点，仅对需要调整参数的层进行处理，因此，如激活层或者池化层，在计算函数中，应将当前层self.FP_result' \
        '置空，在计算损失和优化参数时，当前层不操作！！！！！！！！！！！！！！！---------》》》》对应检查每一个计算函数 '
        # self.w_h_to_o += self.lr * np.dot((hidden_lay2_errors * hidden_lay2_active * (1.0 - hidden_lay2_active)),
        #                                   np.transpose(hidden_lay1_active))
        # pass

    def fun_init_graph(self):
        '对ARGUMENT数组进行初始化，构建初始权重参数等内容，存放于ARGUMENT_L中'
        '为识别ARGUMENT中初始化应进行的操作，应对每一个元素添加标志位以明确操作类型'
        '实现过程包括，对ARGUMENT数组进行迭代，分析每个元素的标志位并进行的操作'
        len_ARG = len(self.ARGUMENT)
        for i in range(len_ARG):
            cur_arg = self.ARGUMENT[i]
            if cur_arg[0]:
                cur_arg_data = cur_arg[1]
                flag_ = cur_arg[0]
                # 为权重初始化，需调用前面层的结点数
                last_arg_data = []
                for j in range(i - 1, -1, -1):
                    last_arg = self.ARGUMENT[j]
                    len_last_arg = len(last_arg)  # 此处后期需优化，last_arg 长度为2，且第二项非空
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

            # 确定是否为可调参数，修改标志位
            if (flag_ in self.opti_true_flag_l) and (type(cur_arg_data_).__name__ != 'int'):
                self.opti_flag_l.append(1)
            else:
                self.opti_flag_l.append(0)

            self.ARGUMENT_L.append(cur_arg_data_)  # 此处去掉了[]，可能后面出错

    def fun_sum(self, input_, argument=[]):
        self.GRAPH.append(fun_sum_)
        # self.struct_check_list.append([input_])
        self.ARGUMENT.append([argument])

    def fun_mul(self, input_, argument=[]):
        self.GRAPH.append(fun_mul_)
        # self.struct_check_list.append([input_])
        self.ARGUMENT.append([argument])

    def input_layer(self, input_):
        '''
        定义输入层信息，定义输入格式用于检查实际输入是否正确，同时配置参数用于初始化
        :param input_: 输入的尺寸，目前只支持一维，即数据长度
        :return:
        '''
        self.input_node = input_
        # pass

    def fun_full_connect(self, input_, argument=[]):
        '''
        全连接层的实现，主要负责将全连接层运算和相关参数添加到会话中
        :param input_: 预留输入
        :param argument: 参数配置，默认为空[] 或 自定义时需输入[节点数, 权重均值, 权重标准差]
        :return:
        '''
        # assert ((len(argument) == 0) or argument.any()), '输入的argument维度不匹配'
        self.GRAPH.append(fun_full_connect_)
        # self.struct_check_list.append([input_])
        self.ARGUMENT.append(['full_connect', argument])

    def fun_active(self, input_, argument=[]):
        '''
        激活层的实现，对输入进行非线性变换
        :param input_: 输入
        :param argument: 参数配置，暂定为空[]，后期扩展添加激活方法选择
        :return:
        '''
        self.GRAPH.append(fun_active_)
        # self.struct_check_list.append([input_])
        self.ARGUMENT.append([argument])


if __name__ == '__main__':
    nn = Sess_NN()
    a = []
    b = []
    '结构定义的输入输出不参与判断，只用于逻辑的理顺'
    fun1 = nn.fun_sum([a, b])
    fun2 = nn.fun_mul(fun1)
    fun3 = nn.fun_full_connect(fun2)





