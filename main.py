#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/30 20:46
# @Author : ManQi
# @E-mail : zehuadu@126.com
# @Site : 
# @File : main.py
# @Software: PyCharm

import os
import struct
import numpy as np
import scipy.special
from DataProcessFun import *
from ManQiNeuralNetworks import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.01

    # 声明网络
    sess = Sess_NN()
    # 定义输入尺寸
    sess.input_node = input_nodes
    # 定义网络结构
    full_lay_1 = sess.fun_full_connect(input_nodes, [100, 0.0, pow(input_nodes, -0.5)])
    active_lay_1 = sess.fun_active(full_lay_1)
    full_lay_2 = sess.fun_full_connect(active_lay_1, [10, 0.0, pow(hidden_nodes, -0.5)])
    active_lay_2 = sess.fun_active(full_lay_2)
    # 网络结构初始化
    sess.fun_init_graph()

    # 读入训练数据
    x_train, y_train = LoadData('', kind='train')
    # 读入测试数据
    x_test, y_test = LoadData('', kind='t10k')
    test_num = y_test.shape[0]
    x_test_oned, _ = PreprocessDataSet(x_test, y_test)
    y_test = [int(i) for i in y_test]

    batch_size = 64
    for idx, epoch in enumerate(gen_epochs(40, x_train, y_train, batch_size=batch_size)):
        print('------------第 %d 轮迭代------------' % idx)
        for step, (X, Y) in enumerate(epoch):
            x_train_oned, y_train_onehot = PreprocessDataSet(X, Y)

            out = sess.train(x_train_oned)
            loss = y_train_onehot - out
            sess.fun_optimization(loss, 'GradientDescent', learning_rate)

            # 读入图片显示
            # for i in range(batch_size):
            #     plt.subplot(221 + i)
            #     img = X[i].reshape(28, 28)
            #     plt.imshow(img, cmap='Greys', interpolation='nearest')
            # plt.show()

            if step % 100 == 0:
                outputs = sess.predict(x_test_oned)
                label = np.argmax(outputs, axis=1)
                count = CountSameNum(label, y_test)
                print("第 %d 步，准确率为：%.4f: " % (step, count/test_num))
