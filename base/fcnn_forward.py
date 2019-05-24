#coding:utf-8
import tensorflow as tf

INPUT_NODE = 1380
OUTPUT_NODE = 11
LAYER1_NODE = 300

def get_weight(shape, regularizer): # 加入正则化参数
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) # 计算正则化
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))  
    return b

def forward(x, regularizer): # 前向传播
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y