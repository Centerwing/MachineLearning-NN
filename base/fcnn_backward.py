#coding:utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

import fcnn_forward
import fcnn_test
from fcnn_util import store_accuracy

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
DATA_PATH = './npz/grayData.npz'

REGULARIZER = 0.0001
STEPS = 10000
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model"
RESULT_PATH = "./result/"

param_to_change_and_value = 'BASE'


def backward(x_data, y_data):
    x_data = np.reshape(x_data, (x_data.shape[0], fcnn_forward.INPUT_NODE))/255 # 拉直x
    y_data = np.eye(y_data.shape[0], fcnn_forward.OUTPUT_NODE)[y_data].tolist()  # one_hot y

    x = tf.placeholder(tf.float32, [None, fcnn_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, fcnn_forward.OUTPUT_NODE])
    y = fcnn_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) # 经过softmax函数再求出交叉熵
    cem = tf.reduce_mean(ce)
    tf.add_to_collection('losses', cem)
    loss = tf.add_n(tf.get_collection('losses')) # 在损失函数中加入正则化
    #loss = tf.reduce_mean(ce)

    learning_rate = tf.train.exponential_decay( # 指数衰减学习率
        LEARNING_RATE_BASE,
        global_step,
        50,
        LEARNING_RATE_DECAY,
        staircase = False)
    
    #learning_rate = LEARNING_RATE_BASE

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver()

    training_step = []
    result = []
    los = []
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xds = []
            yds = []
            for j in range(BATCH_SIZE):
                s_index = (i*BATCH_SIZE+j) % x_data.shape[0]

                xds.append(x_data[s_index])
                yds.append(y_data[s_index])
                
            _, l_rate, label_loss, loss_value, step = sess.run([train_step, learning_rate, cem, loss, global_step], feed_dict={x: xds, y_: yds})
            if (i+1) % 200 == 0:
                saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME, global_step=global_step)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                print("The label loss is %g." % (label_loss))
                print("Current learning rate is %g." % (l_rate))

                accuracy = fcnn_test.main(step)

                training_step.append(step)
                los.append(loss_value)
                result.append(accuracy)

                store_accuracy(step, accuracy)

    plt.plot(training_step, result)
    plt.savefig(RESULT_PATH + "result.png")

    plt.clf()
    plt.plot(training_step, los)
    plt.savefig(RESULT_PATH + "loss.png")


def main():
    gray_data = np.load(DATA_PATH)
    x_data = gray_data['X_train']
    y_data = gray_data['Y_train']

    backward(x_data, y_data)
    for file in os.listdir('model'):
        os.remove(('model/' + file))

if __name__ == '__main__':
    main()


