#coding:utf-8
import tensorflow as tf
import numpy as np
#import os
import matplotlib.image as mpti
import xlrd
import xlwt

import forward

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99
IMAGE_PATH = './splitimages/'
MODEL_SAVE_PATH="./model/"
MODEL_NAME="model"

def get_label(sheet, index):
    n = int(sheet.cell(index, 1).value)
    label = [0]*11
    label[n] = 1
    return label
    
def get_input(sheet, index):
    img = mpti.imread(IMAGE_PATH + sheet.cell(index,0).value)
    return np.expand_dims(img, axis = 3)

def backward(sheet, num):
    x = tf.placeholder(tf.float32,[
	BATCH_SIZE,
	forward.IMAGE_HEIGH,
	forward.IMAGE_WIDTH,
	forward.NUM_CHANNELS]) 
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER) 

    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) # 经过softmax函数再求出交叉熵
    cem = tf.reduce_mean(ce)
    tf.add_to_collection('losses', cem)
    loss = tf.add_n(tf.get_collection('losses')) # 在损失函数中加入正则化

    learning_rate = tf.train.exponential_decay( # 指数衰减学习率
        LEARNING_RATE_BASE,
        global_step,
        STEPS/200,
        LEARNING_RATE_DECAY,
        staircase = False)
    
    #learning_rate = LEARNING_RATE_BASE

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs = []
            ys = []
            for j in range(BATCH_SIZE):
                index = (i*BATCH_SIZE+j)%num
                xs.append(get_input(sheet, index))
                ys.append(get_label(sheet, index))
                
            _, l_rate, label_loss, loss_value, step = sess.run([train_step, learning_rate, cem, loss, global_step], feed_dict={x: xs, y_: ys})
            if (i+1) % 200 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                print("The label loss is %g." % (label_loss))
                print("Current learning rate is %g." % (l_rate))
                
        saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME, global_step=global_step)


def main():
    train_data = xlrd.open_workbook(r'TrainDataInfo.xlsx')
    train_sheet = train_data.sheet_by_index(0)
    train_num = train_sheet.nrows

    backward(train_sheet, train_num)

if __name__ == '__main__':
    main()


