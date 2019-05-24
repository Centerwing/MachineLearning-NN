#coding:utf-8
import tensorflow as tf
import numpy as np
#import os
import matplotlib.image as mpti
import xlrd
import xlwt

import forward
import backward

def test(sheet, num):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[
        num,
        forward.IMAGE_HEIGH,
        forward.IMAGE_WIDTH,
        forward.NUM_CHANNELS]) 
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                xs = []
                ys = []
                for index in range(num):
                    xs.append(backward.get_input(sheet, index))
                    ys.append(backward.get_label(sheet, index))
                        
                accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                print("There're %d test samples" % num)
                print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')


def main():
    test_data = xlrd.open_workbook(r'TestDataInfo.xlsx')
    test_sheet = test_data.sheet_by_index(0)
    test_num = test_sheet.nrows
    
    test(test_sheet, test_num)

if __name__ == '__main__':
    main()
