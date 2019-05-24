#coding:utf-8
import tensorflow as tf
import numpy as np
import fcnn_dropout_forward
import fcnn_dropout_backward
from fcnn_dropout_util import plot_roc, store_report


def test(x_data, y_data, step):

    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fcnn_dropout_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None])
        y = fcnn_dropout_forward.forward(x, None, False)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.cast(y_, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        saver = tf.train.Saver()

        result = None
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(fcnn_dropout_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                xs = np.reshape(x_data, (x_data.shape[0], fcnn_dropout_forward.INPUT_NODE))
                ys = y_data
                        
                accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                result = accuracy_score

                if step == fcnn_dropout_backward.STEPS:
                    y_pred = sess.run(y, feed_dict={x:xs, y_:ys})
                    plot_roc(y_data, y_pred)
                    store_report(y_data, y_pred)

                
                print("There're %d test samples" % x_data.shape[0])
                print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
        
        return result


def main(step):
    gray_data = np.load(fcnn_dropout_backward.DATA_PATH)
    x_data = gray_data['X_test']
    y_data = gray_data['Y_test']

    return test(x_data, y_data, step)


if __name__ == '__main__':
    main()
