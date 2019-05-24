import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os
import fcnn_dropout_backward

N_CLASSES = 11
lw = 2
labels = list(range(10)).append('_')

def plot_roc(y_data, y_pred):
    fpr = {}
    tpr = {}
    roc_auc = {}
    ysoftmax = tf.nn.softmax(y_pred)
    y_onehot = tf.one_hot(y_data, N_CLASSES, 1, 0)
    if 'images' not in os.listdir():
        os.makedirs('images')

    with tf.Session() as sess:
        y_labels, y_softmax = sess.run([y_onehot, ysoftmax])

    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_softmax[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(y_labels.ravel(), y_softmax.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(11)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= N_CLASSES
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('images/micro_macro_' + fcnn_dropout_backward.param_to_change_and_value + '.png')
    plt.clf()
    for i in range(N_CLASSES):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('images/every_class_' + fcnn_dropout_backward.param_to_change_and_value + '.png')

def store_report(y_data, y_pred):
    if 'report_files' not in os.listdir():
        os.makedirs('report_files')
    report_file = open('report_files/' + fcnn_dropout_backward.param_to_change_and_value + '_' + 'report.txt', 'a+')
    y_pred = np.argmax(y_pred, axis=1)
    report = classification_report(y_data, y_pred, labels=labels)
    report_file.write(report + '\n')
    report_file.close()

def store_accuracy(step, accuracy):
    if 'accuracy_files' not in os.listdir():
        os.makedirs('accuracy_files')
    accuracy_file = open('accuracy_files/' + fcnn_dropout_backward.param_to_change_and_value + '.txt', 'a+')
    accuracy_file.write(str(step) + ' ' + str(accuracy) + '\n')
    accuracy_file.close()