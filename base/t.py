import os
import matplotlib.image as mpti
import numpy as np
import xlrd
import xlwt

DATA_PATH = './npz/grayData.npz'

gray_data = np.load(DATA_PATH)
xtrain = gray_data['X_train']
ytrain = gray_data['Y_train']
xtest = gray_data['X_test']
ytest = gray_data['Y_test']

x = []
x_ = []
y = []
y_ = []
xn = [0]*11

for i in range(len(xtrain)):
    xn[ytrain[i]] += 1
    if xn[ytrain[i]] < 149:
        x.append(xtrain[i])
        y.append(ytrain[i])

xn = [0]*11
for j in range(len(xtest)):
    xn[ytest[j]] += 1
    if xn[ytest[j]] < 47:
        x_.append(xtest[j])
        y_.append(ytest[j])

np.savez('dataz.npz', X_train=x, Y_train=y, X_test=x_, Y_test=y_)






'''
y = []

for i in range(y_data.shape[0]):
    ys = [0]*11
    ys[y_data[i]] = 1
    y.append(ys)

print(y)

np.save('all_y.npy', y)
'''
'''
imgs = os.listdir('./splitimages')
img_num = len(imgs)

print(img_num)

img = mpti.imread('./splitimages/' + imgs[0])
print(img.shape)

print(imgs[1000])
print(imgs[1000].split('x')[1][0])
print(np.reshape(img,(1, 1380)).shape)
'''
'''
print(img)
print('切片')
print(img[:,0:30,:])
print('拉伸')
print(np.reshape(img[:,0:30,:],(1,4140))[0])
'''
