#!/usr/bin/python
# coding:utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_PATH = os.path.join(BASE_DIR, 'train')
TEST_IMG_PATH = os.path.join(BASE_DIR, 'test')
VALUE_PATH = os.path.join(BASE_DIR, 'value')

x_train_imgs_name = os.listdir(TRAIN_IMG_PATH)
y_train = np.load(os.path.join(VALUE_PATH, 'y_train.npy'))
x_test_imgs_name = os.listdir(TEST_IMG_PATH)
y_test = np.load(os.path.join(VALUE_PATH, 'y_test.npy'))

print (len(x_train_imgs_name))
print (len(y_train))
print (len(x_test_imgs_name))
print (len(y_test))

print ('---------------- train ----------------')
# for i in xrange(0, len(x_train_imgs_name)):
for i in xrange(0, 10):
    img = cv2.imread(os.path.join(TRAIN_IMG_PATH, x_train_imgs_name[i]), 0)
    if img is not None:
        print ('%d/%d' % (i + 1, len(x_train_imgs_name)))
        ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print ret1
        plt.figure()
        plt.subplot(221), plt.imshow(img, 'gray')
        plt.subplot(222), plt.hist(img.ravel(), 256)  # .ravel方法将矩阵转化为一维
        plt.subplot(223), plt.imshow(th1, 'gray')
        plt.subplot(224), plt.hist(th1.ravel(), 256)
        # cv2.imshow('img', img)
        plt.show()
        # cv2.waitKey(0)
    else:
        print ('img is empty')
# write test images
# print ('---------------- test ----------------')
# for i in xrange(0, len(x_test_imgs_name)):
#     img = cv2.imread(os.path.join(TEST_IMG_PATH, x_test_imgs_name[i]), 0)
#     if img is not None:
#         print ('%d/%d' % (i + 1, len(x_test_imgs_name)))
#         # cv2.imshow('img', img)
#         # cv2.waitKey(0)
#     else:
#         print ('img is empty')
