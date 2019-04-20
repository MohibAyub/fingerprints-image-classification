#!/usr/bin/python
# coding:utf-8

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_IMG_PATH = os.path.join(BASE_DIR, 'task2img', 'backgroundimg')
PORES_IMG_PATH = os.path.join(BASE_DIR, 'task2img', 'poresimg')
TRAIN_IMG_PATH = os.path.join(BASE_DIR, 'train')
TEST_IMG_PATH = os.path.join(BASE_DIR, 'test')
VALUE_PATH = os.path.join(BASE_DIR, 'value')

background_imgs = os.listdir(BACKGROUND_IMG_PATH)  # image name
pores_imgs = os.listdir(PORES_IMG_PATH)  # image name
print ('-----------------------------------------------')
print ('                DIP Project 2                  ')
print ('-----------------------------------------------')
print ('background images have %d' % len(background_imgs))
print ('pores images have %d' % len(pores_imgs))
print ('\n')
# split test and train data set by using np.random.permutation
print ('----------------start to split data set----------------')
# b_shuffle_indexes = np.random.permutation(len(background_imgs))
# p_shuffle_indexes = np.random.permutation(len(pores_imgs))
# test_ratio = 0.2
# b_test_size = int(len(background_imgs) * test_ratio)
# p_test_size = int(len(pores_imgs) * test_ratio)
#
# b_test_indexes = b_shuffle_indexes[:b_test_size]
# b_train_indexes = b_shuffle_indexes[b_test_size:]
#
# p_test_indexes = p_shuffle_indexes[:p_test_size]
# p_train_indexes = p_shuffle_indexes[p_test_size:]
#
# x_train = [background_imgs[x] for x in b_train_indexes] + [pores_imgs[x] for x in p_train_indexes]
# y_train = [0] * len(b_train_indexes) + [1] * len(p_train_indexes)
#
# x_test = [background_imgs[x] for x in b_test_indexes] + [pores_imgs[x] for x in p_test_indexes]
# y_test = [0] * len(b_test_indexes) + [1] * len(p_test_indexes)
x1 = background_imgs
x2 = pores_imgs
y1 = [0] * len(background_imgs)
y2 = [1] * len(pores_imgs)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.2)  # background image name
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2)  # pores image name
x_train = []  # image path
x_test = []
for img in x_train1:
    x_train.append(os.path.join(BACKGROUND_IMG_PATH, img))
for img in x_train2:
    x_train.append(os.path.join(PORES_IMG_PATH, img))

for img in x_test1:
    x_test.append(os.path.join(BACKGROUND_IMG_PATH, img))
for img in x_test2:
    x_test.append(os.path.join(PORES_IMG_PATH, img))

y_train = y_train1 + y_train2
y_test = y_test1 + y_test2
print ('x_train size = %d' % len(x_train))
print ('y_train size = %d' % len(y_train))
print ('x_test size = %d' % len(x_test))
print ('y_test size = %d' % len(y_test))
print ('----------------end to split data set!----------------')
print ('\n')

# save y_train and y_test value to file
print ('----------------writing y_train and y_test value----------------')
y_train_file = file(os.path.join(VALUE_PATH, 'y_train.npy'), 'wb')
y_test_file = file(os.path.join(VALUE_PATH, 'y_test.npy'), 'wb')
np.save(y_train_file, y_train)
np.save(y_test_file, y_test)
y_train_file.close()
y_test_file.close()

# write train images
print ('----------------writing train images----------------')
for i in xrange(0, len(x_train)):
    img = cv2.imread(x_train[i], 0)
    if img is not None:
        # print os.path.join(TRAIN_IMG_PATH, str(i)+'.jpg')
        cv2.imwrite(os.path.join(TRAIN_IMG_PATH, str(i) + '.jpg'), img)
        print ('%d/%d' % (i + 1, len(x_train)))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    else:
        print ('img is empty')
# write test images
print ('----------------writing test images----------------')
for i in xrange(0, len(x_test)):
    img = cv2.imread(x_test[i], 0)
    if img is not None:
        # print os.path.join(TRAIN_IMG_PATH, str(i)+'.jpg')
        cv2.imwrite(os.path.join(TEST_IMG_PATH, str(i) + '.jpg'), img)
        print ('%d/%d' % (i + 1, len(x_test)))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    else:
        print ('img is empty')
