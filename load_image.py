# -*- coding:utf-8 -*-
# !/usr/bin/python


import os
from sklearn.model_selection import train_test_split



#######################################
### @input: 'back' 加载背景图像
###         'pore' 加载汗孔图像
###         'train' 加载train集图像和对应label
###         'test' 加载test集图像和对应label
###         'all' 返回上述所有
### @return: 返回图像全路径
def LoadImages(select):
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

    ###################################
    ### split test and train data set by using np.random.permutation
    print ('----------------start to split data set----------------')

    # 利用sklearn自动分割
    ret_back_imgs = []
    ret_pore_imgs = []
    x1 = background_imgs
    x2 = pores_imgs
    y1 = [0] * len(background_imgs)
    y2 = [1] * len(pores_imgs)

    x_train, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.2)  # background image name
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2)  # pores image name
    ret_x_train = []  # image path
    ret_x_test = []
    for img in x_train:
        ret_x_train.append(os.path.join(BACKGROUND_IMG_PATH, img))
        ret_back_imgs.append(os.path.join(BACKGROUND_IMG_PATH, img))
    for img in x_train2:
        ret_x_train.append(os.path.join(PORES_IMG_PATH, img))
        ret_pore_imgs.append(os.path.join(PORES_IMG_PATH, img))

    for img in x_test1:
        ret_x_test.append(os.path.join(BACKGROUND_IMG_PATH, img))
        ret_back_imgs.append(os.path.join(BACKGROUND_IMG_PATH, img))
    for img in x_test2:
        ret_x_test.append(os.path.join(PORES_IMG_PATH, img))
        ret_pore_imgs.append(os.path.join(PORES_IMG_PATH, img))

    ret_y_train = y_train1 + y_train2
    ret_y_test = y_test1 + y_test2
    print ('ret_back_imgs size = %d' % len(ret_back_imgs))
    print ('ret_pore_imgs size = %d' % len(ret_pore_imgs))
    print ('ret_x_train size = %d' % len(ret_x_train))
    print ('ret_y_train size = %d' % len(ret_y_train))
    print ('ret_x_test size = %d' % len(ret_x_test))
    print ('ret_y_test size = %d' % len(ret_y_test))
    print ('----------------end to split data set!----------------')
    print ('\n')

    if select == 'back':
        return ret_back_imgs
    elif select == 'pore':
        return ret_pore_imgs
    elif select == 'train':
        return (ret_x_train, ret_y_train)
    elif select == 'test':
        return (ret_x_test, ret_y_test)
    elif select == 'all':
        return (ret_back_imgs, ret_pore_imgs, ret_x_train, ret_y_train, ret_x_test, ret_y_test)
    else:
        print ('wrong input param!')
        return
