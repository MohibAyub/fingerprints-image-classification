# -*- coding:utf-8 -*-
# !/usr/bin/python

from load_image import LoadImages
import cv2
import numpy as np
import matplotlib.pyplot as plt

ret_back_imgs, ret_pore_imgs, ret_x_train, ret_y_train, ret_x_test, ret_y_test = LoadImages('all')


################################
### 均值和方差特征效果不好
### ret_flag=None 返回std，mean，median，argmax
### ret_flag='m'
### ret_flag='mu'
### ret_flag='nu'

def GetImageStatistics(imgs, flag=None):
    std = []
    mean = []
    median = []
    argmax = []
    m00 = []
    m10 = []
    m01 = []
    m20 = []
    m11 = []
    m02 = []
    m30 = []
    m21 = []
    m12 = []
    m03 = []  # 空间矩

    mu20 = []
    mu11 = []
    mu02 = []
    mu30 = []
    mu21 = []
    mu12 = []
    mu03 = []  # 中心矩

    nu20 = []
    nu11 = []
    nu02 = []
    nu30 = []
    nu21 = []
    nu12 = []
    nu03 = []  # 中心归一化矩
    white_pixels = []
    black_pixels = []
    min_contours = []
    for i in xrange(0, len(imgs)):
        img = cv2.imread(imgs[i], 0)
        if img is not None:
            dst = np.zeros(img.shape, np.uint8)
            cv2.medianBlur(img, 3, dst)
            cv2.GaussianBlur(dst, (3, 3), 1, dst)
            cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU, dst)
            dst = dst[10:12, 10:12]  # 取中间6*6的图像

            mements = cv2.moments(dst, True)
            m00.append(mements['m00'])
            m10.append(mements['m10'])
            m01.append(mements['m01'])
            m20.append(mements['m20'])
            m11.append(mements['m11'])
            m02.append(mements['m02'])
            m30.append(mements['m30'])
            m21.append(mements['m21'])
            m12.append(mements['m12'])
            m03.append(mements['m03'])

            mu20.append(mements['mu20'])
            mu11.append(mements['mu11'])
            mu02.append(mements['mu02'])
            mu30.append(mements['mu30'])
            mu21.append(mements['mu21'])
            mu12.append(mements['mu12'])
            mu03.append(mements['mu03'])

            nu20.append(mements['nu20'])
            nu11.append(mements['nu11'])
            nu02.append(mements['nu02'])
            nu30.append(mements['nu30'])
            nu21.append(mements['nu21'])
            nu12.append(mements['nu12'])
            nu03.append(mements['nu03'])

            dst = dst.flatten()  # 将图像展开为一维数组
            std.append(np.std(dst))
            mean.append(np.mean(dst))
            median.append(np.median(dst))
            argmax.append(np.argmax(np.bincount(dst)))

            # 统计中间区域像素个数
            white_pixels.append(np.sum(dst == 255))
            black_pixels.append(np.sum(dst == 0))
        else:
            print ('img is empty')
            return

    if flag is None:
        return (std, mean, median, argmax)
    elif flag == 'm':
        return (m00, m10, m01, m20, m11, m02, m30, m21, m12, m03)
    elif flag == 'mu':
        return (mu20, mu11, mu02, mu30, mu21, mu12, mu03)
    elif flag == 'nu':
        return (nu20, nu11, nu02, nu30, nu21, nu12, nu03)
    elif flag == 'pixel':
        return (white_pixels, black_pixels)
    elif flag == 'contour':
        return min_contours
    else:
        print ('flag setting is wrong!')
        return


def AnySubPlotImageStatistics(data_cell, size):
    num = size[0] * size[1]
    if (len(data_cell) != size[0] * size[1]):
        print ('data_cell and size are not matched!')
        return
    for i in xrange(num):
        plt.subplot(size[0], size[1], i + 1)
        y = data_cell[i]
        x = np.linspace(0, 100, len(y))
        plt.plot(x, y, 'g+')
    plt.show()


################################
### 画图std和mean
### 比较好的特征 back_mean mu00
def PlotImageStatistics():
    back_std, back_mean, back_median, back_argmax = GetImageStatistics(ret_back_imgs)
    pore_std, pore_mean, pore_median, pore_argmax = GetImageStatistics(ret_pore_imgs)
    # AnySubPlotImageStatistics((back_std, back_mean, back_median, back_argmax, pore_std, pore_mean, pore_median, pore_argmax), (2, 4))
    #
    # m00, m10, m01, m20, m11, m02, m30, m21, m12, m03 = GetImageStatistics(ret_back_imgs, 'm')
    # mu20, mu11, mu02, mu30, mu21, mu12, mu03 = GetImageStatistics(ret_back_imgs, 'mu')
    # nu20, nu11, nu02, nu30, nu21, nu12, nu03 = GetImageStatistics(ret_back_imgs, 'nu')
    #
    # AnySubPlotImageStatistics((m00, m10, m01, m20, m11, m02, m30, m21, m12, m03), (2, 5))
    # AnySubPlotImageStatistics((mu20, mu11, mu02, mu30, mu21, mu12, mu03), (1, 7))
    # AnySubPlotImageStatistics((nu20, nu11, nu02, nu30, nu21, nu12, nu03), (1, 7))
    #
    # m00, m10, m01, m20, m11, m02, m30, m21, m12, m03 = GetImageStatistics(ret_pore_imgs, 'm')
    # mu20, mu11, mu02, mu30, mu21, mu12, mu03 = GetImageStatistics(ret_pore_imgs, 'mu')
    # nu20, nu11, nu02, nu30, nu21, nu12, nu03 = GetImageStatistics(ret_pore_imgs, 'nu')
    #
    # AnySubPlotImageStatistics((m00, m10, m01, m20, m11, m02, m30, m21, m12, m03), (2, 5))
    # AnySubPlotImageStatistics((mu20, mu11, mu02, mu30, mu21, mu12, mu03), (1, 7))
    # AnySubPlotImageStatistics((nu20, nu11, nu02, nu30, nu21, nu12, nu03), (1, 7))

    white_pixels1, black_pixels1 = GetImageStatistics(ret_back_imgs, 'pixel')
    white_pixels2, black_pixels2 = GetImageStatistics(ret_pore_imgs, 'pixel')
    AnySubPlotImageStatistics((white_pixels1, black_pixels1, white_pixels2, black_pixels2), (2, 2))


def JudgePore(background_imgs):
    # background_imgs = os.listdir(BACKGROUND_IMG_PATH)
    background_pore_count = [0] * len(background_imgs)
    background_pore_coordinate = [[0] * 2 for i in range(len(background_imgs))]
    count = 0
    processing = 0
    total = len(background_imgs)
    for name in background_imgs:
        processing += 1
        print ('JugePore: ', processing, '/', total)
        img_pore = cv2.imread(name, 0)
        # ret1是当前的阈值
        ret1_pore, img2_pore = cv2.threshold(img_pore, 0, 255, cv2.THRESH_OTSU)
        hist_full_pore = cv2.calcHist([img_pore], [0], None, [256], [0, 256])

        # find contours
        img3_pore = img2_pore
        # ret:连通域个数 labels:标注之后的图像
        ret_pore, labels_pore = cv2.connectedComponents(img3_pore, 4)
        number_pore = [0] * 20
        # print labels_pore
        # cv2.imshow('pore',img_pore)
        # cv2.waitKey(0)
        # ----------------------------------------------------------
        cut_lenth = 2
        cut_label = labels_pore[9 - cut_lenth:9 + cut_lenth, 9 - cut_lenth:9 + cut_lenth]
        # print cut_label
        # print labels_pore
        cal_arry = np.zeros((cut_lenth * 2, cut_lenth * 2))
        # print cal_arry

        for i in range(len(labels_pore)):
            for j in range(len(labels_pore[0])):
                for k in range(len(cut_label)):
                    for l in range(len(cut_label)):
                        if (labels_pore[i][j] == cut_label[k][l]):
                            cal_arry[k][l] += 1
        # print cal_arry
        # 输出统计的最小值--汗孔代表的连通域的数字的数量的最小值
        min_value = 1000
        min_value_coordinate = [0, 0]
        for i in range(len(cut_label)):
            for j in range(len(cut_label)):
                if (cal_arry[i][j] < min_value):
                    min_value = cal_arry[i][j]
                    min_value_coordinate = [i, j]
        threshold_value = 20
        # 这个min value记录的是 汗孔代表的连通域的数字的数量
        if (min_value < threshold_value):
            background_pore_count[count] = 1
            # 再统计坐标的位置
            x_sum = 0
            y_sum = 0
            x_cdnt = 0
            y_cdnt = 0
            min_value_count = 0
            min_value_in_img = cut_label[min_value_coordinate[0]][min_value_coordinate[1]]
            # print "min value:%d"%min_value_in_img
            for i in range(len(labels_pore)):
                for j in range(len(labels_pore[0])):
                    # print labels_pore[i][j]
                    if labels_pore[i][j] == min_value_in_img:
                        # print "in"
                        x_sum += i
                        y_sum += j
                        min_value_count += 1
            # print min_value_count
            x_cdnt = x_sum / min_value_count
            y_cdnt = y_sum / min_value_count
            background_pore_coordinate[count] = [x_cdnt, y_cdnt]
            # print pores_imgs_pore_coordinate[count]
        # print min_value
        count += 1
    return background_pore_count, background_pore_coordinate
