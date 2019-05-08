# -*- coding:utf-8 -*-
# !/usr/bin/python

from load_image import LoadImages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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
    m00 = m10 = m01 = m20 = m11 = m02 = m30 = m21 = m12 = m03 = []  # 空间矩
    mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 = []  # 中心矩
    nu20 = nu11 = nu02 = nu30 = nu21 = nu12 = nu03 = []  # 中心归一化矩
    white_pixels = black_pixels = []
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

            # 统计小的连通域个数
            # contours = []
            # contours = cv2.findContours(dst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            # con_cnt = 0
            # for con in contours:  # 遍历每一个轮廓
            #     if len(con) < 8:
            #         con_cnt += 1  # 碰到小的轮廓进行记录
            # min_contours.append(con_cnt)  # 保存小轮廓的个数
            # print contours

            # 显示图像，继续进行或者退出
            # cv2.imshow('dst', dst)
            # if (cv2.waitKey(0) == 27):
            #     break
            # else:
            #     continue
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


if __name__ == "__main__":
    PlotImageStatistics()
