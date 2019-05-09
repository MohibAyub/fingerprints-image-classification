#!/usr/bin/python
# coding:utf-8
import xgboost as xgb
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import feature_extract
import load_image

###################################
### 读取图片
imgs = load_image.LoadPredictImagesFromPath('./predict_img')
y_true = np.array([0] * len(imgs))  # load y true

###################################
### 加载模型
model = xgb.Booster(model_file='./model/xgb.model')

###################################
### 提取特征
print ('extract features...')
tr_std, tr_mean, tr_median, tr_argmax = feature_extract.GetImageStatistics(imgs)
tr_m00, tr_m10, tr_m01, tr_m20, tr_m11, tr_m02, tr_m30, tr_m21, tr_m12, tr_m03 = feature_extract.GetImageStatistics(imgs, 'm')
tr_nu20, tr_nu11, tr_nu02, tr_nu30, tr_nu21, tr_nu12, tr_nu03 = feature_extract.GetImageStatistics(imgs, 'nu')
tr_white, tr_black = feature_extract.GetImageStatistics(imgs, 'pixel')
tr_pore, tr_pore_coor1, tr_pore_coor2, tr_pore_coor3 = feature_extract.JudgePore(imgs)

###################################
### 特征组合,预测
print ('compose features...')
test = np.array(
    [tr_median, tr_std, tr_pore, tr_mean, tr_m00, tr_m02, tr_pore_coor1, tr_pore_coor2, tr_pore_coor3]).transpose()
xgtest = xgb.DMatrix(test)
y_pred = model.predict(xgtest)

###################################
### 预测置信度
# print ('AUC: %.4f' % metrics.roc_auc_score(y_true, y_pred))
print ('ACC: %.4f' % metrics.accuracy_score(y_true, y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_true, y_pred))
print ('F1-score: %.4f' % metrics.f1_score(y_true, y_pred))
print ('Precesion: %.4f' % metrics.precision_score(y_true, y_pred))
print(metrics.confusion_matrix(y_true, y_pred))

xgb.plot_importance(model)
plt.show()
