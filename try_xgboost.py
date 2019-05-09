#!/usr/bin/python
# coding:utf-8
import xgboost as xgb
import pandas as pd
import time
import numpy as np
from feature_extract import GetImageStatistics
from feature_extract import JudgePore
from load_image import LoadImages
from sklearn import metrics

# # for i in xrange(0, len(x_train_imgs_name)):
# for i in xrange(0, 10):
#     img = cv2.imread(os.path.join(TRAIN_IMG_PATH, x_train_imgs_name[i]), 0)
#     if img is not None:
#         print ('%d/%d' % (i + 1, len(x_train_imgs_name)))
#         ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         print ret1
#         plt.figure()
#         plt.subplot(221), plt.imshow(img, 'gray')
#         plt.subplot(222), plt.hist(img.ravel(), 256)  # .ravel方法将矩阵转化为一维
#         plt.subplot(223), plt.imshow(th1, 'gray')
#         plt.subplot(224), plt.hist(th1.ravel(), 256)
#         # cv2.imshow('img', img)
#         plt.show()
#         # cv2.waitKey(0)
#     else:
#         print ('img is empty')


###################################
### 读取图片
print ('read images...')
ret_back_imgs, ret_pore_imgs, ret_x_train, ret_y_train, ret_x_test, ret_y_test = LoadImages('all')

###################################
### 提取特征
print ('extract features...')
tr_std, tr_mean, tr_median, tr_argmax = GetImageStatistics(ret_x_train)
tr_m00, tr_m10, tr_m01, tr_m20, tr_m11, tr_m02, tr_m30, tr_m21, tr_m12, tr_m03 = GetImageStatistics(ret_x_train, 'm')
tr_white, tr_black = GetImageStatistics(ret_x_train, 'pixel')
tr_pore, tr_pore_coor = JudgePore(ret_x_train)

te_std, te_mean, te_median, te_argmax = GetImageStatistics(ret_x_test)
te_m00, te_m10, te_m01, te_m20, te_m11, te_m02, te_m30, te_m21, te_m12, te_m03 = GetImageStatistics(ret_x_test, 'm')
te_white, te_black = GetImageStatistics(ret_x_test, 'pixel')
te_pore, te_pore_coor = JudgePore(ret_x_test)

###################################
### 特征组合
print ('compose features...')
# train = np.array([tr_std, tr_mean, tr_median, tr_m00, tr_white, tr_black]).transpose()
train = np.array([tr_mean, tr_m00, tr_pore]).transpose()
tr_labels = np.array(ret_y_train)
# test = np.array([te_std, te_mean, te_median, te_m00, te_white, te_black]).transpose()
test = np.array([te_mean, te_m00, te_pore]).transpose()
te_labels = np.array(ret_y_test)

###################################
### 进行xgb training
print ('xgb trainning...')
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'num_class': 2,  # 类数，与 multisoftmax 并用
    'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth': 12,  # 构建树的深度 [1:]
    # 'lambda':450,  # L2 正则项权重
    'subsample': 0.4,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
    # 'min_child_weight':12, # 节点的最少特征数
    'silent': 0,  # 取0时表示打印出运行时信息
    'eta': 0.005,  # 如同学习率
    'seed': 710,
    'nthread': 4,  # cpu 线程数,根据自己U的个数适当调整
}

plst = list(params.items())
num_rounds = 500  # 迭代你次数

#####################################
### predict dataset
xgtest = xgb.DMatrix(test)

#####################################
### 划分训练集与验证集
offset = int(len(tr_labels) * 0.8)
print 'train shape: ', tr_labels.shape, type(tr_labels)

xgtrain = xgb.DMatrix(train[:offset, :], tr_labels[:offset])
xgval = xgb.DMatrix(train[offset:, :], tr_labels[offset:])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'), (xgval, 'val')]

# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
# model.save_model('./model/xgb.model') # 用于存储训练出的模型
y_pred = model.predict(xgtest, ntree_limit=model.best_iteration)

# # 将预测结果写入文件，方式有很多，自己顺手能实现即可
# np.savetxt('submission_xgb_MultiSoftmax.csv', np.c_[range(1, len(test) + 1), preds],
#            delimiter=',', header='ImageId,Label', comments='', fmt='%d')


print ('AUC: %.4f' % metrics.roc_auc_score(te_labels, y_pred))
print ('ACC: %.4f' % metrics.accuracy_score(te_labels, y_pred))
print ('Recall: %.4f' % metrics.recall_score(te_labels, y_pred))
print ('F1-score: %.4f' % metrics.f1_score(te_labels, y_pred))
print ('Precesion: %.4f' % metrics.precision_score(te_labels, y_pred))
print(metrics.confusion_matrix(te_labels, y_pred))
