# -*- coding: UTF-8 -*-
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from plot_metric.functions import BinaryClassification
import matplotlib.pyplot as plt
import numpy as np
import os

y_pred = np.load('./two_vari_fold_1_pred_list.fm.npy').tolist()
y_test = np.load('./two_vari_fold_1_label_list.fm.npy').tolist()

bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 

if not os.path.exists('./images'):
    os.makedirs('./images')
# Figures
plt.figure(figsize=(6,6))
# plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
# bc.plot_roc_curve(title='Receiver Operating Characteristic')
bc.plot_roc_curve(title='')
plt.savefig('images/plot_roc_curve.png', dpi=500,bbox_inches = 'tight')
plt.close()
# plt.subplot2grid((2,6), (0,2), colspan=2)
# bc.plot_precision_recall_curve(title='Precision and Recall Curve')
bc.plot_precision_recall_curve(title='')
plt.savefig('images/plot_precision_recall_curve.png', dpi=500,bbox_inches = 'tight')
plt.close()
# plt.subplot2grid((2,6), (0,4), colspan=2)
bc.plot_class_distribution(title='预测分类分布')
plt.savefig('images/plot_class_distribution.png', dpi=500,bbox_inches = 'tight')
plt.close()
# plt.subplot2grid((2,6), (1,1), colspan=2)
# bc.plot_confusion_matrix(title='混淆矩阵')
# plt.savefig('images/plot_confusion_matrix.png')
# plt.close()
# plt.subplot2grid((2,6), (1,3), colspan=2)
bc.plot_confusion_matrix(title='混淆矩阵',normalize=True)
# Save figure
plt.savefig('images/plot_confusion_matrix_normalize.png', dpi=500,bbox_inches = 'tight')
# Display Figure
plt.close()

# 画在一张图上
# Figures
plt.figure(figsize=(15,10))
plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
bc.plot_roc_curve()
plt.subplot2grid((2,6), (0,2), colspan=2)
bc.plot_precision_recall_curve()
plt.subplot2grid((2,6), (0,4), colspan=2)
bc.plot_class_distribution()
plt.subplot2grid((2,6), (1,1), colspan=2)
bc.plot_confusion_matrix()
plt.subplot2grid((2,6), (1,3), colspan=2)
bc.plot_confusion_matrix(normalize=True)
# Save figure
plt.savefig('images/plot_all.png', dpi=500,bbox_inches = 'tight')
# Display Figure
plt.close()
# end

# Full report of the classification
bc.print_report()

# Example custom param using dictionnary
param_pr_plot = {
    'c_pr_curve':'blue',
    'c_mean_prec':'cyan',
    'c_thresh_lines':'red',
    'c_f1_iso':'green',
    'beta': 2,
}

plt.figure(figsize=(6,6))
bc.plot_precision_recall_curve(**param_pr_plot)

# Save figure
plt.savefig('images/example_binary_class_PRCurve_custom.png')

# Display Figure
plt.close()