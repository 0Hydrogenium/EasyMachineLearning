import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

from visualization.draw_line_graph import draw_line_graph


def calculate_classification_metrics(pred_data, real_data, model_name):
    info = {}

    real_data = np.round(real_data, 0).astype(int)
    pred_data = np.round(pred_data, 0).astype(int)

    cur_confusion_matrix = confusion_matrix(real_data[:, 0], pred_data)
    info["Confusion matrix of "+model_name] = cur_confusion_matrix

    info["Accuracy of "+model_name] = np.sum(cur_confusion_matrix.diagonal()) / np.sum(cur_confusion_matrix)
    info["Precision of "+model_name] = cur_confusion_matrix.diagonal() / np.sum(cur_confusion_matrix, axis=1)
    info["Recall of "+model_name] = cur_confusion_matrix.diagonal() / np.sum(cur_confusion_matrix, axis=0)
    info["F1-score of "+model_name] = np.mean(2 * np.multiply(info["Precision of "+model_name], info["Recall of "+model_name]) / \
                                      (info["Precision of "+model_name] + info["Recall of "+model_name]))

    max_class = max(real_data)[0]
    min_class = min(real_data)[0]
    pred_data_ = label_binarize(pred_data, classes=range(min_class, max_class+1))
    real_data_ = label_binarize(real_data, classes=range(min_class, max_class+1))

    for i in range(max_class - min_class):
        fpr, tpr, thresholds = roc_curve(real_data_[:, i], pred_data_[:, i])
        # draw_line_graph(fpr, tpr, "ROC curve with AUC={:.2f}".format(auc(fpr, tpr)))

    info["AUC of "+model_name] = roc_auc_score(real_data_, pred_data_)

    return info

