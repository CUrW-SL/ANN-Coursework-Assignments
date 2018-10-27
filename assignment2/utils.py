import pandas as pd
import numpy as np
from assignment2 import DATA_SET_FP, attribute_formation, attribute_converters


def read_dataset():
    df = pd.read_csv(DATA_SET_FP, sep=',', index_col=False, header=None, names=attribute_formation)

    # convert column values accordingly
    df = df.transform(attribute_converters)[attribute_formation]
    return df


def purity_score(purity_matrix):
    N = 0
    sum_of_majority = 0
    for cls_metric in purity_matrix:
        N += cls_metric.sum()
        sum_of_majority += cls_metric.max()
    return sum_of_majority / N


def get_purity_matrix(true_clustering, predicted_clustering):
    """
    classification of classes of predicted clustering against the true clustering classes.
    purity_of_pred_classes = {'pred_c1': {'true_c1': <count>, 'true_c2': <count>, 'true_c3': <count>},
                            'pred_c2': {'true_c1': <count>, 'true_c2': <count>, 'true_c3': <count>},
                            'pred_c3': {'true_c1': <count>, 'true_c2': <count>, 'true_c3': <count>}}
    """
    purity_of_pred_classes = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]

    for idx, pred_cls in predicted_clustering.iteritems():
        true_cls = true_clustering.get(idx)
        pred_cls_idx = pred_cls - 1
        true_cls_idx = true_cls - 1
        purity_of_pred_classes[pred_cls_idx][true_cls_idx] += 1

    return purity_of_pred_classes


def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))
