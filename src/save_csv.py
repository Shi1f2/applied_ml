import numpy as np


def save_as_csv(pred_labels, location='.'):
    assert pred_labels.shape[0] == 1434, 'wrong number of labels, should be 1434 test labels'
    np.savetxt(location + '/results_task1.csv', pred_labels, delimiter=',')
