# Provided Task 2 saver from the Colab worksheet — do not roll our own.
# Expects (554, 5, 2) points at original image resolution.
import numpy as np


def save_as_csv(points, location='.'):
    assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:]) == 5 * 2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
