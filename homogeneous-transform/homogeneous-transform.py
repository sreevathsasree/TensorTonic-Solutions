import numpy as np

def apply_homogeneous_transform(T, points):

    points = np.asarray(points)

    single_point = False
    if points.ndim == 1:
        points = points[None, :]
        single_point = True

    N = points.shape[0]

    ones = np.ones((N, 1))
    points_h = np.hstack((points, ones))

    transformed = (T @ points_h.T).T

    result = transformed[:, :3]

    if single_point:
        return result[0]

    return result