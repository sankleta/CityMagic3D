import numpy as np


def get_2d_bounding_box_of_point_set(point_cloud, point_indices):
    coords = np.asarray(point_cloud.points)[point_indices]
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords
    return min_x, min_y, max_x, max_y

