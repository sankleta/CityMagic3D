import numpy as np
import open3d as o3d


def load_point_cloud(filename):
    return o3d.io.read_point_cloud(filename)


def split_into_cubes(pcd, cube_size):
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Compute grid dimensions based on cube size
    grid_dims = np.ceil((max_bound - min_bound) / cube_size).astype(int)
    cubes = {}

    for point in points:
        grid_coord = np.floor((point - min_bound) / cube_size).astype(int)
        grid_index = tuple(grid_coord)
        if grid_index not in cubes:
            cubes[grid_index] = []
        cubes[grid_index].append(point)

    return cubes


def downscale_cube(cube_points):
    # Example: calculate the mean of the points
    return np.mean(cube_points, axis=0)


# Load the point cloud
pcd = load_point_cloud('your_point_cloud.pcd')

# Define the cube size in meters
cube_size_in_meters = 0.5  # Change this to your desired cube size

# Split the point cloud into cubes of the specified size
cubes = split_into_cubes(pcd, cube_size=cube_size_in_meters)

# Downscale each cube
downscaled_cubes = {idx: downscale_cube(points) for idx, points in cubes.items()}

# Print or process the downscaled cubes
print(downscaled_cubes)
