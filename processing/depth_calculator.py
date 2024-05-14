import cv2
import open3d as o3d
import numpy as np

from BlocksExchange_xml_parser import parse_xml
from instance_masks_from_images.utils import get_extrinsic_matrix

camera_xml_path = '/Users/sankleta/Downloads/OCCC example/OCCC_CamInfoCC.xml'
imgs_base_path = "/Users/sankleta/Downloads/OCCC example"

intrinsic_matrix, poses_for_images, _, _ = parse_xml(camera_xml_path)

# Load the point cloud
point_cloud = o3d.io.read_point_cloud("/Users/sankleta/Downloads/STPLS3D/RealWorldData/OCCC_points.ply")

for filename in poses_for_images:
    # Load the RGB image
    img_path = f'{imgs_base_path}/{filename}'
    rgb_image = cv2.imread(img_path)

    # Example camera pose matrix
    extrinsic_matrix = get_extrinsic_matrix(*poses_for_images[filename])

    # Convert Open3D PointCloud to NumPy array
    points = np.asarray(point_cloud.points)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates

    # Transform points to camera coordinates
    points_camera = np.dot(extrinsic_matrix, points_homogeneous.T).T

    # Project points to the image plane
    points_image = np.dot(intrinsic_matrix, points_camera[:, :3].T).T
    points_image[:, 0] /= points_image[:, 2]  # Normalize x by z
    points_image[:, 1] /= points_image[:, 2]  # Normalize y by z

    # Initialize depth image
    depth_image = np.zeros(rgb_image.shape[:2], dtype=np.float32)

    # Fill in the depth image
    for x, y, z in points_image:
        u, v = int(round(x)), int(round(y))
        if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
            # Set depth value if no depth is set or found depth is closer
            if depth_image[v, u] == 0 or depth_image[v, u] > z:
                depth_image[v, u] = z

    # Optionally, you might want to normalize the depth image for visualization
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('Depth Map', depth_normalized.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
