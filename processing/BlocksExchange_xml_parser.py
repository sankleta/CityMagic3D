import xml.etree.ElementTree as ET
import numpy as np


# Function to extract the intrinsic matrix
def get_intrinsic_matrix(photogroup):
    focal_length = float(photogroup.find('FocalLengthPixels').text)
    px = float(photogroup.find('PrincipalPoint/x').text)
    py = float(photogroup.find('PrincipalPoint/y').text)
    intrinsic_matrix = np.array([
        [focal_length, 0, px],
        [0, focal_length, py],
        [0, 0, 1]
    ])
    return intrinsic_matrix


# Function to extract pose matrix from a Photo element
def get_pose_matrix(photo):
    rotation_elements = photo.find('Pose/Rotation')
    center_elements = photo.find('Pose/Center')
    rotation_matrix = np.array([
        [float(rotation_elements.find('M_00').text), float(rotation_elements.find('M_01').text),
         float(rotation_elements.find('M_02').text)],
        [float(rotation_elements.find('M_10').text), float(rotation_elements.find('M_11').text),
         float(rotation_elements.find('M_12').text)],
        [float(rotation_elements.find('M_20').text), float(rotation_elements.find('M_21').text),
         float(rotation_elements.find('M_22').text)]
    ])
    center_vector = np.array([float(center_elements.find('x').text), float(center_elements.find('y').text),
                              float(center_elements.find('z').text), 1])
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = center_vector[:3]
    return pose_matrix


def parse_xml(filename):
    # Load and parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    intrinsic_matrix = None
    poses_for_images = dict()
    # Loop through each photogroup and photo to extract matrices and image path
    for photogroup in root.findall('.//Photogroup'):
        print("Photogroup Name:", photogroup.find('Name').text)
        intrinsic_matrix = get_intrinsic_matrix(photogroup)
        print("Intrinsic Matrix:")
        print(intrinsic_matrix)

        for photo in photogroup.findall('Photo'):
            print("Photo ID:", photo.find('Id').text)
            image_name = photo.find('ImagePath').text.split("/")[-1]  # Extracting the image name
            print("Image Path:", image_name)
            pose_matrix = get_pose_matrix(photo)
            print("Pose Matrix:")
            print(pose_matrix)
            poses_for_images[image_name] = pose_matrix

    return intrinsic_matrix, poses_for_images
