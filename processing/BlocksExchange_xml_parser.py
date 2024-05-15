import logging

import xml.etree.ElementTree as ET
import numpy as np

logger = logging.getLogger(__name__)


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


def get_image_dimensions(photogroup):
    width = int(photogroup.find('ImageDimensions/Width').text)
    height = int(photogroup.find('ImageDimensions/Height').text)
    return width, height


# Function to extract rotation matrix and center vector from a Photo element
def get_rotation_matrix_and_center(photo):
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
                              float(center_elements.find('z').text)])
    return rotation_matrix, center_vector


def parse_xml(filename):
    # Load and parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    intrinsic_matrix = None
    poses_for_images = dict()
    width, height = None, None
    # Loop through each photogroup and photo to extract matrices and image path
    for photogroup in root.findall('.//Photogroup'):
        logger.debug("Photogroup Name:", photogroup.find('Name').text)
        intrinsic_matrix = get_intrinsic_matrix(photogroup)
        logger.debug("Intrinsic Matrix:")
        logger.debug(intrinsic_matrix)
        width, height = get_image_dimensions(photogroup)
        logger.debug("Image Dimensions:", width, height)

        for photo in photogroup.findall('Photo'):
            logger.debug("Photo ID:", photo.find('Id').text)
            image_name = photo.find('ImagePath').text.split("/")[-1]  # Extracting the image name
            logger.debug("Image Path:", image_name)
            rot_m, center = get_rotation_matrix_and_center(photo)
            logger.debug("Pose Matrix:")
            logger.debug(rot_m, center)
            poses_for_images[image_name] = rot_m, center

    return intrinsic_matrix, poses_for_images, width, height
