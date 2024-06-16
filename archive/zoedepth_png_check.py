from PIL import Image
import numpy as np

'''This is a dummy script to check the statistics of the ZoeDepth metric depth estimator. 
Some people swore by it, but it looks like it doesn't generalize well for outdoor scene'''


def get_min_max_from_png(file_path):
    # Load the image
    img = Image.open(file_path)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Find the minimum and maximum values
    min_value = np.min(img_array)
    max_value = np.max(img_array)

    return min_value, max_value


if __name__ == "__main__":
    min_val, max_val = get_min_max_from_png('C:\\Users\\sankl\\Downloads\\RA\\ZoeDepth\\DJI_0001.png')
    print(f"Minimum value: {min_val}, Maximum value: {max_val}")
