import numpy as np


def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        print("Array loaded successfully:")
        print(data)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")


array = load_npy_file(r"C:\Users\sankl\Downloads\RA\Marigold\depth_npy\DJI_0094_pred.npy")
min_val = np.min(array)
max_val = np.max(array)
print(f"Minimum value: {min_val}")
print(f"Maximum value: {max_val}")
