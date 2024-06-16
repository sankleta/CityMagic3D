import os
import fnmatch
import json
from PIL import Image
import csv

'''
This script is to prepare the FMoW dataset (generate cropped images and requited text-image csv) to fine-tune CLIP. 
The experimental task was to train to recognize pool. We didn't have enough time/computational resources to go this 
route.
However setup-wise it was not that complicated and the dataset is extensive.
'''


def traverse_and_collect_files(root_dir):
    collected_files = {'jpeg': [], 'json': []}

    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file matches the desired pattern
            if fnmatch.fnmatch(filename, '*_rgb.jpeg') or fnmatch.fnmatch(filename, '*_rgb.jpg'):
                collected_files['jpeg'].append(os.path.join(dirpath, filename))
            elif fnmatch.fnmatch(filename, '*_rgb.json'):
                collected_files['json'].append(os.path.join(dirpath, filename))

    return collected_files


def extract_bbox_and_crop_image(json_path, image_path, output_dir):
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract bounding box from the first entry in the 'bounding_boxes' list
    if 'bounding_boxes' in data and data['bounding_boxes']:
        bbox = data['bounding_boxes'][0]['box']
        x, y, width, height = bbox
    else:
        print(f"No bounding boxes found in {json_path}")
        return None

    # Open the image
    with Image.open(image_path) as img:
        # Crop the image using the bbox
        cropped_img = img.crop((x, y, x + width, y + height))

        # Create output file path
        base_name = os.path.basename(image_path).replace('_rgb.jpeg', '_cropped.jpeg').replace('_rgb.jpg',
                                                                                               '_cropped.jpg')
        cropped_image_path = os.path.join(output_dir, base_name)

        # Save the cropped image
        cropped_img.save(cropped_image_path)
        print(f"Cropped image saved to {cropped_image_path}")

        return cropped_image_path


def create_csv(cropped_files, output_csv):
    caption = "\"A picture of an outdoor pool with water\""

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for file_path in cropped_files:
            csvwriter.writerow([file_path, caption])
    print(f"CSV file created at {output_csv}")


def main():
    root_dir = r'C:\Users\sankl\Downloads\my'
    output_dir = r'C:\Users\sankl\Downloads\pool'
    output_csv = r'C:\Users\sankl\Downloads\cropped_images.csv'  # Replace with the path to the output CSV file

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    collected_files = traverse_and_collect_files(root_dir)

    # Create a mapping from base filename to its paths
    base_to_paths = {}
    for jpeg_path in collected_files['jpeg']:
        base_name = os.path.basename(jpeg_path).replace('_rgb.jpeg', '').replace('_rgb.jpg', '')
        base_to_paths[base_name] = {'jpeg': jpeg_path}
    for json_path in collected_files['json']:
        base_name = os.path.basename(json_path).replace('_rgb.json', '')
        if base_name in base_to_paths:
            base_to_paths[base_name]['json'] = json_path

    # Process each pair of JSON and JPEG files and collect cropped file paths
    cropped_files = []
    for base_name, paths in base_to_paths.items():
        if 'jpeg' in paths and 'json' in paths:
            cropped_file_path = extract_bbox_and_crop_image(paths['json'], paths['jpeg'], output_dir)
            if cropped_file_path:
                cropped_files.append(cropped_file_path)

    # Create CSV file with cropped image paths and captions
    create_csv(cropped_files, output_csv)


if __name__ == "__main__":
    main()