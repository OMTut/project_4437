import cv2
import os
import numpy as np

def refine_labels(label_dir, image_dir, output_dir, home_color_range, opponent_color_range):
    """
    Refine YOLO labels based on the dominant color within bounding boxes.

    Args:
        label_dir (str): Directory containing YOLO annotation files.
        image_dir (str): Directory containing corresponding images.
        output_dir (str): Directory to save the refined annotation files.
        home_color_range (tuple): Tuple of two numpy arrays representing the lower and upper bounds of the home team's color range in BGR format.
        opponent_color_range (tuple): Tuple of two numpy arrays representing the lower and upper bounds of the opponent team's color range in BGR format.
    """
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        
        image_file = label_file.replace('.txt', '.jpg')  # Adjust for your image extension
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, label_file)

        # Check if image and label files exist
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue

        image = cv2.imread(image_path)
        refined_labels = []

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])

            # Extract bounding box region
            roi = image[y_min:y_max, x_min:x_max]

            # Calculate color histograms for each channel
            hist_b = cv2.calcHist([roi], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([roi], [2], None, [256], [0, 256])

            # Find the dominant color in each channel
            dominant_b = np.argmax(hist_b)
            dominant_g = np.argmax(hist_g)
            dominant_r = np.argmax(hist_r)

            dominant_color = np.array([dominant_b, dominant_g, dominant_r])

            # Print debug information
            print(f"Processing {label_file}:")
            print(f"  Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")
            print(f"  Dominant color: {dominant_color}")

            # Assign class based on color
            if np.all(home_color_range[0] <= dominant_color) and np.all(dominant_color <= home_color_range[1]):
                refined_labels.append(f"0 {x_center} {y_center} {width} {height}\n")  # player_home
                print("  Assigned class: player_home")
            elif np.all(opponent_color_range[0] <= dominant_color) and np.all(dominant_color <= opponent_color_range[1]):
                refined_labels.append(f"1 {x_center} {y_center} {width} {height}\n")  # player_opponent
                print("  Assigned class: player_opponent")
            else:
                print("  No class assigned")

        # Save refined labels
        with open(output_path, 'w') as f:
            f.writelines(refined_labels)

        # Print debug information
        print(f"Refined labels saved to: {output_path}")
        print(f"Number of refined labels: {len(refined_labels)}")

# Example usage
refine_labels(
    label_dir='runs/detect/predict2/labels',
    image_dir='_img/raw_images',
    output_dir='_img/annotated',
    home_color_range=(np.array([0, 150, 50]), np.array([10, 255, 150])),  # Adjust for maroon
    opponent_color_range=(np.array([200, 200, 200]), np.array([255, 255, 255]))  # Adjust for white
)