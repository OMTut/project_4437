# Project: Using YOLO to Identify and Track TXST Football Players

### Team
Rykard, Lord of Blasphemy

## FIles
### vidrunner.py
Processes a video to detect football players using a YOLO model. It takes three arguments.
- video_path (str): Path to the input video
- output_path (str): Path to save the processed video
- model_path (str): Path to the YOLO model (e.g., yolo11n.pt)

### teamsbycolor.py
This algorithm looks at the pixes within a bounding box established by YOLO and attemtps to modify the label to reflect the dominant color found within the bounding box. For instance, if the dominant color in a bounding box surrounding a TXST player is maroon, it would label the box as TXST

The alogirthm takes 5 arguments:
- label_dir (str): Directory containing YOLO annotation files.
- image_dir (str): Directory containing corresponding images.
- output_dir (str): Directory to save the refined annotation files.
- home_color_range (tuple): Tuple of two numpy arrays representing the lower and upper bounds of the home team's color range in BGR format.
- opponent_color_range (tuple): Tuple of two numpy arrays representing the lower and upper bounds of the opponent team's color range in BGR format.

## Imagetrainer.py && Imagetrainer11.py
These files accomplish similar things. They are both models meant to train on images finding football players.

**imagetrainer.py**
- utilizes cuda
- includes TensorBoard logging and augmentation pipeline

**imagetrainer11.py**
- no augmentation
- implements early stopping