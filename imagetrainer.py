from ultralytics import YOLO
import cv2
import glob
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

# Define the augmentation pipeline
transform = A.Compose([
    A.LongestMaxSize(max_size=640),  # Resize the longest side to 640 while keeping the aspect ratio
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),  # Pad to 640x640
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ToTensorV2()
])

def main():
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a new model and move it to the specified device
    model = YOLO('yolov11n.yaml').to(device)

    # TensorBoard writer
    writer = SummaryWriter('runs/yolov11n')

    # Train the model
    train_results = model.train(data='data_TXST.yaml', epochs=5, device=device, lr0=0.001, augment=True)

    # Evaluate the model
    val_results = model.val(device=device)

    # Print results to inspect their structure)
    print("Train Results Type:", type(train_results))
    print("Validation Results Type:", type(val_results))


    # Train the model
    for epoch in range(5):
        train_results = model.train(data='data.yaml', epochs=1, device=device, lr0=0.001, augment=True)
        val_results = model.val(device=device)

        # Log training and validation metrics to TensorBoard
        # Extract metrics from train and val results
        train_mp, train_mr, train_map50, train_map = train_results.box.mean_results()
        val_mp, val_mr, val_map50, val_map = val_results.box.mean_results()

        # Log metrics to TensorBoard
        writer.add_scalar('Precision/train', train_mp, epoch)
        writer.add_scalar('Recall/train', train_mr, epoch)
        writer.add_scalar('mAP/train_50', train_map50, epoch)
        writer.add_scalar('mAP/train_50_95', train_map, epoch)

        writer.add_scalar('Precision/val', val_mp, epoch)
        writer.add_scalar('Recall/val', val_mr, epoch)
        writer.add_scalar('mAP/val_50', val_map50, epoch)
        writer.add_scalar('mAP/val_50_95', val_map, epoch)
        # writer.add_scalar('Loss/train/box', train_results.metrics.box_loss, epoch)
        # writer.add_scalar('Loss/train/obj', train_results.metrics.obj_loss, epoch)
        # writer.add_scalar('Loss/train/cls', train_results.metrics.cls_loss, epoch)
        # writer.add_scalar('Loss/val/box', val_results.metrics.box_loss, epoch)
        # writer.add_scalar('Loss/val/obj', val_results.metrics.obj_loss, epoch)
        # writer.add_scalar('Loss/val/cls', val_results.metrics.cls_loss, epoch)
        # writer.add_scalar('mAP/val', val_results.metrics.mAP_0_5, epoch)

    writer.close()

    # Get all validation images
    val_images = glob.glob('dataset/images/val/*.jpg')  # Adjust the pattern if your images have different extensions

    # Create a directory to save the output images
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)

    for img_path in val_images:
        # Load a validation image
        img = cv2.imread(img_path)
        original_size = img.shape[:2]  # Save the original size (height, width)

        # Apply augmentations
        augmented = transform(image=img)
        img_resized = augmented['image'].unsqueeze(0)  # Add batch dimension

        # Ensure the tensor is of type float32
        img_resized = img_resized.float() / 255.0

        # Resize the image to 640x640 for model input
        #img_resized = cv2.resize(img, (640, 640))


        # Make predictions
        results = model.predict(img_resized, device=device)

        # Visualize predictions on the original image
        for result in results:
            for box in result.boxes:
                if len(box.xyxy) == 4:
                    x1, y1, x2, y2 = box.xyxy
                    # Scale the coordinates back to the original image size
                    x1 = int(x1 * original_size[1] / 640)
                    y1 = int(y1 * original_size[0] / 640)
                    x2 = int(x2 * original_size[1] / 640)
                    y2 = int(y2 * original_size[0] / 640)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{box.cls} {box.conf:.2f}', 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, (0, 255, 0), 2)

        # Save the image with predictions
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)

if __name__ == '__main__':
    main()