import cv2
from ultralytics import YOLO

def process_video(video_path, output_path, model_path, confidence_threshold=0.3):
    """
    Process a video to detect football players using a YOLO model.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the processed video.
        model_path (str): Path to the YOLO model (e.g., yolo11n.pt).
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can try other codecs like 'XVID' or 'MJPG'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: Could not initialize VideoWriter.")
        return
    
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Draw detections on the frame
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and class info
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                class_id = box.cls[0]  # Class ID

                if confidence > confidence_threshold:
                    label = f"{model.names[int(class_id)]} {confidence:.2f}"
                    color = (0, 255, 0)  # Green color for bounding boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the processed frame to the output video
        if out.isOpened():
            out.write(frame)
        else:
            print("Error: Frame could not be written.")
        
        print("Frame processed")

        # Display the frame in a window (optional)
        #cv2.imshow('YOLO Detection', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release resources
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    # Replace these paths with your actual file paths
    video_path = "video/output.mp4"  # Path to the input video
    output_path = "video/results/output_video.mp4"  # Path to save the output video
    #model_path = "runs/detect/train83333/weights/best.pt"  # Path to your trained YOLO model
    model_path = "yolo11n.pt"
    process_video(video_path, output_path, model_path)
