import numpy as np
import pandas as pd
import cv2
import os
import json
import glob
import subprocess
import pandas as pd
import sys


def get_ball_pos(video_file, outLabel):
    """
    This function processes a video, detects objects in each frame, and logs the 
    frame number, x and y coordinates of the center of the detection box, and prediction probability 
    for all detections with a score greater than or equal to 0.2.
    """
    cap = cv2.VideoCapture(video_file)
    frame_number_count = 0
    log_list = []
    cv2.namedWindow('image')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)

        # Detect objects in the frame
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        # Get detection boxes, scores, and classes
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_scores = detections['detection_scores'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy()

        # Filter detections with scores >= 0.2
        valid_detections = [(box, score) for box, score in zip(detection_boxes, detection_scores) if score >= 0.2]

        for box, score in valid_detections:
            ymin, xmin, ymax, xmax = box
            image_height, image_width, _ = frame.shape
            center_x = int((xmin + xmax) / 2 * image_width)
            center_y = int((ymin + ymax) / 2 * image_height)

            # Log the frame number, center coordinates, and prediction probability
            log_list.append([frame_number_count, center_x, center_y, score])

        frame_number_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save the log to a DataFrame and CSV
    df = pd.DataFrame(log_list, columns=['frame', 'x', 'y', 'prob'])
    df.to_csv(f"{outLabel}.shuttlecock_pos.csv", index=False)
    return df
# Paths
video_folder = sys.argv[1]  # Folder containing videos

video_list = glob.glob(os.path.join(video_folder, '*.MOV'))  # Change extension if needed
# video_list = glob.glob(os.path.join(video_folder, 'smash.mp4'))  # Change extension if needed


# Process each video file
for video_file in video_list:
    # Get the base name of the video file (without extension)
    video_name = os.path.basename(video_file).split('.')[0]
    print (f"Reading {video_name}")
    get_ball_pos(video_file,video_name)
