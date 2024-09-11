import tensorflow as tf
import cv2
import os
import random
import pandas as pd
from object_detection.utils import dataset_util, label_map_util
from sklearn.model_selection import train_test_split
from collections import namedtuple

# Global variables
clicked_positions = []
frame_data = []

# Mouse callback to record click position
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_positions.append((x, y))

# Function to convert coordinates to bounding boxes and create TFExample
def create_tf_example(image, image_filename, label, center_x, center_y, img_width, img_height, width=25, height=25):
    xmin = (center_x - width / 2) / img_width
    ymin = (center_y - height / 2) / img_height
    xmax = (center_x + width / 2) / img_width
    ymax = (center_y + height / 2) / img_height

    encoded_image = cv2.imencode('.jpg', image)[1].tobytes()

    # Create the TensorFlow Example following the structure expected by the API
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_height),
        'image/width': dataset_util.int64_feature(img_width),
        'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature([xmin]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([ymin]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([xmax]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([ymax]),
        'image/object/class/text': dataset_util.bytes_feature(label.encode('utf8')),
        'image/object/class/label': dataset_util.int64_list_feature([1])  # Assuming label is always 'object' (label 1)
    }))
    
    return tf_example

# Write the TFRecord file
def write_tfrecord(output_path, data):
    writer = tf.io.TFRecordWriter(output_path)

    for image, filename, label, (x, y), img_width, img_height in data:
        tf_example = create_tf_example(image, filename, label, x, y, img_width, img_height)
        writer.write(tf_example.SerializeToString())

    writer.close()

# Function to process video frames and generate TFRecord
def main(video_path, output_train_record, output_test_record, label_map):
    global clicked_positions

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"frame_{current_frame}.jpg"
        img_height, img_width, _ = frame.shape
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', click_event)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if len(clicked_positions) > 0:
                for pos in clicked_positions:
                    frame_data.append((frame, frame_filename, label_map['object'], pos, img_width, img_height))
            clicked_positions = []
            current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(frame_data, test_size=0.2, random_state=42)

    # Write the TFRecord files
    write_tfrecord(output_train_record, train_data)
    write_tfrecord(output_test_record, test_data)

    print(f"TFRecord files created: {output_train_record}, {output_test_record}")

if __name__ == "__main__":
    video_file = sys.argv[1]  # Path to your video file
    label = os.path.basename(video_file).split('.')[0] 
    output_train_record = f'{label}.train.record'
    output_test_record = f'{label}.test.record'

    main(video_file, output_train_record, output_test_record)
