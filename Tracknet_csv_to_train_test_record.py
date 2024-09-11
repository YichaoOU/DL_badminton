import tensorflow as tf
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from object_detection.utils import dataset_util

# Function to convert coordinates to bounding boxes and create TFExample
def create_tf_example(image, image_filename, x, y, img_width, img_height, width=20, height=20, label=1):
    # Create bounding box based on object center (x, y)
    xmin = (x - width / 2) / img_width
    ymin = (y - height / 2) / img_height
    xmax = (x + width / 2) / img_width
    ymax = (y + height / 2) / img_height

    encoded_image = cv2.imencode('.jpg', image)[1].tobytes()

    # Create TensorFlow Example
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
        'image/object/bbox/text': dataset_util.bytes_feature("ball".encode('utf-8')),
        'image/object/class/label': dataset_util.int64_list_feature([1]),
    }))
    
    return tf_example

# Function to draw bounding boxes on images
def draw_bounding_box(image, x, y, img_width, img_height, width=50, height=50):
    # Calculate bounding box coordinates
    xmin = int(x - width / 2)
    ymin = int(y - height / 2)
    xmax = int(x + width / 2)
    ymax = int(y + height / 2)
    
    # Draw the rectangle (box) on the image
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    return image

# Function to write TFRecord
def write_tfrecord(output_path, data):
    writer = tf.io.TFRecordWriter(output_path)

    for image, filename, x, y, img_width, img_height in data:
        tf_example = create_tf_example(image, filename, x, y, img_width, img_height)
        writer.write(tf_example.SerializeToString())

    writer.close()

# Main function to generate train and test TFRecord from video and CSV
def main(video_path, csv_path, output_train_record, output_test_record):
    # Load the CSV with object x, y coordinates and frame numbers
    df = pd.read_csv(csv_path)
    df = df[df.Ball==1]  # Filter rows where the object (ball) is present
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_data = []

    # Iterate over the frames of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    while current_frame < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        # Get image dimensions
        img_height, img_width, _ = frame.shape

        # Process each row of the CSV corresponding to the current frame
        frame_rows = df[df['Frame'] == current_frame]
        # print (current_frame,frame_rows)
        for _, row in frame_rows.iterrows():
            x, y = row['x'], row['y']
            x = int(x*img_width)
            y = int(y*img_height)
            frame_filename = f"frame_{current_frame}.jpg"
            
            # Highlight the object on the image
            # highlighted_frame = draw_bounding_box(frame, x, y, img_width, img_height)
            
            # Show the frame with the bounding box
            # cv2.imshow('Object Detection', highlighted_frame)
            
            # Press any key to proceed to the next frame, 'q' to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     return
            # Press any key to proceed to the next frame, 'q' to quit
            # if cv2.waitKey(0) & 0xFF == ord(' '):
            #     print ("")          
            # Add the frame data for TFRecord writing
            frame_data.append((frame, frame_filename, x, y, img_width, img_height))

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    # Split frame data into training and testing
    train_data = [data for data in frame_data if data[1] in train_df['Frame'].unique()]
    test_data = [data for data in frame_data if data[1] in test_df['Frame'].unique()]

    # Split frame data into training and testing
    train_data = [data for data in frame_data if int(data[1].split('_')[1].split('.')[0]) in train_df['Frame'].unique()]
    test_data = [data for data in frame_data if int(data[1].split('_')[1].split('.')[0]) in test_df['Frame'].unique()]
    # print (train_data)
    # print (train_df)
    # Write train and test data to TFRecord files
    write_tfrecord(output_train_record, train_data)
    write_tfrecord(output_test_record, test_data)

    print(f"TFRecord files created: {output_train_record}, {output_test_record}")
import glob
if __name__ == "__main__":
    files = glob.glob("*mp4")
    for f in files:
        print (f)
        label = os.path.basename(f).split('.')[0] 
        # label = "00001"
        video_path = f'{label}.mp4'  # Path to your video file
        csv_path = f'{label}.csv'   # Path to your labeled CSV file
        output_train_record = f'{label}.train.record'
        output_test_record = f'{label}.test.record'

        main(video_path, csv_path, output_train_record, output_test_record)
