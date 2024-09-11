import os
import json
import glob
import subprocess
import pandas as pd
import sys

# Paths
video_folder = sys.argv[1]  # Folder containing videos
output_json_folder = 'output_json_folder/'  # Folder where JSON files will be saved
openpose_executable = 'E:\\TensorFlow\\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\\openpose\\bin\\OpenPoseDemo.exe'  # Path to OpenPose executable

# Glob to get a list of all video files (change the file extension as needed)
video_list = glob.glob(os.path.join(video_folder, 'IMG*.MOV'))  # Change extension if needed
# video_list = glob.glob(os.path.join(video_folder, 'smash.mp4'))  # Change extension if needed

# Function to run OpenPose on a video file and save JSON output
def run_openpose(video_path, output_json_folder):
    # Construct the command to run OpenPose
    command = [
        openpose_executable,
        '--video', video_path,
        '--write_json', output_json_folder,
        '--display', '0',
        # '--net_resolution', '-1x368',
        '--net_resolution', '480x320', # faster with good accuracy
        '--frame_step', '5',
        '--model_folder', 'E:\\TensorFlow\\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\\openpose\\models',
        '--render_pose', '0'
    ]
    # Run the OpenPose command
    # print (" ".join(command))
    subprocess.run(command, check=True)
def clean_json_folder(json_folder):
    files = glob.glob(os.path.join(json_folder, '*.json'))
    for f in files:
        os.remove(f)
    print(f"Removed {len(files)} files from {json_folder}.")
# Function to extract ankle positions from JSON files and save to CSV
def extract_and_save_csv(json_folder, csv_output_path,video_name):
    data_list = []

    # Iterate through all JSON files in the folder
    for json_file in sorted(glob.glob(os.path.join(json_folder, '*.json'))):
        with open(json_file, 'r') as f:
            data = json.load(f)
        json_file = json_file.replace(video_name,"")
        frame_number = os.path.basename(json_file).split('_')[1]  # Extract frame number from file name
        # Check if there are people in the frame
        if len(data['people']) > 0:
            # Loop over all detected people in the frame
            for person in data['people']:
                keypoints = person['pose_keypoints_2d']

                # Extract left and right ankle coordinates for each person
                l_ankle_x, l_ankle_y = keypoints[19 * 3], keypoints[19 * 3 + 1]
                r_ankle_x, r_ankle_y = keypoints[22 * 3], keypoints[22 * 3 + 1]

                # Append the extracted data to the list (including the frame number)
                data_list.append([frame_number, l_ankle_x, l_ankle_y, r_ankle_x, r_ankle_y])

    # Create a pandas DataFrame from the data list
    df = pd.DataFrame(data_list, columns=['frame_number', 'l_foot_x', 'l_foot_y', 'r_foot_x', 'r_foot_y'])

    # Save DataFrame to CSV
    df.to_csv(csv_output_path, index=False)



# Process each video file
for video_file in video_list:
    # Get the base name of the video file (without extension)
    video_name = os.path.basename(video_file).split('.')[0]
    try:
        print (f"Reading {video_name}")
        # Run OpenPose on the video
        run_openpose(video_file, output_json_folder)

        # Extract keypoints from JSON output and save to CSV
        csv_output_file = f'{video_name}_foot_positions.csv'
        extract_and_save_csv(output_json_folder, csv_output_file,video_name)
        print(f"Processed {video_file} and saved data to {csv_output_file}")
        # Clean the JSON folder before processing
        clean_json_folder(output_json_folder)  
    except Exception as e:
        print (e)
        print (f"Failed {video_file}")
  
    
