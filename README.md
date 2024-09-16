# Automatic badminton labeling

My aim is automate my analysis here in this repo: https://github.com/YichaoOU/Badminton_video_analysis

Basically, I need to know the players position.

# Video analysis

```

python final_video_analysis.py shuttlecock_pos.csv foot_pos.csv

```

We need shuttlecock position and player's position to determine the hit position for each round. See below how to do it. 

## Badminton shuttlecock detection

### TF ENV setup 

I followed this tutorial to setup an AI ENV: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/object_detection_camera.html

### Label your own video

The more training data, the better performance.

```

python video_shuttlecock_labeling.py Li_liang_M1_A_7_7.mp4

```

My videos are currently 1280p at 30fps. Too slow for badminton. In the future, I'm going to use 720p at 60fps.

### Converting tracknet V2 and V3 data to tf.record format


V2 and V3 are 720p data. Default box width/height = 20 seems work well.

Modify `files = glob.glob("*mp4")` and then run the code.

```

python Tracknet_csv_to_train_test_record.py

```

### merge tf record

```
python merge_tf_record.py

```

### Training

My P1000 only has 4G memory. So I used the `centernet_resnet50_v1_fpn_512x512_coco17_tpu-8` model, which has 27ms speed and 31.2 mAP. I didn't use TrackNet V2 or V3 models, as I don't know their speed. 

See `TF_model_zoo_training` for required files, such as `pipeline.config`. 

```

python model_main_tf2.py --model_dir=models\resnet --pipeline_config_path=models\resnet\pipeline2.config          

```          

### Exporting Model

```
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models\resnet\pipeline2.config --trained_checkpoint_dir models\resnet\ --output_directory exported-models-resnet 
```

### Visualize the performance

```

python VIS_object_detection.py  Li_liang_M1_A_3_7.mp4 test   

```

### Get shuttlecock position



```

python run_ballpos.py [video_folder]
```

Will save `{outLabel}.shuttlecock_pos.csv` in the current folder.

## Player's position

`run_openpose.py` glob a list of videos (default is MOV file) and for each video, it extracts the foot positions and save it as `{video_name}_foot_positions.csv`.

```


python run_openpose.py E:\badminton\2024_MEM\8-2024


```

#### OpenPose Usage

Little pain: openpose can only track one person in a video, so we have to do some post processing to figure out the movement of single or double matches.

This is what used inside the above python script.

I used the OpenPose.exe to detect player's position: https://github.com/CMU-Perceptual-Computing-Lab/openpose

`bin\OpenPoseDemo.exe  --render_pose 0 --net_resolution 400x320 --video  E:\badminton\2024_MEM\smash2.mp4 --write_json output_json_folder/ --display 0`

The json file is generated for each video (too big) and I created a csv file to store the "bigToe" positions.





