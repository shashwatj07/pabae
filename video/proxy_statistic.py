import os
import cv2
import numpy as np
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Set the path to your Mask R-CNN model file
MODEL_PATH = 'mask_rcnn_coco.h5'  # Replace with the path to your model file
VIDEO_FOLDER = '2017-12-17'  # Replace with the path to your video folder
OUTPUT_CSV = 'night_street_oracle.csv'

# Load the Mask R-CNN model
class InferenceConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='')

# Load pre-trained weights (COCO)
model.load_weights(MODEL_PATH, by_name=True)

# Function to process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    result_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.detect([frame], verbose=0)
        r = results[0]

        # Check if car is present in the frame
        has_car = any(class_id in r['class_ids'] for class_id in [3, 8, 6])  # COCO class IDs for cars

        result_data.append({
            'video_name': os.path.basename(video_path),
            'frame_index': frame_index,
            'hasCar': has_car,
        })

        frame_index += 1

    cap.release()
    return result_data

# Process all videos in the folder
all_video_data = []
import csv

video_files = sorted(os.listdir(VIDEO_FOLDER))

with open(OUTPUT_CSV, 'a') as csvfile:
    csvfile.write("video_name,frame_index,has_car\n")
    for video_file in video_files:
        if video_file.endswith('.mp4'):  # Assuming all videos are in mp4 format
            video_path = os.path.join(VIDEO_FOLDER, video_file)
            res = process_video(video_path)
            for r in res:
                csvfile.write(r['video_name']+','+str(r['frame_index'])+','+str(r['hasCar'])+'\n')


# Write the results to CSV


    
    # writer.writeheader()
    # writer.writerows(all_video_data)
