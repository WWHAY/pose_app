from ultralytics import YOLO
import os

model = YOLO('yolov8n-pose.pt')
# 指定保存路径
current_directory = os.getcwd()
save_directory = current_directory+"/output"

results = model(source="./h4.mp4",show=True,conf=0.3,save=True, name=save_directory)