import cv2
import torch
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('yolov8n-pose.pt')

# 初始化壶铃摆动次数
swing_count = 0
previous_position = None

# 读取视频文件
def process_video(video_path):
    global swing_count, previous_position

    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 YOLOv8 进行姿态检测
        results = model(frame)

        # 获取所有检测到的人的关键点信息
        for result in results:
            if result.keypoints is not None:  # 确保关键点存在
                keypoints = result.keypoints.cpu().numpy()

                # 假设我们只关心第一个人的关键点
                if keypoints.shape[0] > 0:
                    current_position = keypoints[0]  # 获取第一个人的关键点
                    y_position = current_position[5][1]  # 获取手腕的 y 坐标（以 index 5 为例）

                    if previous_position is not None:
                        # 判断摆动方向和增量
                        if previous_position > y_position + 20:  # 上摆
                            swing_count += 1
                            print(f"壶铃摆动次数: {swing_count}")

                    previous_position = y_position

        # 显示检测结果
        cv2.imshow('壶铃摆动检测', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置视频文件路径
    video_path = './video.mp4'
    process_video(video_path)
