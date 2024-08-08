import cv2
import torch
import os
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('yolov8n-pose.pt')

# 初始化壶铃摆动次数
swing_count = 0
previous_distance = None
in_swing = False  # 记录当前是否处于摆动中

# 读取视频文件
def process_video(video_path):
    global swing_count, previous_distance, in_swing

    cap = cv2.VideoCapture(video_path)
    


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 设置标准线的 y 坐标 (可以调整这个值)

        # 使用 YOLOv8 进行姿态检测
        results = model(frame)
        # positions = results[0].boxes.xyxy
        # print(f" postions:{positions}")
        # height = int(positions[3]-positions[1])
        # standard_line = int(height * 0.8)  # 中轴线
        # print(f"=== height{height}, postions:{positions}, ys:{positions[3]}, y1:{positions[1]}")
        # 获取所有检测到的人的关键点信息
        for result in results:
            if result.keypoints is not None:  # 确保关键点存在
                keypoints = result.keypoints.cpu().numpy()
                
                # 假设我们只关心第一个人的关键点
                if keypoints.shape[0] > 0:
                    # 获取右髋部（Index 12）和右手腕（Index 16）的坐标
                    right_hip = keypoints[0][12]
                    right_wrist = keypoints[0][16]

                    # 计算髋部和手腕之间的垂直距离
                    current_distance = abs(right_hip[1] - right_wrist[1])

                    # if previous_distance is not None:
                    #     # 判断手腕是否超过髋部一定距离，视为一次有效摆动
                    #     if current_distance > previous_distance + 10:
                    #         swing_count += 1
                    #         print(f"壶铃摆动次数: {swing_count}")

                    # previous_distance = current_distance
                    if current_distance > 5 and not in_swing:
                        swing_count+=1
                        in_swing = True
                    elif current_distance <= 1:
                        in_swing = False
                    # current_position = keypoints[0]  # 获取第一个人的关键点

                    # # 选择特定的关键点 (例如手腕) 来计算摆动
                    # y_position = current_position[9][1]  # 获取手腕的 y 坐标（以 index 5 为例）

                    # # 判断摆动方向和增量
                    # if previous_position is not None:
                    #     if y_position < standard_line and not in_swing:
                    #         swing_count += 1
                    #         in_swing = True
                    #     elif y_position >= standard_line:
                    #         in_swing = False

                    # previous_position = y_position

                    # 绘制关键点和骨架
                    # for keypoint in current_position:
                    #     x, y, _ = keypoint  # 解包每个关键点的信息 (x, y, confidence)
                    #     cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # 关键点

                    # # 连接关键点，绘制骨架
                    # connections = [
                    #     (0, 1), (1, 2), (2, 3), (3, 4),
                    #     (5, 6), (6, 7), (7, 8), (5, 11),
                    #     (11, 12), (12, 13), (13, 14),
                    #     (5, 2), (2, 9), (9, 10), (10, 9),
                    #     (9, 11), (11, 12)
                    # ]
                    # for start, end in connections:
                    #     if start < len(current_position) and end < len(current_position):
                    #         start_point = (int(current_position[start][0]), int(current_position[start][1]))
                    #         end_point = (int(current_position[end][0]), int(current_position[end][1]))
                    #         cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # 骨架连接线


        # 在画面上显示壶铃摆动次数
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, f'count: {swing_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # 显示检测结果
        cv2.imshow('count', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置视频文件路径
    video_path = './h4.mp4'
    process_video(video_path)
