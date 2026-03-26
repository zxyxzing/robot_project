#!/usr/bin/env python3
"""
测试 YOLO 能否识别角色1的仿真人
"""

import sys
import cv2

# 导入你的 YOLO
from vision.yolo_detector_optimized import OptimizedYOLODetector

# 导入角色1的仿真环境
from env.simulation_env import SimulationEnv   # 根据实际路径调整

# 初始化
detector = OptimizedYOLODetector(conf_threshold=0.25)
env = SimulationEnv()

# 获取一帧图像
image = env.get_image()   # 根据实际方法名调整
print(f"图像形状: {image.shape}")

# 检测
dets = detector.detect(image)
print(f"检测到 {len(dets)} 个人")

if len(dets) > 0:
    vis = detector.visualize(image, dets)
    cv2.imwrite("result.jpg", vis)
    print("结果已保存: result.jpg")