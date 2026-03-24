#!/usr/bin/env python3
"""
yolo_detector_stable.py
稳定版YOLO检测器
添加连续帧确认，防止抖动
"""

import numpy as np
import cv2
from collections import deque
from yolo_detector_optimized import OptimizedYOLODetector

class StableYOLODetector(OptimizedYOLODetector):
    """稳定版检测器：连续3帧确认"""
    
    def __init__(self, history_length=3, **kwargs):
        """
        初始化稳定检测器
        
        参数:
            history_length: 需要连续多少帧确认
            **kwargs: 传给OptimizedYOLODetector的参数
        """
        super().__init__(**kwargs)
        self.history_length = history_length
        self.detection_history = deque(maxlen=history_length)
        self.last_valid_detections = []
        self.lost_frame_count = 0
        self.max_lost_frames = 3  # 连续丢失3帧才认为真的丢失
        
    def detect_stable(self, image):
        """
        稳定检测：连续多帧都检测到才输出
        
        返回:
            detections: 稳定后的检测结果
            status: 'detected', 'lost', 'tracking'
        """
        # 当前帧检测
        current_dets = self.detect(image)
        
        # 记录是否有检测到人
        has_person = len(current_dets) > 0
        self.detection_history.append(has_person)
        
        # 情况1：当前帧检测到人
        if has_person:
            self.lost_frame_count = 0
            self.last_valid_detections = current_dets
            
            # 检查历史记录
            if len(self.detection_history) == self.history_length:
                # 如果最近history_length帧都检测到
                if all(self.detection_history):
                    return current_dets, 'detected'
                else:
                    # 刚恢复检测，但还不够稳定
                    return current_dets, 'tracking'
            else:
                return current_dets, 'tracking'
        
        # 情况2：当前帧没检测到人
        else:
            self.lost_frame_count += 1
            
            # 如果连续丢失不超过max_lost_frames，返回上一次的有效结果
            if self.lost_frame_count <= self.max_lost_frames and self.last_valid_detections:
                return self.last_valid_detections, 'tracking'
            else:
                # 真的丢失了
                self.last_valid_detections = []
                return [], 'lost'
    
    def reset(self):
        """重置状态"""
        self.detection_history.clear()
        self.last_valid_detections = []
        self.lost_frame_count = 0
        print("稳定检测器已重置")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("稳定检测器测试")
    print("=" * 60)
    
    from sim_video_client import SimVideoClient
    import time
    
    camera = SimVideoClient()
    detector = StableYOLODetector(history_length=3, input_size=320)
    
    print("\n开始测试（按 'q' 退出）...")
    
    # 模拟遮挡场景
    遮挡 = False
    遮挡计数器 = 0
    
    while True:
        # 获取图像
        image = camera.GetImageSample()
        
        # 模拟遮挡：每50帧遮挡10帧
        遮挡计数器 += 1
        if 遮挡计数器 % 50 == 0:
            遮挡 = True
            print("\n⚠️ 模拟遮挡开始...")
        elif 遮挡计数器 % 60 == 0:
            遮挡 = False
            print("✓ 遮挡解除")
        
        # 如果遮挡，返回黑图（模拟目标丢失）
        if 遮挡:
            test_image = np.zeros_like(image)
        else:
            test_image = image
        
        # 稳定检测
        detections, status = detector.detect_stable(test_image)
        
        # 可视化
        vis_image = test_image.copy()
        if detections:
            vis_image = detector.visualize(vis_image, detections)
        
        # 显示状态
        status_colors = {
            'detected': (0, 255, 0),   # 绿色：稳定检测
            'tracking': (255, 255, 0), # 黄色：追踪中
            'lost': (0, 0, 255)        # 红色：丢失
        }
        color = status_colors.get(status, (255, 255, 255))
        cv2.putText(vis_image, f"Status: {status}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_image, f"Lost count: {detector.lost_frame_count}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示
        cv2.imshow('Stable Detection Test', vis_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()