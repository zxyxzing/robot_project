#!/usr/bin/env python3
"""
yolo_detector_optimized.py
YOLOv8人体检测模块（优化版）
针对速度优化，目标帧率≥30FPS
"""

from ultralytics import YOLO
import numpy as np
import cv2
import time
import logging

# 关闭YOLO的日志输出
logging.getLogger('ultralytics').setLevel(logging.WARNING)

class OptimizedYOLODetector:
    """优化的YOLO检测器"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.45, 
                 device='cpu', input_size=320):
        """
        初始化优化检测器
        
        参数:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值（降低到0.45减少漏检）
            device: 推理设备
            input_size: 输入图像尺寸（默认320x320，可提高帧率）
        """
        print(f"加载优化版YOLO模型...")
        print(f"  输入尺寸: {input_size}x{input_size}")
        print(f"  置信度阈值: {conf_threshold}")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.input_size = input_size
        self.person_class_id = 0
        
        # 性能统计
        self.inference_times = []
        self.frame_count = 0
        
        # 预热模型（跑一次推理，让模型加载到内存）
        dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        print("✓ 模型预热完成")
        print("-" * 50)
    
    def detect(self, image):
        """
        优化的检测方法
        """
        self.frame_count += 1
        start_time = time.time()
        
        # 使用固定输入尺寸加速
        results = self.model(image, 
                            conf=self.conf_threshold, 
                            device=self.device,
                            imgsz=self.input_size,  # 固定输入尺寸，关键优化！
                            verbose=False)[0]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # 解析结果
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls[0]) != self.person_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                
                # 只返回高置信度的检测
                if confidence >= self.conf_threshold:
                    detections.append([center_x, center_y, width, height, confidence])
        
        return detections
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.inference_times:
            return 0, 0
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        return avg_time, fps
    
    def visualize(self, image, detections):
        """可视化检测结果"""
        vis_image = image.copy()
        
        for det in detections:
            center_x, center_y, width, height, conf = det
            
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            # 边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 中心点
            cv2.circle(vis_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            # 置信度
            cv2.putText(vis_image, f"{conf:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示性能
        avg_time, fps = self.get_performance_stats()
        cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_image, f"Input: {self.input_size}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_image


# 性能对比测试
if __name__ == "__main__":
    print("=" * 60)
    print("性能优化测试")
    print("=" * 60)
    
    from sim_video_client import SimVideoClient
    
    camera = SimVideoClient()
    
    # 测试不同输入尺寸的性能
    input_sizes = [640, 480, 320]
    
    for size in input_sizes:
        print(f"\n测试输入尺寸: {size}x{size}")
        detector = OptimizedYOLODetector(input_size=size)
        
        # 测试100帧
        times = []
        for i in range(50):
            image = camera.GetImageSample()
            start = time.time()
            dets = detector.detect(image)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        fps = 1.0 / np.mean(times)
        print(f"  平均推理时间: {avg_time:.1f} ms")
        print(f"  帧率: {fps:.1f} FPS")
        print(f"  检测到人数: {len(dets) if 'dets' in locals() else 0}")