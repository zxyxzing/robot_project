#!/usr/bin/env python3
"""
yolo_detector.py
YOLOv8人体检测模块
用于四足机器人人体跟随项目
"""

from ultralytics import YOLO
import numpy as np
import cv2
import time
import os

class YOLODetector:
    """YOLO目标检测器类"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, device='cpu'):
        """
        初始化YOLO检测器
        
        参数:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            device: 推理设备 ('cpu' 或 'cuda')
        """
        print(f"正在加载YOLOv8模型: {model_path}")
        print(f"  置信度阈值: {conf_threshold}")
        print(f"  使用设备: {device}")
        
        # 加载模型
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # COCO数据集中人的类别ID是0
        self.person_class_id = 0
        
        # 性能统计
        self.inference_times = []
        self.frame_count = 0
        
        # 如果模型文件不存在，会自动下载
        print("✓ 模型加载成功！")
        print("-" * 50)
    
    def detect(self, image):
        """
        对单张图像进行检测
        
        参数:
            image: 输入图像 (H, W, 3), RGB格式
        
        返回:
            detections: 检测结果列表，每个元素为 [center_x, center_y, width, height, confidence]
        """
        self.frame_count += 1
        start_time = time.time()
        
        # YOLO推理
        results = self.model(image, 
                            conf=self.conf_threshold, 
                            device=self.device,
                            verbose=False)[0]  # 关闭日志输出
        
        # 记录推理时间
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # 解析检测结果
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                # 检查类别是否为人
                if int(box.cls[0]) != self.person_class_id:
                    continue
                
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                
                # 转换为中心点坐标 + 宽高
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                
                detections.append([center_x, center_y, width, height, confidence])
        
        return detections
    
    def detect_batch(self, images):
        """
        批量检测（用于测试性能）
        
        参数:
            images: 图像列表
        
        返回:
            检测结果列表
        """
        start_time = time.time()
        
        results = self.model(images, 
                            conf=self.conf_threshold, 
                            device=self.device,
                            verbose=False)
        
        batch_time = time.time() - start_time
        print(f"批量检测 {len(images)} 张图，耗时: {batch_time*1000:.1f}ms，平均: {batch_time/len(images)*1000:.1f}ms/张")
        
        all_detections = []
        for result in results:
            frame_dets = []
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) != self.person_class_id:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2
                    frame_dets.append([center_x, center_y, width, height, conf])
            all_detections.append(frame_dets)
        
        return all_detections
    
    def get_performance_stats(self):
        """
        获取性能统计信息
        
        返回:
            avg_time: 平均推理时间（秒）
            fps: 帧率
        """
        if not self.inference_times:
            return 0, 0
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        return avg_time, fps
    
    def visualize(self, image, detections, show_info=True):
        """
        在图像上绘制检测结果
        
        参数:
            image: 原始图像
            detections: detect()返回的结果
            show_info: 是否显示检测信息
        
        返回:
            vis_image: 绘制后的图像
        """
        vis_image = image.copy()
        
        for det in detections:
            center_x, center_y, width, height, conf = det
            
            # 计算左上角坐标
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            # 绘制边界框（绿色）
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制中心点（红色）
            cv2.circle(vis_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # 显示置信度
            label = f"Person: {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if show_info and detections:
            # 显示检测到的人数
            cv2.putText(vis_image, f"Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis_image
    
    def reset_stats(self):
        """重置性能统计"""
        self.inference_times = []
        self.frame_count = 0
        print("性能统计已重置")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("YOLODetector 测试程序")
    print("=" * 60)
    
    # 导入模拟相机
    from sim_video_client import SimVideoClient
    
    # 1. 创建检测器
    print("\n[1] 初始化 YOLODetector...")
    detector = YOLODetector(conf_threshold=0.5, device='cpu')
    
    # 2. 创建模拟相机
    print("\n[2] 初始化 SimVideoClient...")
    camera = SimVideoClient()
    
    # 3. 单帧测试
    print("\n[3] 单帧检测测试...")
    image = camera.GetImageSample()
    detections = detector.detect(image)
    print(f"   检测到 {len(detections)} 个人")
    
    # 4. 性能测试
    print("\n[4] 性能测试 (30帧)...")
    times = []
    for i in range(30):
        image = camera.GetImageSample()
        start = time.time()
        dets = detector.detect(image)
        infer_time = time.time() - start
        times.append(infer_time)
        
        if (i+1) % 10 == 0:
            print(f"   已测试 {i+1} 帧")
    
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    fps = 1.0 / np.mean(times)
    print(f"\n  平均推理时间: {avg_time:.1f} ms")
    print(f"  估计帧率: {fps:.1f} FPS")
    
    # 5. 实时检测演示
    print("\n[5] 实时检测演示 (10秒)...")
    print("   按 'q' 键退出")
    
    import cv2
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 10:
        # 获取图像
        image = camera.GetImageSample()
        
        # 检测
        detections = detector.detect(image)
        
        # 可视化
        vis_image = detector.visualize(image, detections)
        
        # 显示性能
        avg_t, current_fps = detector.get_performance_stats()
        cv2.putText(vis_image, f"FPS: {current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示
        cv2.imshow('YOLO Detection Test', vis_image)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    print(f"\n  实时测试完成，处理了 {frame_count} 帧")
    print("\n" + "=" * 60)
    print("测试全部完成")
    print("=" * 60)