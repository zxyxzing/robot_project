#!/usr/bin/env python3
"""
vision_to_control.py
视觉模块到控制模块的适配器（调试版）
"""

import numpy as np
import time
import cv2
from collections import deque

class VisionToControl:
    def __init__(self, detector, image_width=1920, image_height=1080, 
                 smooth_window=5, max_lost_frames=3):
        self.detector = detector
        self.image_width = image_width
        self.image_height = image_height
        self.max_lost_frames = max_lost_frames
        
        self.lost_count = 0
        self.last_valid_info = None
        
        self.smooth_window = smooth_window
        self.center_x_history = deque(maxlen=smooth_window)
        self.center_y_history = deque(maxlen=smooth_window)
        self.size_history = deque(maxlen=smooth_window)
        
        # 性能统计
        self.process_times = []
        self.total_frames = 0
        self.detected_frames = 0
        
        # 调试计数
        self.debug_frame_count = 0
        
        print("=" * 60)
        print("VisionToControl 适配器初始化（调试模式）")
        print(f"  图像尺寸: {image_width}x{image_height}")
        print(f"  平滑窗口: {smooth_window}")
        print(f"  最大丢失帧: {max_lost_frames}")
        print(f"  检测器类型: {type(detector).__name__}")
        print("=" * 60)
    
    def process_frame(self, image):
        """
        处理一帧图像，返回给控制器的信息
        """
        start_time = time.time()
        self.total_frames += 1
        self.debug_frame_count += 1
        
        # 调试：每30帧打印一次状态
        if self.debug_frame_count % 30 == 0:
            print(f"\n[调试] 第 {self.debug_frame_count} 帧")
            print(f"  图像形状: {image.shape}")
            print(f"  图像类型: {image.dtype}")
            print(f"  像素范围: [{image.min()}, {image.max()}]")
        
        # 1. 检测
        try:
            detections = self.detector.detect(image)
            
            # 调试：打印检测结果
            if self.debug_frame_count % 30 == 0:
                print(f"  检测到 {len(detections)} 个人")
                if len(detections) > 0:
                    print(f"  第一个检测结果: {detections[0]}")
                    
        except Exception as e:
            print(f"检测出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 2. 处理检测结果
        if len(detections) > 0:
            self.detected_frames += 1
            self.lost_count = 0
            
            # 取置信度最高的
            best_det = max(detections, key=lambda x: x[4])
            center_x, center_y, width, height, conf = best_det
            
            # 更新平滑队列
            self.center_x_history.append(center_x)
            self.center_y_history.append(center_y)
            self.size_history.append(np.sqrt(width * height))
            
            # 计算平滑值
            smooth_x = np.mean(self.center_x_history) if self.center_x_history else center_x
            smooth_y = np.mean(self.center_y_history) if self.center_y_history else center_y
            smooth_size = np.mean(self.size_history) if self.size_history else np.sqrt(width * height)
            
            # 构建信息
            human_info = {
                'center_x': smooth_x,
                'center_y': smooth_y,
                'width': width,
                'height': height,
                'confidence': conf,
                'timestamp': time.time(),
                'norm_x': self._normalize_x(smooth_x),
                'norm_y': self._normalize_y(smooth_y),
                'norm_size': self._normalize_size(smooth_size),
                'raw_detections': len(detections)
            }
            
            self.last_valid_info = human_info
            process_time = time.time() - start_time
            self.process_times.append(process_time)
            
            return human_info
        
        else:
            # 没有检测到人
            self.lost_count += 1
            
            if self.debug_frame_count % 30 == 0:
                print(f"  未检测到人，丢失计数: {self.lost_count}")
            
            if self.lost_count <= self.max_lost_frames and self.last_valid_info:
                last_info = self.last_valid_info.copy()
                last_info['confidence'] *= 0.8
                last_info['predicted'] = True
                last_info['timestamp'] = time.time()
                return last_info
            else:
                self.last_valid_info = None
                self._clear_history()
                return None
    
    def _normalize_x(self, x):
        return (x / self.image_width - 0.5) * 2
    
    def _normalize_y(self, y):
        return (0.5 - y / self.image_height) * 2
    
    def _normalize_size(self, size):
        diag = np.sqrt(self.image_width**2 + self.image_height**2)
        return min(size / diag * 2, 1.0)
    
    def _clear_history(self):
        self.center_x_history.clear()
        self.center_y_history.clear()
        self.size_history.clear()
    
    def get_statistics(self):
        accuracy = self.detected_frames / max(self.total_frames, 1) * 100
        avg_process = np.mean(self.process_times) * 1000 if self.process_times else 0
        
        return {
            'total_frames': self.total_frames,
            'detected_frames': self.detected_frames,
            'accuracy': accuracy,
            'avg_process_time_ms': avg_process,
            'lost_count': self.lost_count
        }
    
    def reset(self):
        self.lost_count = 0
        self.last_valid_info = None
        self._clear_history()
        self.process_times = []
        self.total_frames = 0
        self.detected_frames = 0
        print("适配器已重置")


# 测试代码（简化版）
if __name__ == "__main__":
    print("=" * 60)
    print("VisionToControl 调试测试")
    print("=" * 60)
    
    from sim_video_client import SimVideoClient
    from yolo_detector_optimized import OptimizedYOLODetector
    import cv2
    
    # 初始化
    print("\n初始化相机...")
    camera = SimVideoClient()
    
    print("初始化检测器...")
    detector = OptimizedYOLODetector(input_size=320)
    
    print("初始化适配器...")
    adapter = VisionToControl(detector)
    
    print("\n开始实时测试（按 'q' 退出）...")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        frame_count += 1
        
        # 获取图像
        image = camera.GetImageSample()
        
        # 处理
        human_info = adapter.process_frame(image)
        
        # 可视化
        vis_image = image.copy()
        
        if human_info:
            # 绘制检测信息
            x = int(human_info['center_x'])
            y = int(human_info['center_y'])
            w = int(human_info['width'])
            h = int(human_info['height'])
            
            # 边界框
            cv2.rectangle(vis_image, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
            # 中心点
            cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)
            
            # 显示归一化坐标（加大字体，放在明显位置）
            norm_text = f"norm: ({human_info['norm_x']:.2f}, {human_info['norm_y']:.2f})"
            size_text = f"size: {human_info['norm_size']:.2f}"
            
            # 在图像顶部用大字体显示
            cv2.putText(vis_image, norm_text, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(vis_image, size_text, (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # 也在框旁边显示（小字体）
            cv2.putText(vis_image, norm_text, (x+20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 显示是否预测
            if human_info.get('predicted', False):
                cv2.putText(vis_image, "PREDICTED", (x-50, y-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # 没检测到时显示提示
            cv2.putText(vis_image, "NO DETECTION", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示统计
        stats = adapter.get_statistics()
        cv2.putText(vis_image, f"Accuracy: {stats['accuracy']:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_image, f"Detected: {stats['detected_frames']}/{stats['total_frames']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis_image, f"Lost: {stats['lost_count']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示当前帧数
        cv2.putText(vis_image, f"Frame: {frame_count}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow('Vision to Control Test (Debug)', vis_image)
        
        # 每30帧打印一次终端调试信息
        if frame_count % 30 == 0:
            print(f"\n[状态] 帧 {frame_count}")
            print(f"  Accuracy: {stats['accuracy']:.1f}%")
            print(f"  检测到: {stats['detected_frames']}/{stats['total_frames']}")
            print(f"  有人类信息: {human_info is not None}")
            if human_info:
                print(f"  norm: ({human_info['norm_x']:.2f}, {human_info['norm_y']:.2f})")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("测试结束")
    print("最终统计:")
    stats = adapter.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=" * 60)