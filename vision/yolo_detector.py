"""
YOLO 人体检测模块
角色4（视觉模块）使用此文件进行人体检测和跟踪

使用方法:
    from vision.yolo_detector import YOLODetector
    
    detector = YOLODetector()
    detections = detector.detect(image)
    error = detector.get_human_error(image)
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    YOLO 人体检测器
    支持检测、跟踪、偏差计算
    """
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, device=None):
        """
        初始化 YOLO 检测器
        
        Args:
            model_name: YOLO 模型名称或路径
                - 'yolov8n.pt' (轻量)
                - 'yolov8s.pt' (标准)
                - 'yolov8m.pt' (精度高)
            conf_threshold: 置信度阈值 (0-1)，越高越严格
            device: 计算设备，'cuda' 或 'cpu'，None 则自动选择
        """
        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        
        # 人类类别ID（COCO数据集中 person = 0）
        self.PERSON_CLASS_ID = 0
        
        # 跟踪相关
        self.track_history = {}
        self.next_track_id = 0
        
        print(f"✅ YOLO 检测器初始化成功")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
        print(f"   置信度阈值: {conf_threshold}")
    
    def detect(self, image, classes=None):
        """
        检测图像中的目标
        
        Args:
            image: numpy array，BGR 格式（OpenCV 默认格式）
            classes: 要检测的类别列表，如 [0] 只检测人，None 则检测所有
        
        Returns:
            List[Dict]: 检测结果列表，每个字典包含：
                - 'bbox': (x1, y1, x2, y2) 边界框坐标
                - 'confidence': 置信度
                - 'class_id': 类别ID
                - 'class_name': 类别名称
        """
        if image is None or image.size == 0:
            return []
        
        # YOLO 模型默认输入 RGB，需要从 BGR 转换
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 推理
        results = self.model(rgb_image, conf=self.conf_threshold, device=self.device, classes=classes)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def detect_humans(self, image):
        """
        只检测人体（便捷方法）
        
        Args:
            image: 输入图像
        
        Returns:
            List[Dict]: 人体检测结果列表
        """
        # classes=[0] 表示只检测 person 类别
        return self.detect(image, classes=[self.PERSON_CLASS_ID])
    
    def get_human_error(self, image, frame_center=None):
        """
        获取人体相对于画面中心的水平偏差
        
        Args:
            image: 输入图像
            frame_center: 画面中心 x 坐标，None 则自动计算
        
        Returns:
            float: 偏差值，范围 -1 到 1
                - 负数：人偏左
                - 正数：人偏右
                - None：未检测到人体
        """
        # 检测人体
        humans = self.detect_humans(image)
        
        if len(humans) == 0:
            return None
        
        # 选择最大的人体（面积最大，最可能是主要目标）
        largest_human = max(humans, key=lambda h: 
            (h['bbox'][2] - h['bbox'][0]) * (h['bbox'][3] - h['bbox'][1])
        )
        
        # 计算人体中心点
        x1, y1, x2, y2 = largest_human['bbox']
        human_center_x = (x1 + x2) // 2
        
        # 获取画面中心
        if frame_center is None:
            h, w = image.shape[:2]
            frame_center_x = w // 2
        else:
            frame_center_x = frame_center
        
        # 计算归一化偏差 (-1 到 1)
        h, w = image.shape[:2]
        error = (human_center_x - frame_center_x) / (w / 2)
        
        # 限制范围
        error = max(-1.0, min(1.0, error))
        
        return error
    
    def get_human_position(self, image):
        """
        获取人体位置信息（更详细）
        
        Args:
            image: 输入图像
        
        Returns:
            Dict: 包含以下字段，None 表示未检测到
                - 'center_x': 人体中心 x 坐标
                - 'center_y': 人体中心 y 坐标
                - 'width': 人体宽度
                - 'height': 人体高度
                - 'error': 水平偏差 (-1~1)
                - 'confidence': 检测置信度
        """
        humans = self.detect_humans(image)
        
        if len(humans) == 0:
            return None
        
        # 选择最大的人体
        largest_human = max(humans, key=lambda h:
            (h['bbox'][2] - h['bbox'][0]) * (h['bbox'][3] - h['bbox'][1])
        )
        
        x1, y1, x2, y2 = largest_human['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        h, w = image.shape[:2]
        error = (center_x - w/2) / (w/2)
        error = max(-1.0, min(1.0, error))
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'error': error,
            'confidence': largest_human['confidence'],
            'bbox': (x1, y1, x2, y2)
        }
    
    def track(self, image, prev_detections=None, iou_threshold=0.5):
        """
        跟踪已检测目标（为每个目标分配唯一ID）
        
        Args:
            image: 输入图像
            prev_detections: 上一帧的检测结果（带 track_id）
            iou_threshold: IOU 匹配阈值
        
        Returns:
            List[Dict]: 带 track_id 的跟踪结果
        """
        current_detections = self.detect_humans(image)
        
        if prev_detections is None or len(prev_detections) == 0:
            # 第一帧，为每个检测分配新ID
            for det in current_detections:
                det['track_id'] = self.next_track_id
                self.track_history[self.next_track_id] = det
                self.next_track_id += 1
            return current_detections
        
        tracked_objects = []
        used_prev = set()
        used_curr = set()
        
        # 匹配当前检测和上一帧检测
        for i, curr_det in enumerate(current_detections):
            best_iou = 0
            best_j = -1
            curr_box = curr_det['bbox']
            
            for j, prev_det in enumerate(prev_detections):
                if j in used_prev:
                    continue
                prev_box = prev_det['bbox']
                iou = self._calculate_iou(curr_box, prev_box)
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_j = j
            
            if best_j != -1:
                # 匹配成功，继承 track_id
                curr_det['track_id'] = prev_detections[best_j]['track_id']
                self.track_history[curr_det['track_id']] = curr_det
                tracked_objects.append(curr_det)
                used_prev.add(best_j)
                used_curr.add(i)
        
        # 新出现的目标，分配新ID
        for i, curr_det in enumerate(current_detections):
            if i not in used_curr:
                curr_det['track_id'] = self.next_track_id
                self.track_history[self.next_track_id] = curr_det
                self.next_track_id += 1
                tracked_objects.append(curr_det)
        
        return tracked_objects
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的 IOU（交并比）
        
        Args:
            box1, box2: (x1, y1, x2, y2) 格式
        
        Returns:
            float: IOU 值，范围 0-1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0
        
        return intersection / union
    
    def draw_detections(self, image, detections, show_center=True):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像（会被修改）
            detections: detect() 或 track() 返回的结果
            show_center: 是否绘制人体中心点
        
        Returns:
            image: 绘制后的图像
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det.get('confidence', 0)
            name = det.get('class_name', 'person')
            track_id = det.get('track_id', -1)
            
            # 根据是否有 track_id 选择颜色
            if track_id != -1:
                color = (track_id * 50 % 255, (track_id * 100) % 255, (track_id * 150) % 255)
                label = f"ID:{track_id} {name}:{conf:.2f}"
            else:
                color = (0, 255, 0)  # 绿色
                label = f"{name}:{conf:.2f}"
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            
            # 绘制标签文字
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 绘制人体中心点
            if show_center:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return image
    
    def draw_debug_info(self, image, error=None, fps=None):
        """
        绘制调试信息（画面中心线、偏差值、FPS等）
        
        Args:
            image: 输入图像
            error: 偏差值（-1~1）
            fps: 帧率
        
        Returns:
            image: 绘制后的图像
        """
        h, w = image.shape[:2]
        center_x = w // 2
        
        # 绘制画面中心线
        cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        
        # 绘制中心点
        cv2.circle(image, (center_x, h // 2), 5, (255, 255, 255), -1)
        
        # 显示偏差值
        if error is not None:
            direction = "左" if error < 0 else "右"
            cv2.putText(image, f"Error: {error:.2f} ({direction})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示 FPS
        if fps is not None:
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return image


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO 检测器测试")
    print("=" * 50)
    
    # 初始化检测器
    detector = YOLODetector(conf_threshold=0.5)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        exit()
    
    print("✅ 摄像头已打开，按 'q' 退出")
    print("   'd': 切换显示模式")
    
    show_mode = 0  # 0: 普通检测, 1: 带偏差, 2: 跟踪
    prev_tracks = None
    
    import time
    fps_counter = 0
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 计算 FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start = time.time()
        else:
            current_fps = None
        
        if show_mode == 0:
            # 普通检测模式
            detections = detector.detect_humans(frame)
            result = detector.draw_detections(frame, detections, show_center=True)
            error = detector.get_human_error(frame)
            result = detector.draw_debug_info(result, error=error, fps=current_fps)
            
        elif show_mode == 1:
            # 偏差显示模式
            error = detector.get_human_error(frame)
            pos_info = detector.get_human_position(frame)
            
            if pos_info:
                result = frame
                cv2.rectangle(result, pos_info['bbox'][:2], pos_info['bbox'][2:], (0, 255, 0), 2)
                cv2.circle(result, (pos_info['center_x'], pos_info['center_y']), 5, (0, 0, 255), -1)
                cv2.putText(result, f"Error: {pos_info['error']:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                result = frame
                cv2.putText(result, "No human detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            result = detector.draw_debug_info(result, error=error, fps=current_fps)
            
        else:
            # 跟踪模式
            tracks = detector.track(frame, prev_detections=prev_tracks)
            prev_tracks = tracks
            result = detector.draw_detections(frame, tracks, show_center=True)
            error = detector.get_human_error(frame)
            result = detector.draw_debug_info(result, error=error, fps=current_fps)
        
        # 显示模式提示
        mode_names = ["Normal", "Error Mode", "Tracking"]
        cv2.putText(result, f"Mode: {mode_names[show_mode]}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('YOLO Detector', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_mode = (show_mode + 1) % 3
            print(f"切换到模式: {mode_names[show_mode]}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 测试结束")