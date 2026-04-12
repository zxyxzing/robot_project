# 视觉模块接口说明

## 输出格式
`human_info` 字典：
- `center_x`, `center_y`: 像素坐标
- `width`, `height`: 人体尺寸（像素）
- `confidence`: 置信度 0-1
- `norm_x`, `norm_y`: 归一化坐标，范围 -1~1
- `norm_size`: 归一化大小
- `timestamp`: 时间戳

## 使用方法
```python
from vision.yolo_detector_optimized import OptimizedYOLODetector
detector = OptimizedYOLODetector()
human_info = detector.detect(image)