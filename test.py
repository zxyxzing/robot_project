#!/usr/bin/env python3
"""
测试 YOLO 能否识别角色1的仿真人图片
"""

import cv2
import sys
import os

# 导入你的 YOLO
from vision.yolo_detector_stable import OptimizedYOLODetector
from vision.vision_to_control import VisionToControl

def main():
    print("=" * 60)
    print("测试 YOLO 识别角色1的仿真人图片")
    print("=" * 60)
    
    # 1. 找到图片
    image_path = None
    possible_paths = [                 # 可能在 env 目录
        "vision/human.jpg",                     # 可能在根目录
        "test_image.jpg",                # 其他可能
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    # 如果找不到，让用户输入
    if image_path is None:
        print("请提供图片路径:")
        print("可以执行: find . -name '*.jpg' -o -name '*.png'")
        image_path = input("请输入图片路径: ")
    
    print(f"\n[1] 加载图片: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"✗ 无法加载图片: {image_path}")
        print("请检查文件是否存在")
        return
    
    print(f"   图像尺寸: {img.shape}")
    
    # 2. 初始化 YOLO 检测器
    print("\n[2] 初始化 YOLO 检测器...")
    detector = OptimizedYOLODetector(conf_threshold=0.25, input_size=320)
    
    # 3. 运行检测
    print("\n[3] 运行 YOLO 检测...")
    detections = detector.detect(img)
    print(f"   检测到 {len(detections)} 个人")
    
    # 4. 显示检测结果
    if len(detections) > 0:
        print("\n[4] 检测结果:")
        for i, det in enumerate(detections):
            cx, cy, w, h, conf = det
            print(f"   第{i+1}个: 中心({cx:.0f}, {cy:.0f}), 大小({w:.0f}x{h:.0f}), 置信度{conf:.2f}")
        
        # 可视化并保存
        vis_img = detector.visualize(img, detections)
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, vis_img)
        print(f"\n   结果已保存: {output_path}")
        
        # 测试适配器
        print("\n[5] 测试适配器输出...")
        adapter = VisionToControl(detector)
        human_info = adapter.process_frame(img)
        if human_info:
            print(f"   归一化坐标: ({human_info['norm_x']:.2f}, {human_info['norm_y']:.2f})")
            print(f"   归一化大小: {human_info['norm_size']:.2f}")
            print(f"   置信度: {human_info['confidence']:.2f}")
        
        print("\n✅ 测试完成！YOLO 成功检测到仿真人！")
    else:
        print("\n❌ 未检测到人")
        print("\n可能原因:")
        print("  1. 图片中的仿真人不逼真，YOLO 认不出来")
        print("  2. 图片尺寸太小或太大")
        print("  3. 需要降低置信度阈值")
        print("\n尝试降低阈值:")
        detector_low = OptimizedYOLODetector(conf_threshold=0.1, input_size=320)
        detections_low = detector_low.detect(img)
        print(f"   阈值0.1时检测到 {len(detections_low)} 个人")
        
        if len(detections_low) > 0:
            vis_img = detector_low.visualize(img, detections_low)
            cv2.imwrite("detection_result_low.jpg", vis_img)
            print("   结果已保存: detection_result_low.jpg")

if __name__ == "__main__":
    main()