#!/usr/bin/env python3
"""
示例程序：展示如何使用get_camera_image模块获取图像
支持单帧和连续帧获取
"""

import sys
import cv2
import numpy as np
from get_camera_image import CameraCapture

# 全局变量，用于保持摄像头实例
_camera_instance = None

def initialize_camera():
    """
    初始化摄像头

    Returns:
        CameraCapture: 摄像头实例，如果失败则返回None
    """
    global _camera_instance

    # 如果已经初始化，直接返回现有实例
    if _camera_instance is not None:
        return _camera_instance

    # 模型文件路径
    model_path = "/home/yuan/dog/venv_yolo_follow/src/model/go2_human_follow_new.xml"

    # 创建摄像头捕获器
    _camera_instance = CameraCapture(model_path, camera_name="go2_head")

    # 初始化摄像头
    if not _camera_instance.initialize():
        print("摄像头初始化失败")
        _camera_instance = None
        return None

    return _camera_instance

def get_camera_image():
    """
    获取单帧相机图像

    Returns:
        np.ndarray: 捕获的图像，如果失败则返回None
    """
    camera = initialize_camera()
    if camera is None:
        return None

    # 获取单帧图像
    image = camera.capture_frame()

    return image

def get_camera_frames(duration: int = 60, fps: int = 30):
    """
    连续获取相机图像帧

    Args:
        duration: 获取图像的持续时间（秒）
        fps: 每秒帧数

    Yields:
        np.ndarray: 捕获的图像帧
    """
    camera = initialize_camera()
    if camera is None:
        return

    start_time = 0
    frame_count = 0

    try:
        while True:
            # 获取当前时间
            current_time = 0
            if frame_count == 0:
                import time
                start_time = time.time()
                current_time = start_time
            else:
                import time
                current_time = time.time()

            # 检查是否超过持续时间
            if duration > 0 and (current_time - start_time) > duration:
                break

            # 获取图像帧
            image = camera.capture_frame()
            if image is None:
                print("获取图像失败")
                break

            # 调整图像大小
            image = cv2.resize(image, (640, 480))

            # 更新帧计数
            frame_count += 1

            # 返回图像帧
            yield image

            # 控制帧率
            import time
            time.sleep(1.0 / fps)

    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"获取图像帧出错: {str(e)}")

def main():
    """主函数 - 演示连续获取图像帧"""
    print("开始获取连续图像帧...")

    # 获取连续图像帧（持续60秒）
    for frame in get_camera_frames(duration=60, fps=30):
        # 显示图像
        cv2.imshow("Camera Frame", frame)

        # 按ESC键退出
        key = cv2.waitKey(1)
        if key == 27:  # ESC键
            print("用户按ESC键退出")
            break

    cv2.destroyAllWindows()
    print("程序结束")

if __name__ == "__main__":
    main()
