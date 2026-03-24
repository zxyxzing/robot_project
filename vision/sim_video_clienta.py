#!/usr/bin/env python3
"""
sim_video_client.py
模拟机器人相机图像获取模块
用于四足机器人人体跟随项目
"""

import numpy as np
import cv2
import time
import os
from datetime import datetime

class SimVideoClient:
    """模拟视频客户端类"""
    
    def __init__(self, width=1920, height=1080, fps=200):
        """
        初始化模拟视频客户端
        
        参数:
            width: 图像宽度（像素），默认1920
            height: 图像高度（像素），默认1080
            fps: 模拟帧率，默认200Hz（项目要求）
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps  # 每帧间隔时间
        
        # 帧率控制
        self.last_time = time.time()
        self.frame_count = 0
        
        # 模拟人形移动参数
        self.human_x = width // 2  # 起始x坐标（画面中心）
        self.human_y = height // 2  # 起始y坐标（画面中心）
        self.move_direction_x = 1   # x方向移动方向：1向右，-1向左
        self.move_direction_y = 1   # y方向移动方向：1向下，-1向上
        self.move_speed = 5         # 移动速度（像素/帧）
        
        # 人形尺寸（像素）
        self.human_width = 200
        self.human_height = 400
        
        # 性能统计
        self.frame_times = []
        self.log_file = None
        
        print(f"SimVideoClient 初始化完成")
        print(f"  图像尺寸: {width} x {height}")
        print(f"  目标帧率: {fps} FPS")
        print(f"  帧间隔: {self.frame_interval*1000:.2f} ms")
        print(f"  人形起始位置: ({self.human_x}, {self.human_y})")
        print("-" * 50)
    
    def GetImageSample(self):
        """
        获取一帧模拟图像
        
        返回:
            numpy数组，形状 (height, width, 3)，RGB格式
        """
        # 帧率控制：确保不超过设定fps
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed < self.frame_interval:
            # 如果比预期快，就睡一会儿
            sleep_time = self.frame_interval - elapsed
            time.sleep(sleep_time)
        
        # 更新计时
        self.last_time = time.time()
        self.frame_count += 1
        
        # 记录实际帧率
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        self.frame_times.append(self.last_time)
        
        # 生成图像
        image = self._generate_image()
        
        return image
    
    def _generate_image(self):
        """
        生成一帧包含人形的模拟图像
        """
        # 创建灰色背景 (128,128,128)
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        
        # 更新人形位置（模拟移动）
        self._update_human_position()
        
        # 绘制人形（用绿色矩形表示）
        # 头部（圆形）
        head_center = (self.human_x, self.human_y - self.human_height//3)
        cv2.circle(image, head_center, self.human_width//6, (0, 255, 0), -1)
        
        # 身体（矩形）
        body_x1 = self.human_x - self.human_width//4
        body_y1 = self.human_y - self.human_height//3
        body_x2 = self.human_x + self.human_width//4
        body_y2 = self.human_y + self.human_height//3
        cv2.rectangle(image, (body_x1, body_y1), (body_x2, body_y2), (0, 255, 0), -1)
        
        # 左臂
        arm_x1 = self.human_x - self.human_width//3
        arm_y1 = self.human_y - self.human_height//6
        arm_x2 = self.human_x - self.human_width//4
        arm_y2 = self.human_y
        cv2.rectangle(image, (arm_x1, arm_y1), (arm_x2, arm_y2), (0, 255, 0), -1)
        
        # 右臂
        arm_x1 = self.human_x + self.human_width//4
        arm_y1 = self.human_y - self.human_height//6
        arm_x2 = self.human_x + self.human_width//3
        arm_y2 = self.human_y
        cv2.rectangle(image, (arm_x1, arm_y1), (arm_x2, arm_y2), (0, 255, 0), -1)
        
        # 左腿
        leg_x1 = self.human_x - self.human_width//4
        leg_y1 = self.human_y + self.human_height//3
        leg_x2 = self.human_x - self.human_width//6
        leg_y2 = self.human_y + self.human_height//2
        cv2.rectangle(image, (leg_x1, leg_y1), (leg_x2, leg_y2), (0, 255, 0), -1)
        
        # 右腿
        leg_x1 = self.human_x + self.human_width//6
        leg_y1 = self.human_y + self.human_height//3
        leg_x2 = self.human_x + self.human_width//4
        leg_y2 = self.human_y + self.human_height//2
        cv2.rectangle(image, (leg_x1, leg_y1), (leg_x2, leg_y2), (0, 255, 0), -1)
        
        # 添加中心点标记（红色）
        cv2.circle(image, (self.human_x, self.human_y), 5, (0, 0, 255), -1)
        
        # 添加边框
        cv2.rectangle(image, 
                     (self.human_x - self.human_width//2, self.human_y - self.human_height//2),
                     (self.human_x + self.human_width//2, self.human_y + self.human_height//2),
                     (255, 0, 0), 2)
        
        return image
    
    def _update_human_position(self):
        """
        更新人形位置（模拟移动）
        """
        # x方向移动
        self.human_x += self.move_speed * self.move_direction_x
        
        # 边界检查（不出画面）
        if self.human_x > self.width - self.human_width//2:
            self.human_x = self.width - self.human_width//2
            self.move_direction_x *= -1
        elif self.human_x < self.human_width//2:
            self.human_x = self.human_width//2
            self.move_direction_x *= -1
        
        # y方向移动
        self.human_y += self.move_speed * self.move_direction_y
        
        # 边界检查
        if self.human_y > self.height - self.human_height//2:
            self.human_y = self.height - self.human_height//2
            self.move_direction_y *= -1
        elif self.human_y < self.human_height//2:
            self.human_y = self.human_height//2
            self.move_direction_y *= -1
    
    def get_current_fps(self):
        """
        计算当前实际帧率
        """
        if len(self.frame_times) < 2:
            return 0
        
        # 用最近几帧计算
        time_diff = self.frame_times[-1] - self.frame_times[0]
        frame_count = len(self.frame_times)
        
        if time_diff > 0:
            return frame_count / time_diff
        return 0
    
    def save_frame(self, filename=None):
        """
        保存当前帧到文件
        
        参数:
            filename: 保存的文件名，None时自动生成
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/frame_{timestamp}.jpg"
        
        image = self.GetImageSample()
        cv2.imwrite(filename, image)
        print(f"帧已保存到: {filename}")
        return filename
    
    def test(self, duration=10, show_display=True):
        """
        测试函数：连续获取图像并显示
        
        参数:
            duration: 测试时长（秒）
            show_display: 是否显示图像窗口
        """
        print(f"\n开始测试 {duration} 秒...")
        print("按 'q' 键提前退出")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # 获取图像
            image = self.GetImageSample()
            frame_count += 1
            
            if show_display:
                # 在图像上添加信息
                display_image = image.copy()
                
                # 添加文本信息
                current_fps = self.get_current_fps()
                elapsed = time.time() - start_time
                
                cv2.putText(display_image, 
                           f"FPS: {current_fps:.1f}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(display_image, 
                           f"Frame: {frame_count}", 
                           (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(display_image, 
                           f"Time: {elapsed:.1f}/{duration}s", 
                           (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(display_image, 
                           f"Human pos: ({self.human_x}, {self.human_y})", 
                           (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示图像
                cv2.imshow('Simulated Camera', display_image)
                
                # 检查按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        
        # 统计结果
        actual_fps = frame_count / duration
        print(f"\n测试完成！")
        print(f"  总帧数: {frame_count}")
        print(f"  平均帧率: {actual_fps:.2f} FPS")
        print(f"  目标帧率: {self.fps} FPS")
        print(f"  达标: {'✓' if actual_fps >= self.fps else '✗'}")
        
        return actual_fps
    
    def validate_image_format(self, image):
        """
        验证图像格式是否符合要求
        
        参数:
            image: 要验证的图像
        
        返回:
            True: 格式正确
            False: 格式错误
        """
        try:
            # 检查形状
            assert image.shape == (self.height, self.width, 3), \
                   f"图像尺寸错误: {image.shape}，应为 ({self.height}, {self.width}, 3)"
            
            # 检查数据类型
            assert image.dtype == np.uint8, \
                   f"图像数据类型错误: {image.dtype}，应为 uint8"
            
            # 检查值范围
            assert image.min() >= 0 and image.max() <= 255, \
                   f"图像值范围错误: [{image.min()}, {image.max()}]，应为 [0, 255]"
            
            print("✓ 图像格式验证通过")
            return True
            
        except AssertionError as e:
            print(f"✗ 图像格式验证失败: {e}")
            return False


# 测试代码（当直接运行此文件时执行）
if __name__ == "__main__":
    print("=" * 60)
    print("SimVideoClient 测试程序")
    print("=" * 60)
    
    # 1. 创建客户端实例
    print("\n[1] 初始化 SimVideoClient...")
    client = SimVideoClient(width=1920, height=1080, fps=200)
    
    # 2. 测试单帧获取
    print("\n[2] 测试单帧获取...")
    image = client.GetImageSample()
    print(f"   获取到图像: 形状 {image.shape}, 类型 {image.dtype}")
    
    # 3. 验证格式
    print("\n[3] 验证图像格式...")
    client.validate_image_format(image)
    
    # 4. 保存一帧测试
    print("\n[4] 保存测试帧...")
    client.save_frame()
    
    # 5. 连续测试
    print("\n[5] 连续运行测试 (5秒)...")
    client.test(duration=5, show_display=True)
    
    print("\n" + "=" * 60)
    print("测试全部完成")
    print("=" * 60)