#!/usr/bin/env python3
"""
sim_video_client_realistic.py
更逼真的人体模拟器（让YOLO能检测到）
"""

import numpy as np
import cv2
import time

class SimVideoClient:
    """生成逼真人体图像，让YOLO能检测到"""
    
    def __init__(self, width=1920, height=1080, fps=200):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_time = time.time()
        
        # 人形参数
        self.human_x = width // 2
        self.human_y = height // 2
        self.move_direction = 1
        self.move_speed = 3
        self.frame_count = 0
        
        print("RealisticSimVideoClient 初始化完成")
        print(f"  图像尺寸: {width}x{height}")
        print(f"  目标帧率: {fps} FPS")
        print("  人形风格: 逼真模式（肤色+衣服纹理）")
        print("-" * 50)
    
    def GetImageSample(self):
        """获取一帧图像"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        self.last_time = time.time()
        self.frame_count += 1
        
        # 生成逼真人体
        image = self._generate_realistic_human()
        
        return image
    
    def _generate_realistic_human(self):
        """生成逼真人体（纹理、阴影、肤色）"""
        
        # 1. 创建背景（渐变+草地纹理）
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            # 天空（上部分）和地面（下部分）
            if y < self.height * 0.6:
                # 天空渐变
                blue = int(100 + 100 * (1 - y / (self.height * 0.6)))
                image[y, :] = [135 + blue//3, 206 + blue//5, 235 + blue//10]
            else:
                # 草地绿色
                green = int(100 + 50 * np.sin(y * 0.05))
                image[y, :] = [50, green + 80, 50]
        
        # 添加草地纹理（随机噪点）
        grass_noise = np.random.randint(0, 30, (self.height, self.width, 3), dtype=np.uint8)
        image = cv2.addWeighted(image, 0.9, grass_noise, 0.1, 0)
        
        # 2. 更新人形位置
        self.human_x += self.move_speed * self.move_direction
        if self.human_x > self.width - 150 or self.human_x < 150:
            self.move_direction *= -1
        
        x, y = self.human_x, self.human_y
        
        # 3. 绘制阴影（让人物更立体）
        shadow_offset = 8
        shadow_color = (50, 50, 50)
        
        # 4. 绘制身体各部位（带阴影）
        
        # 上衣（深蓝色，带高光）
        shirt_x1 = x - 50
        shirt_y1 = y - 100
        shirt_x2 = x + 50
        shirt_y2 = y - 20
        
        # 阴影
        cv2.rectangle(image, 
                     (shirt_x1 + shadow_offset, shirt_y1 + shadow_offset),
                     (shirt_x2 + shadow_offset, shirt_y2 + shadow_offset),
                     shadow_color, -1)
        # 上衣
        cv2.rectangle(image, (shirt_x1, shirt_y1), (shirt_x2, shirt_y2), (40, 70, 140), -1)
        
        # 添加衣服褶皱（水平条纹）
        for i in range(3):
            stripe_y = shirt_y1 + 20 + i * 20
            cv2.line(image, (shirt_x1 + 10, stripe_y), (shirt_x2 - 10, stripe_y), (100, 130, 200), 2)
        
        # 裤子（深灰色）
        pants_x1 = x - 45
        pants_y1 = y - 20
        pants_x2 = x + 45
        pants_y2 = y + 100
        
        cv2.rectangle(image, 
                     (pants_x1 + shadow_offset, pants_y1 + shadow_offset),
                     (pants_x2 + shadow_offset, pants_y2 + shadow_offset),
                     shadow_color, -1)
        cv2.rectangle(image, (pants_x1, pants_y1), (pants_x2, pants_y2), (30, 30, 80), -1)
        
        # 裤腿分界线
        cv2.line(image, (pants_x1, y + 40), (pants_x2, y + 40), (50, 50, 100), 2)
        
        # 5. 头部（肤色，带五官）
        # 阴影
        cv2.circle(image, (x + shadow_offset, y - 170 + shadow_offset), 38, shadow_color, -1)
        # 脸部
        cv2.circle(image, (x, y - 170), 38, (200, 160, 120), -1)
        
        # 头发（深棕色）
        cv2.ellipse(image, (x, y - 200), (35, 25), 0, 0, 360, (80, 60, 40), -1)
        
        # 眼睛
        cv2.circle(image, (x - 15, y - 180), 4, (0, 0, 0), -1)
        cv2.circle(image, (x + 15, y - 180), 4, (0, 0, 0), -1)
        # 眼白
        cv2.circle(image, (x - 15, y - 180), 2, (255, 255, 255), -1)
        cv2.circle(image, (x + 15, y - 180), 2, (255, 255, 255), -1)
        
        # 鼻子
        cv2.circle(image, (x, y - 170), 3, (150, 100, 80), -1)
        
        # 嘴巴（微笑）
        cv2.ellipse(image, (x, y - 160), (12, 6), 0, 0, 180, (80, 50, 40), 2)
        
        # 6. 手臂（肤色）
        # 左臂
        arm_left_points = np.array([
            [x - 50, y - 70],
            [x - 80, y - 40],
            [x - 75, y - 30],
            [x - 45, y - 60]
        ], np.int32)
        cv2.fillPoly(image, [arm_left_points], (200, 160, 120))
        
        # 右臂
        arm_right_points = np.array([
            [x + 50, y - 70],
            [x + 80, y - 40],
            [x + 75, y - 30],
            [x + 45, y - 60]
        ], np.int32)
        cv2.fillPoly(image, [arm_right_points], (200, 160, 120))
        
        # 7. 腿部（裤子颜色）
        # 左腿
        cv2.rectangle(image, (x - 45, y + 100), (x - 15, y + 170), (30, 30, 80), -1)
        # 右腿
        cv2.rectangle(image, (x + 15, y + 100), (x + 45, y + 170), (30, 30, 80), -1)
        
        # 8. 鞋子
        cv2.ellipse(image, (x - 30, y + 170), (15, 8), 0, 0, 360, (60, 50, 40), -1)
        cv2.ellipse(image, (x + 30, y + 170), (15, 8), 0, 0, 360, (60, 50, 40), -1)
        
        # 9. 添加光影效果（让人物更立体）
        # 头部高光
        cv2.circle(image, (x - 10, y - 185), 3, (240, 200, 160), -1)
        
        # 10. 添加运动模糊（模拟移动）
        if abs(self.move_speed) > 0:
            motion_blur = np.random.randint(0, 10, (self.height, self.width, 3), dtype=np.uint8)
            image = cv2.addWeighted(image, 0.95, motion_blur, 0.05, 0)
        
        # 11. 添加相机噪点（模拟真实照片）
        noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
        image = cv2.addWeighted(image, 0.98, noise, 0.02, 0)
        
        # 12. 轻微高斯模糊（模拟对焦）
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return image
    
    def test(self, duration=10):
        """测试函数"""
        import cv2
        
        print(f"\n开始测试 {duration} 秒...")
        print("按 'q' 键退出")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            image = self.GetImageSample()
            frame_count += 1
            
            # 显示图像
            cv2.imshow('Realistic Human Test', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        fps = frame_count / duration
        print(f"\n测试完成")
        print(f"  总帧数: {frame_count}")
        print(f"  实际帧率: {fps:.1f} FPS")
        return fps
    
    def save_frame(self, filename="realistic_human.jpg"):
        """保存当前帧"""
        image = self.GetImageSample()
        cv2.imwrite(filename, image)
        print(f"图像已保存: {filename}")
        return filename


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("逼真人体模拟器测试")
    print("=" * 60)
    
    # 创建客户端
    camera = SimVideoClient()
    
    # 保存一帧
    camera.save_frame("realistic_human.jpg")
    
    # 运行测试
    camera.test(duration=10)