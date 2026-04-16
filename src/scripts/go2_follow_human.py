#!/usr/bin/env python3
"""
Go2机器狗跟随人体行走程序
在MuJoCo仿真平台中实现Go2机器狗跟随人体目标
"""

import os
import cv2
import time
import numpy as np
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('go2_follow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入MuJoCo
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
    logger.info("MuJoCo已安装")
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.error("MuJoCo未安装，程序无法运行")


class Go2FollowHuman:
    """Go2机器狗跟随人体控制类"""

    def __init__(self, model_path: str):
        """
        初始化Go2跟随人体控制

        Args:
            model_path: Go2机器人的MuJoCo模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.data = None
        self.viewer = None
        self.is_running = False

        # 人体运动控制参数
        self.human_speed = 0.2  # 人体移动速度 (m/s)
        self.human_turn_angle = 10.0  # 最大转向角度 (度)
        self.human_turn_interval = (2.0, 5.0)  # 转向间隔范围 (秒)
        self.human_movement_range = 5.0  # 运动范围 (米)
        self.human_last_turn_time = 0.0
        self.human_current_turn_interval = 0.0
        self.human_orientation = 180.0  # 初始朝向 (度)，面向机器人

        # Go2跟随控制参数
        self.follow_distance = 1.5  # 跟随距离 (米)
        self.max_speed = 0.5  # 最大速度 (m/s)
        self.turn_speed = 1.0  # 转向速度 (rad/s)

        # 摄像头参数
        self.camera_resolution = (1920, 1080)
        self.camera_fps = 30

        # 状态变量
        self.frame_count = 0
        self.last_time = time.time()
        self.fps_history = []

    def initialize(self) -> bool:
        """
        初始化MuJoCo环境和模型

        Returns:
            bool: 初始化是否成功
        """
        if not MUJOCO_AVAILABLE:
            logger.error("MuJoCo未安装，无法初始化")
            return False

        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False



            # 加载MuJoCo模型
            logger.info(f"加载Go2模型: {self.model_path}")
            self.model = mujoco.MjModel.from_xml_path(self.model_path)

            
            self.data = mujoco.MjData(self.model)

            logger.info("MuJoCo环境初始化成功")
            return True

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            return False



        

        


        human_worldbody_end = human_xml.find("</worldbody>")
        if human_worldbody_start == -1 or human_worldbody_end == -1:
            logger.error("人体模型XML中未找到worldbody标签")
            raise ValueError("人体模型XML中未找到worldbody标签")
        
        human_worldbody_full = human_xml[human_worldbody_start + len("<worldbody>"):human_worldbody_end]
        
        # 移除人体模型中的floor和light
        human_worldbody = human_worldbody_full
        
        # 移除floor标签
        floor_start = human_worldbody.find('<geom name="floor"')
        if floor_start != -1:
            floor_end = human_worldbody.find('/>', floor_start) + 2
            human_worldbody = human_worldbody[:floor_start] + human_worldbody[floor_end:]
        
        # 移除spotlight标签
        light_start = human_worldbody.find('<light name="spotlight"')
        if light_start != -1:
            light_end = human_worldbody.find('/>', light_start) + 2
            human_worldbody = human_worldbody[:light_start] + human_worldbody[light_end:]
        
        # 移除top light标签
        light_start = human_worldbody.find('<light name="top"')
        if light_start != -1:
            light_end = human_worldbody.find('/>', light_start) + 2
            human_worldbody = human_worldbody[:light_start] + human_worldbody[light_end:]
        
        # 提取狗模型的worldbody内容
        dog_worldbody_start = dog_xml.find("<worldbody>")
        dog_worldbody_end = dog_xml.find("</worldbody>")
        if dog_worldbody_start == -1 or dog_worldbody_end == -1:
            logger.error("狗模型XML中未找到worldbody标签")
            raise ValueError("狗模型XML中未找到worldbody标签")
        
        dog_worldbody = dog_xml[dog_worldbody_start + len("<worldbody>"):dog_worldbody_end]
        
        # 提取狗模型的actuator内容
        dog_actuator_start = dog_xml.find("<actuator>")
        dog_actuator_end = dog_xml.find("</actuator>")
        dog_actuator = ""
        if dog_actuator_start != -1 and dog_actuator_end != -1:
            dog_actuator = dog_xml[dog_actuator_start:dog_actuator_end + len("</actuator>")]
        
        # 提取狗模型的default内容
        dog_default_start = dog_xml.find("<default>")
        dog_default_end = dog_xml.find("</default>")
        dog_default = ""
        if dog_default_start != -1 and dog_default_end != -1:
            dog_default = dog_xml[dog_default_start:dog_default_end + len("</default>")]
        
        # 提取狗模型的compiler内容
        dog_compiler_start = dog_xml.find("<compiler")
        dog_compiler_end = dog_xml.find("/>")
        dog_compiler = ""
        if dog_compiler_start != -1 and dog_compiler_end != -1:
            dog_compiler = dog_xml[dog_compiler_start:dog_compiler_end + 2]
        
        # 提取狗模型的asset内容
        dog_asset_start = dog_xml.find("<asset>")
        dog_asset_end = dog_xml.find("</asset>")
        dog_asset = ""
        if dog_asset_start != -1 and dog_asset_end != -1:
            dog_asset = dog_xml[dog_asset_start:dog_asset_end + len("</asset>")]
        
        # 提取狗模型的keyframe内容
        dog_keyframe_start = dog_xml.find("<keyframe>")
        dog_keyframe_end = dog_xml.find("</keyframe>")
        dog_keyframe = ""
        if dog_keyframe_start != -1 and dog_keyframe_end != -1:
            dog_keyframe = dog_xml[dog_keyframe_start:dog_keyframe_end + len("</keyframe>")]
        
        # 提取人体模型的asset内容
        human_asset_start = human_xml.find("<asset>")
        human_asset_end = human_xml.find("</asset>")
        human_asset = ""
        if human_asset_start != -1 and human_asset_end != -1:
            human_asset = human_xml[human_asset_start:human_asset_end + len("</asset>")]
        
        # 提取人体模型的default内容
        human_default_start = human_xml.find("<default>")
        human_default_end = human_xml.find("</default>")
        human_default = ""
        if human_default_start != -1 and human_default_end != -1:
            human_default = human_xml[human_default_start:human_default_end + len("</default>")]
        
        # 修改人体模型中的类名，使其与狗模型的类名不冲突
        # 将"body"改为"human_body"，"thigh"改为"human_thigh"等
        human_worldbody = human_worldbody.replace('class="body"', 'class="human_body"')
        human_worldbody = human_worldbody.replace('class="thigh"', 'class="human_thigh"')
        human_worldbody = human_worldbody.replace('class="shin"', 'class="human_shin"')
        human_worldbody = human_worldbody.replace('class="foot"', 'class="human_foot"')
        human_worldbody = human_worldbody.replace('class="foot1"', 'class="human_foot1"')
        human_worldbody = human_worldbody.replace('class="foot2"', 'class="human_foot2"')
        human_worldbody = human_worldbody.replace('class="arm_upper"', 'class="human_arm_upper"')
        human_worldbody = human_worldbody.replace('class="arm_lower"', 'class="human_arm_lower"')
        human_worldbody = human_worldbody.replace('class="hand"', 'class="human_hand"')
        human_worldbody = human_worldbody.replace('class="joint_big"', 'class="human_joint_big"')
        human_worldbody = human_worldbody.replace('class="hip_x"', 'class="human_hip_x"')
        human_worldbody = human_worldbody.replace('class="hip_z"', 'class="human_hip_z"')
        human_worldbody = human_worldbody.replace('class="hip_y"', 'class="human_hip_y"')
        human_worldbody = human_worldbody.replace('class="joint_big_stiff"', 'class="human_joint_big_stiff"')
        human_worldbody = human_worldbody.replace('class="knee"', 'class="human_knee"')
        human_worldbody = human_worldbody.replace('class="ankle"', 'class="human_ankle"')
        human_worldbody = human_worldbody.replace('class="ankle_y"', 'class="human_ankle_y"')
        human_worldbody = human_worldbody.replace('class="ankle_x"', 'class="human_ankle_x"')
        human_worldbody = human_worldbody.replace('class="shoulder"', 'class="human_shoulder"')
        human_worldbody = human_worldbody.replace('class="elbow"', 'class="human_elbow"')
        human_worldbody = human_worldbody.replace('childclass="body"', 'childclass="human_body"')
        
        # 修改人体模型default中的类名
        human_default = human_default.replace('class="body"', 'class="human_body"')
        human_default = human_default.replace('class="thigh"', 'class="human_thigh"')
        human_default = human_default.replace('class="shin"', 'class="human_shin"')
        human_default = human_default.replace('class="foot"', 'class="human_foot"')
        human_default = human_default.replace('class="foot1"', 'class="human_foot1"')
        human_default = human_default.replace('class="foot2"', 'class="human_foot2"')
        human_default = human_default.replace('class="arm_upper"', 'class="human_arm_upper"')
        human_default = human_default.replace('class="arm_lower"', 'class="human_arm_lower"')
        human_default = human_default.replace('class="hand"', 'class="human_hand"')
        human_default = human_default.replace('class="joint_big"', 'class="human_joint_big"')
        human_default = human_default.replace('class="hip_x"', 'class="human_hip_x"')
        human_default = human_default.replace('class="hip_z"', 'class="human_hip_z"')
        human_default = human_default.replace('class="hip_y"', 'class="human_hip_y"')
        human_default = human_default.replace('class="joint_big_stiff"', 'class="human_joint_big_stiff"')
        human_default = human_default.replace('class="knee"', 'class="human_knee"')
        human_default = human_default.replace('class="ankle"', 'class="human_ankle"')
        human_default = human_default.replace('class="ankle_y"', 'class="human_ankle_y"')
        human_default = human_default.replace('class="ankle_x"', 'class="human_ankle_x"')
        human_default = human_default.replace('class="shoulder"', 'class="human_shoulder"')
        human_default = human_default.replace('class="elbow"', 'class="human_elbow"')
        
        # 创建合并后的XML字符串
        merged_xml = f'''<mujoco>
            {dog_compiler}
            {dog_default}
            {human_default}
            {dog_asset}
            {human_asset}
            <worldbody>
                <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
                <light name="light" pos="0 0 5" dir="0 0 -1"/>
                {human_worldbody}
                {dog_worldbody}
            </worldbody>
            {dog_actuator}
            {dog_keyframe}
        </mujoco>
        '''
        
        # 从XML字符串创建模型
        return mujoco.MjModel.from_xml_string(merged_xml)

    def add_human_target(self) -> None:
        """
        添加人体模型到仿真环境

        人体尺寸: 0.6m×0.4m×1.7m
        初始位置: 3m正前方
        """
        # 获取人体body ID
        human_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human")
        if human_id < 0:
            logger.warning("未找到人体body，跳过添加人体")
            return

        # 设置人体初始位置为 (0, 3, 0.85)，即3m正前方
        self.data.xpos[human_id] = np.array([0.0, 3.0, 0.85])

        logger.info(f"人体模型添加完成，位置: {self.data.xpos[human_id]}")

    def human_movement(self) -> None:
        """
        控制人体运动

        人体以匀速 0.2m/s 移动，每隔 2-5 秒随机转向 ±10°
        运动范围限制在 ±5m 内，无穿模
        """
        # 获取人体body ID
        human_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human")
        if human_id < 0:
            return

        # 获取当前人体位置
        human_pos = self.data.xpos[human_id].copy()

        current_time = time.time()
        dt = 1.0 / self.camera_fps  # 时间步长

        # 检查是否需要转向
        if current_time - self.human_last_turn_time >= self.human_current_turn_interval:
            # 随机转向 ±10°
            turn_angle = np.random.uniform(-self.human_turn_angle, self.human_turn_angle)
            self.human_orientation += turn_angle

            # 更新转向时间和间隔
            self.human_last_turn_time = current_time
            self.human_current_turn_interval = np.random.uniform(*self.human_turn_interval)

            logger.debug(f"人体转向: {turn_angle:.1f}°, 当前朝向: {self.human_orientation:.1f}°")

        # 计算移动方向（基于当前朝向）
        yaw = np.radians(self.human_orientation)
        direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])

        # 计算新位置（匀速 0.2m/s）
        new_pos = human_pos + direction * self.human_speed * dt

        # 检查是否超出运动范围（±5m）
        distance_from_origin = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        if distance_from_origin > self.human_movement_range:
            # 超出范围，朝向原点方向
            to_origin = -human_pos
            to_origin[2] = 0  # 保持z坐标不变
            to_origin = to_origin / np.linalg.norm(to_origin)

            # 计算新的朝向
            new_yaw = np.degrees(np.arctan2(to_origin[0], to_origin[1]))
            self.human_orientation = new_yaw

            # 重新计算新位置
            yaw = np.radians(new_yaw)
            direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])
            new_pos = human_pos + direction * self.human_speed * dt

            logger.debug(f"人体超出运动范围，调整朝向: {new_yaw:.1f}°, 当前位置: {human_pos[:2]}")

        # 更新人体位置
        self.data.xpos[human_id] = new_pos

    def _init_human_position(self) -> None:
        """初始化人体位置"""
        # 调用add_human_target()函数
        self.add_human_target()

        logger.info(f"人体位置初始化完成")

    def _init_go2_position(self) -> None:
        """初始化Go2位置"""
        # 获取base body ID
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id < 0:
            logger.warning("未找到base body，跳过初始化")
            return

        # 设置base初始位置为 (0, 0, 0.35)
        self.data.xpos[base_id] = np.array([0.0, 0.0, 0.35])

        logger.info(f"Go2位置初始化完成: {self.data.xpos[base_id]}")

    def update_human_position(self) -> None:
        """
        更新人体位置和朝向
        人体以 0.2m/s 的速度移动，每隔 2-5 秒随机转向 ±10°
        运动范围限制在 ±5m 内
        """
        # 获取人体body ID
        human_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human")
        if human_id < 0:
            return

        # 获取当前人体位置
        human_pos = self.data.xpos[human_id].copy()

        current_time = time.time()
        dt = 1.0 / self.camera_fps  # 时间步长

        # 检查是否需要转向
        if current_time - self.human_last_turn_time >= self.human_current_turn_interval:
            # 随机转向 ±10°
            turn_angle = np.random.uniform(-self.human_turn_angle, self.human_turn_angle)
            self.human_orientation += turn_angle

            # 更新转向时间和间隔
            self.human_last_turn_time = current_time
            self.human_current_turn_interval = np.random.uniform(*self.human_turn_interval)

            logger.debug(f"人体转向: {turn_angle:.1f}°, 当前朝向: {self.human_orientation:.1f}°")

        # 计算移动方向（基于当前朝向）
        yaw = np.radians(self.human_orientation)
        direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])

        # 计算新位置（匀速 0.2m/s）
        new_pos = human_pos + direction * self.human_speed * dt

        # 检查是否超出运动范围（±5m）
        distance_from_origin = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        if distance_from_origin > self.human_movement_range:
            # 超出范围，朝向原点方向
            to_origin = -human_pos
            to_origin[2] = 0  # 保持z坐标不变
            to_origin = to_origin / np.linalg.norm(to_origin)

            # 计算新的朝向
            new_yaw = np.degrees(np.arctan2(to_origin[0], to_origin[1]))
            self.human_orientation = new_yaw

            # 重新计算新位置
            yaw = np.radians(new_yaw)
            direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])
            new_pos = human_pos + direction * self.human_speed * dt

            logger.debug(f"人体超出运动范围，调整朝向: {new_yaw:.1f}°, 当前位置: {human_pos[:2]}")

        # 更新人体位置
        self.data.xpos[human_id] = new_pos

    def control_go2_to_follow(self) -> None:
        """
        控制Go2跟随人体
        使用简单的PID控制器实现跟随行为
        """
        # 获取base和human body ID
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        human_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human")

        if base_id < 0 or human_id < 0:
            return

        # 获取当前Go2和人体位置
        go2_pos = self.data.xpos[base_id].copy()
        human_pos = self.data.xpos[human_id].copy()

        # 计算目标位置（人体前方follow_distance米处）
        # 获取人体朝向
        yaw = np.radians(self.human_orientation)
        human_dir = np.array([np.sin(yaw), np.cos(yaw), 0.0])

        # 目标位置为人体前方follow_distance米处
        target_pos = human_pos + human_dir * self.follow_distance

        # 计算从Go2到目标位置的向量
        to_target = target_pos - go2_pos
        to_target[2] = 0  # 只在水平面上移动

        # 计算距离和方向
        distance = np.linalg.norm(to_target)
        if distance > 0.01:  # 避免除零
            # 归一化方向
            direction = to_target / distance

            # 计算目标朝向
            target_yaw = np.arctan2(direction[0], direction[1])

            # 获取当前Go2朝向（简化为y轴旋转）
            current_yaw = 0.0  # 简化，假设Go2初始朝向为0

            # 计算转向误差
            yaw_error = target_yaw - current_yaw

            # 归一化转向误差到[-π, π]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi

            # 设置关节控制信号
            # 这里简化处理，只控制髋关节旋转实现转向
            # FL_hip_joint (关节0), FR_hip_joint (关节3), RL_hip_joint (关节6), RR_hip_joint (关节9)
            hip_joints = [0, 3, 6, 9]

            # 根据转向误差设置髋关节
            for joint_id in hip_joints:
                # 设置转向控制
                self.data.ctrl[joint_id] = np.clip(yaw_error * 2.0, -1.0, 1.0)

            # 设置前进速度控制
            # FL_thigh_joint (关节1), FR_thigh_joint (关节4), RL_thigh_joint (关节7), RR_thigh_joint (关节10)
            thigh_joints = [1, 4, 7, 10]

            # 根据距离设置前进速度
            speed = np.clip(distance * 2.0, 0.0, 1.0)

            for joint_id in thigh_joints:
                # 设置前进控制
                self.data.ctrl[joint_id] = speed

            # 设置膝关节保持稳定
            # FL_calf_joint (关节2), FR_calf_joint (关节5), RL_calf_joint (关节8), RR_calf_joint (关节11)
            calf_joints = [2, 5, 8, 11]

            for joint_id in calf_joints:
                # 设置膝关节控制以保持稳定
                self.data.ctrl[joint_id] = 0.3

    def get_camera_image(self) -> np.ndarray:
        """
        获取摄像头图像

        Returns:
            np.ndarray: 摄像头图像
        """
        # 创建空白图像
        img = np.zeros((self.camera_resolution[1], self.camera_resolution[0], 3), dtype=np.uint8)

        # 在实际应用中，这里应该从真实摄像头获取图像
        # 这里模拟生成一个简单的场景

        # 获取Go2和人体位置
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        human_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human")

        if base_id >= 0 and human_id >= 0:
            go2_pos = self.data.xpos[base_id]
            human_pos = self.data.xpos[human_id]

            # 绘制地面
            img[:, :] = [200, 200, 200]  # 浅灰色背景

            # 绘制人体（红色圆形）
            human_x = int((human_pos[0] + 5) * self.camera_resolution[0] / 10)
            human_y = int((human_pos[1] + 5) * self.camera_resolution[1] / 10)
            cv2.circle(img, (human_x, human_y), 20, (0, 0, 255), -1)

            # 绘制Go2（蓝色矩形）
            go2_x = int((go2_pos[0] + 5) * self.camera_resolution[0] / 10)
            go2_y = int((go2_pos[1] + 5) * self.camera_resolution[1] / 10)
            cv2.rectangle(img, (go2_x-15, go2_y-15), (go2_x+15, go2_y+15), (255, 0, 0), -1)

            # 添加信息文本
            cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(img, f"FPS: {self.fps_history[-1]:.1f}" if self.fps_history else "FPS: 0.0", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return img

    def step(self) -> None:
        """执行一步仿真"""
        if not self.is_running:
            return

        # 更新人体位置
        self.update_human_position()

        # 控制Go2跟随人体
        self.control_go2_to_follow()

        # 执行MuJoCo仿真步
        mujoco.mj_step(self.model, self.data)

        # 计算帧率
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else 0.0
        self.last_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)

        self.frame_count += 1

    def run(self) -> None:
        """运行仿真循环"""
        if not self.initialize():
            logger.error("初始化失败，无法运行")
            return

        self.is_running = True
        logger.info("开始Go2跟随人体仿真")

        try:
            # 启动MuJoCo查看器
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                logger.info("MuJoCo查看器已启动")

                # 设置初始视角
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.distance = 5
                viewer.cam.lookat[:] = [0, 1.5, 0.5]

                # 主循环
                while viewer.is_running():
                    # 执行仿真步
                    self.step()

                    # 同步到查看器
                    viewer.sync()

                    # 控制帧率
                    time.sleep(1.0 / self.camera_fps)

                logger.info("MuJoCo查看器已关闭")

        except Exception as e:
            logger.error(f"仿真运行出错: {str(e)}")
        finally:
            self.is_running = False
            logger.info("仿真结束")


def main():
    """主函数"""
    # 人体模型路径
    human_model_path = "venv_yolo_follow/src/model/unitree_go2/humanoid.xml"
    # Go2狗模型路径
    dog_model_path = "/home/yuan/dog/venv_yolo_follow/src/model/unitree_go2/go2.xml"

    # 创建Go2跟随人体控制器
    controller = Go2FollowHuman(human_model_path, dog_model_path)
    # 运行仿真
    controller.run()


if __name__ == "__main__":
    main()
    main()
