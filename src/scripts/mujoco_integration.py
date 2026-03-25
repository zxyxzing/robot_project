#!/usr/bin/env python3
"""
MuJoCo集成模块
提供MuJoCo仿真平台的集成功能
"""

import os
import time
import numpy as np
from typing import Optional
from config import MUJOCO_AVAILABLE, CAMERA_FPS, CAMERA_FOV, logger

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None
    mujoco_viewer = None


class MuJoCoIntegration:
    """MuJoCo集成类"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化MuJoCo集成

        Args:
            model_path: MuJoCo模型文件路径，如果为None则使用内置模型
        """
        self.model = None
        self.data = None
        self.model_path = model_path
        self.is_initialized = False
        
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

    def initialize(self) -> bool:
        """
        初始化MuJoCo环境

        Returns:
            bool: 初始化是否成功
        """
        if not MUJOCO_AVAILABLE:
            logger.error("MuJoCo未安装")
            return False

        try:
            # 加载或创建MuJoCo模型
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"从文件加载MuJoCo模型: {self.model_path}")
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
            else:
                logger.info("创建内置MuJoCo模型")
                self.model = self._create_builtin_model()

            if self.model is None:
                logger.error("MuJoCo模型创建失败")
                return False

            # 创建数据对象
            self.data = mujoco.MjData(self.model)

            self.is_initialized = True
            logger.info("MuJoCo环境初始化完成")
            return True

        except Exception as e:
            logger.error(f"MuJoCo初始化失败: {str(e)}")
            return False

    def _create_builtin_model(self):
        """
        加载Go2机器人的MuJoCo模型

        Returns:
            MuJoCo模型对象
        """
        try:
            # Go2和人体模型路径
            go2_human_model_path = "/home/yuan/dog/venv_yolo_follow/src/model/go2_human_follow_new.xml"

            # 从XML文件加载模型
            model = mujoco.MjModel.from_xml_path(go2_human_model_path)

            logger.info(f"Go2和人体MuJoCo模型加载成功: {go2_human_model_path}")
            return model

        except Exception as e:
            logger.error(f"加载Go2 MuJoCo模型失败: {str(e)}")
            return None

    def step(self) -> None:
        """执行一步MuJoCo仿真"""
        if self.is_initialized and self.data is not None:
            mujoco.mj_step(self.model, self.data)

    def update_body_position(self, body_name: str, position) -> None:
        """
        更新MuJoCo中物体的位置

        Args:
            body_name: 物体名称
            position: 新的位置
        """
        if self.is_initialized and self.model is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self.data.xpos[body_id] = position

    def get_body_position(self, body_name: str):
        """
        获取MuJoCo中物体的位置

        Args:
            body_name: 物体名称

        Returns:
            物体位置
        """
        if self.is_initialized and self.model is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return self.data.xpos[body_id].copy()
        return None

    def update_human_position(self) -> None:
        """
        更新人体位置和朝向
        人体以 0.2m/s 的速度移动，每隔 2-5 秒随机转向 ±10°
        运动范围限制在 ±5m 内，同时确保不穿模
        """
        if not self.is_initialized or self.model is None:
            return

        # 获取人体当前位置
        human_pos = self.get_body_position("human")
        if human_pos is None:
            return

        current_time = time.time()
        dt = 1.0 / CAMERA_FPS  # 时间步长

        # 检查是否需要转向
        if current_time - self.human_last_turn_time >= self.human_current_turn_interval:
            # 随机转向 ±10°
            turn_angle = np.random.uniform(-self.human_turn_angle, self.human_turn_angle)
            self.human_orientation += turn_angle
            
            # 更新转向时间和间隔
            self.human_last_turn_time = current_time
            self.human_current_turn_interval = np.random.uniform(*self.human_turn_interval)
            
            logger.debug(
                f"人体转向: {turn_angle:.1f}°, "
                f"当前朝向: {self.human_orientation:.1f}°"
            )

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

            logger.debug(
                f"人体超出运动范围，调整朝向: {new_yaw:.1f}°, "
                f"当前位置: {human_pos[:2]}"
            )

        # 检查是否与机器人碰撞（穿模）
        robot_pos = self.get_body_position("robot")
        if robot_pos is not None:
            # 计算人体和机器人之间的距离
            dist_to_robot = np.linalg.norm(new_pos - robot_pos)
            
            # 人体尺寸为 0.6m×0.4m×1.7m，机器人尺寸为 0.5m×0.5m×0.6m
            # 设置最小安全距离
            min_distance = 0.5  # 最小安全距离
            
            if dist_to_robot < min_distance:
                # 发生碰撞，调整位置避免穿模
                to_robot = robot_pos - human_pos
                to_robot[2] = 0  # 保持z坐标不变
                to_robot = to_robot / np.linalg.norm(to_robot)

                # 将人体移离机器人
                new_pos = robot_pos - to_robot * min_distance
                new_pos[2] = 0.85  # 保持人体高度

                # 调整朝向，远离机器人
                new_yaw = np.degrees(np.arctan2(to_robot[0], to_robot[1]))
                self.human_orientation = new_yaw


                logger.warning(
                    f"检测到碰撞，调整人体位置避免穿模: {new_pos[:2]}"
                )

        # 更新人体位置
        self.update_body_position("human", new_pos)

    def control_go2_to_follow_human(self) -> None:
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

    def launch_viewer(self, update_callback, fps: int):
        """
        启动MuJoCo查看器

        Args:
            update_callback: 更新回调函数
            fps: 帧率
        """
        if not self.is_initialized or self.model is None:
            logger.warning("MuJoCo未初始化，无法启动查看器")
            return

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                logger.info("MuJoCo查看器已启动")

                # 设置初始视角
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.distance = 5
                viewer.cam.lookat[:] = [0, 1.5, 0.5]

                # 主循环
                while viewer.is_running():
                    # 更新人体位置
                    self.update_human_position()

                    # 控制Go2跟随人体
                    self.control_go2_to_follow_human()

                    # 执行MuJoCo仿真步
                    self.step()

                    # 同步到查看器
                    viewer.sync()

                    # 控制帧率
                    time.sleep(1.0 / fps)

                logger.info("MuJoCo查看器已关闭")

        except Exception as e:
            logger.error(f"MuJoCo查看器运行失败: {str(e)}")
