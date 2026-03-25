#!/usr/bin/env python3
"""
仿真环境主模块
提供仿真环境的核心功能
支持可视化显示
"""

import time
import json
from datetime import datetime
import numpy as np
from typing import Optional, Dict, Any
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import cv2
from config import (
    CAMERA_RESOLUTION, CAMERA_FPS, TEST_DURATION_MINUTES,
    logger
)
from robot import Robot
from human_target import HumanTarget
from mujoco_integration import MuJoCoIntegration


class Go2SimEnv:
    """Go2仿真环境类"""

    def __init__(self,
                 camera_resolution: tuple = None,
                 camera_fps: int = None,
                 test_duration_minutes: int = None,
                 use_mujoco: bool = False,
                 mujoco_model_path: Optional[str] = None):
        """
        初始化Go2仿真环境

        Args:
            camera_resolution: 摄像头分辨率 (width, height)，默认为(1920, 1080)
            camera_fps: 摄像头帧率，默认为200Hz
            test_duration_minutes: 测试持续时间（分钟），默认为30分钟
            use_mujoco: 是否使用MuJoCo仿真平台，默认为False
            mujoco_model_path: MuJoCo模型文件路径，如果为None则使用内置模型
        """
        # 使用配置文件中的默认值，如果没有提供
        self.camera_resolution = camera_resolution or CAMERA_RESOLUTION
        self.camera_fps = camera_fps or CAMERA_FPS
        self.test_duration_seconds = (test_duration_minutes or TEST_DURATION_MINUTES) * 60

        # MuJoCo相关
        self.use_mujoco = use_mujoco
        self.mujoco_integration = None
        if use_mujoco:
            # 默认使用新的合并模型
            if mujoco_model_path is None:
                mujoco_model_path = "/home/yuan/dog/venv_yolo_follow/src/model/go2_human_follow_new.xml"
            self.mujoco_integration = MuJoCoIntegration(mujoco_model_path)

        # 初始化机器人
        self.robot = Robot()

        # 初始化人体目标
        self.human_target = HumanTarget()

        # 状态变量
        self.is_initialized = False
        self.is_running = False
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps_history = []

        # 碰撞计数
        self.collision_count = 0

        # 可视化相关
        self.visualization_enabled = False
        self.fig = None
        self.ax = None
        self.robot_patch = None
        self.human_patch = None
        self.traj_lines = []
        self.robot_trajectory = []
        self.human_trajectory = []
        self.max_trajectory_length = 100
        self.info_text = None
        self.cv_window_name = "Go2 Simulation Environment"

        logger.info(f"初始化Go2仿真环境，配置：分辨率={self.camera_resolution}, 帧率={self.camera_fps}Hz")

    def initialize(self) -> bool:
        """
        初始化仿真环境

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("正在初始化仿真环境...")

            # 如果使用MuJoCo，初始化MuJoCo环境
            if self.use_mujoco and self.mujoco_integration:
                if not self.mujoco_integration.initialize():
                    logger.error("MuJoCo初始化失败，切换到模拟模式")
                    self.use_mujoco = False
                else:
                    logger.info("MuJoCo环境初始化成功")

            # 初始化摄像头
            self._init_camera()

            # 初始化物理引擎
            self._init_physics_engine()

            self.is_initialized = True
            logger.info("仿真环境初始化成功")
            return True

        except Exception as e:
            logger.error(f"仿真环境初始化失败: {str(e)}")
            return False

    def _init_camera(self) -> None:
        """初始化摄像头"""
        logger.info(f"初始化摄像头：分辨率={self.camera_resolution}, 帧率={self.camera_fps}Hz")

        # 模拟摄像头初始化
        self.camera = {
            'resolution': self.camera_resolution,
            'fps': self.camera_fps,
            'frame_interval': 1.0 / self.camera_fps,
            'is_active': True
        }

        # 预分配帧缓冲区
        self.frame_buffer = np.zeros((*self.camera_resolution[::-1], 3), dtype=np.uint8)
        logger.info("摄像头初始化完成")

    def _init_physics_engine(self) -> None:
        """初始化物理引擎"""
        logger.info("初始化物理引擎")
        # 模拟物理引擎初始化
        logger.info("物理引擎初始化完成")

    def enable_visualization(self, use_matplotlib: bool = True) -> None:
        """
        启用可视化

        Args:
            use_matplotlib: 是否使用matplotlib可视化，否则使用OpenCV
        """
        self.visualization_enabled = True

        if use_matplotlib:
            self._init_matplotlib_visualization()
        else:
            self._init_opencv_visualization()

        logger.info(f"可视化已启用，使用: {'matplotlib' if use_matplotlib else 'OpenCV'}")

    def _init_matplotlib_visualization(self) -> None:
        """初始化matplotlib可视化界面"""
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Go2 Robot Simulation Environment')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')

        # 创建机器人矩形
        robot_width, robot_length = self.robot.dimensions[0], self.robot.dimensions[1]
        self.robot_patch = Rectangle(
            (self.robot.position[0] - robot_width/2, self.robot.position[1] - robot_length/2),
            robot_width, robot_length,
            angle=np.degrees(self.robot.orientation[2]),
            facecolor='blue', alpha=0.7, label='Robot'
        )
        self.ax.add_patch(self.robot_patch)

        # 创建人体目标圆形
        human_radius = self.human_target.dimensions[0] / 2
        self.human_patch = Circle(
            self.human_target.position[:2],
            human_radius,
            facecolor='red', alpha=0.7, label='Human Target'
        )
        self.ax.add_patch(self.human_patch)

        # 创建轨迹线
        self.robot_traj_line, = self.ax.plot([], [], 'b--', linewidth=1, alpha=0.5)
        self.human_traj_line, = self.ax.plot([], [], 'r--', linewidth=1, alpha=0.5)

        # 添加图例
        self.ax.legend(loc='upper right')

        # 添加信息文本
        self.info_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.draw()
        plt.pause(0.01)

    def _init_opencv_visualization(self) -> None:
        """初始化OpenCV可视化界面"""
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cv_window_name, 800, 800)

    def update_visualization(self) -> None:
        """更新可视化界面"""
        if not self.visualization_enabled:
            return

        if self.fig is not None:
            self._update_matplotlib_visualization()
        else:
            self._update_opencv_visualization()

    def _update_matplotlib_visualization(self) -> None:
        """更新matplotlib可视化界面"""
        # 更新机器人位置
        robot_width, robot_length = self.robot.dimensions[0], self.robot.dimensions[1]
        self.robot_patch.set_xy((
            self.robot.position[0] - robot_width/2,
            self.robot.position[1] - robot_length/2
        ))
        self.robot_patch.angle = np.degrees(self.robot.orientation[2])

        # 更新人体目标位置
        self.human_patch.center = self.human_target.position[:2]

        # 更新轨迹
        self.robot_trajectory.append(self.robot.position[:2].copy())
        self.human_trajectory.append(self.human_target.position[:2].copy())

        if len(self.robot_trajectory) > self.max_trajectory_length:
            self.robot_trajectory.pop(0)
        if len(self.human_trajectory) > self.max_trajectory_length:
            self.human_trajectory.pop(0)

        if self.robot_trajectory:
            robot_traj = np.array(self.robot_trajectory)
            self.robot_traj_line.set_data(robot_traj[:, 0], robot_traj[:, 1])
        if self.human_trajectory:
            human_traj = np.array(self.human_trajectory)
            self.human_traj_line.set_data(human_traj[:, 0], human_traj[:, 1])

        # 更新信息文本
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0
        info_str = (
            f"Frame: {self.frame_count}\n"
            f"FPS: {self.fps_history[-1]:.1f} (Avg: {avg_fps:.1f})\n"
            f"Robot Pos: ({self.robot.position[0]:.2f}, {self.robot.position[1]:.2f})\n"
            f"Robot Orientation: {np.degrees(self.robot.orientation[2]):.1f}°\n"
            f"Human Pos: ({self.human_target.position[0]:.2f}, {self.human_target.position[1]:.2f})\n"
            f"Collisions: {self.collision_count}"
        )
        self.info_text.set_text(info_str)

        # 重绘图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_opencv_visualization(self) -> None:
        """更新OpenCV可视化界面"""
        # 创建空白画布
        canvas_size = 800
        scale = 40  # 像素/米
        center = canvas_size // 2

        img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        # 绘制网格
        for i in range(-10, 11):
            pos = int(center + i * scale)
            cv2.line(img, (pos, 0), (pos, canvas_size), (200, 200, 200), 1)
            cv2.line(img, (0, pos), (canvas_size, pos), (200, 200, 200), 1)

        # 绘制机器人
        robot_center = (
            int(center + self.robot.position[0] * scale),
            int(center + self.robot.position[1] * scale)
        )
        robot_width = int(self.robot.dimensions[0] * scale)
        robot_length = int(self.robot.dimensions[1] * scale)
        robot_angle = self.robot.orientation[2]

        # 计算机器人的四个角点
        cos_a, sin_a = np.cos(robot_angle), np.sin(robot_angle)
        corners = [
            (-robot_width/2, -robot_length/2),
            (robot_width/2, -robot_length/2),
            (robot_width/2, robot_length/2),
            (-robot_width/2, robot_length/2)
        ]
        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rotated_corners.append((
                int(robot_center[0] + rx),
                int(robot_center[1] + ry)
            ))

        cv2.fillPoly(img, [np.array(rotated_corners)], (0, 0, 255))

        # 绘制人体目标
        human_center = (
            int(center + self.human_target.position[0] * scale),
            int(center + self.human_target.position[1] * scale)
        )
        human_radius = int(self.human_target.dimensions[0] / 2 * scale)
        cv2.circle(img, human_center, human_radius, (0, 0, 255), -1)

        # 绘制轨迹
        if len(self.robot_trajectory) > 1:
            traj_points = [
                (int(center + x * scale), int(center + y * scale))
                for x, y in self.robot_trajectory
            ]
            cv2.polylines(img, [np.array(traj_points)], False, (255, 0, 0), 2)

        if len(self.human_trajectory) > 1:
            traj_points = [
                (int(center + x * scale), int(center + y * scale))
                for x, y in self.human_trajectory
            ]
            cv2.polylines(img, [np.array(traj_points)], False, (0, 0, 255), 2)

        # 添加信息文本
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0
        info_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.fps_history[-1]:.1f} (Avg: {avg_fps:.1f})",
            f"Robot Pos: ({self.robot.position[0]:.2f}, {self.robot.position[1]:.2f})",
            f"Robot Orientation: {np.degrees(self.robot.orientation[2]):.1f}°",
            f"Human Pos: ({self.human_target.position[0]:.2f}, {self.human_target.position[1]:.2f})",
            f"Collisions: {self.collision_count}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(img, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow(self.cv_window_name, img)
        cv2.waitKey(1)

    def close_visualization(self) -> None:
        """关闭可视化界面"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        else:
            cv2.destroyWindow(self.cv_window_name)

        self.visualization_enabled = False
        logger.info("可视化界面已关闭")

    def start(self) -> bool:
        """
        启动仿真环境

        Returns:
            bool: 启动是否成功
        """
        if not self.is_initialized:
            logger.error("环境未初始化，无法启动")
            return False

        try:
            logger.info("正在启动仿真环境...")
            self.is_running = True
            self.camera['is_active'] = True
            self.last_frame_time = time.time()
            logger.info("仿真环境启动成功")
            return True

        except Exception as e:
            logger.error(f"仿真环境启动失败: {str(e)}")
            return False

    def stop(self) -> None:
        """停止仿真环境"""
        logger.info("正在停止仿真环境...")
        self.is_running = False
        self.camera['is_active'] = False
        
        # 关闭可视化
        if self.visualization_enabled:
            self.close_visualization()
        
        logger.info("仿真环境已停止")

    def step(self) -> Dict[str, Any]:
        """
        执行一步仿真

        Returns:
            Dict[str, Any]: 包含当前状态信息的字典
        """
        if not self.is_running:
            logger.warning("环境未运行，无法执行step")
            return {}

        # 如果使用MuJoCo，更新MuJoCo仿真
        if self.use_mujoco and self.mujoco_integration and self.mujoco_integration.is_initialized:
            # 更新人体位置到MuJoCo
            if self.human_target.is_active:
                self.mujoco_integration.update_body_position("human", self.human_target.position)

            # 执行MuJoCo仿真步
            self.mujoco_integration.step()

            # 从MuJoCo获取机器人位置
            robot_pos = self.mujoco_integration.get_body_position("robot")
            if robot_pos is not None:
                self.robot.set_position(robot_pos)

        # 获取摄像头帧
        frame = self._get_camera_frame()

        # 更新物理状态
        self._update_physics()

        # 更新机器人状态
        self.robot.update()

        # 更新人体目标位置
        if self.human_target.is_active:
            # 使用新的 human_movement() 方法更新人体位置
            self.human_movement()
            # 检查碰撞
            if self.human_target.check_collision(self.robot.position, self.robot.dimensions):
                self.collision_count += 1
                if self.collision_count >= 10:
                    logger.error(
                        f"频繁检测到碰撞（{self.collision_count}次），"
                        f"可能存在穿模问题！"
                    )

        # 计算实际帧率
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if current_time > self.last_frame_time else 0.0
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)

        self.frame_count += 1

        # 更新可视化
        if self.visualization_enabled:
            self.update_visualization()

        # 返回状态信息
        return {
            'frame': frame,
            'frame_count': self.frame_count,
            'fps': fps,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0.0,
            'robot_position': self.robot.position,
            'robot_orientation': self.robot.orientation,
            'joint_positions': self.robot.joint_positions,
            'joint_velocities': self.robot.joint_velocities
        }

    def _get_camera_frame(self) -> np.ndarray:
        """
        获取摄像头帧

        Returns:
            np.ndarray: 摄像头帧数据
        """
        if not self.camera['is_active']:
            return np.zeros((*self.camera_resolution[::-1], 3), dtype=np.uint8)

        # 模拟生成摄像头帧
        # 在实际应用中，这里应该从真实的摄像头获取图像
        frame = self.frame_buffer.copy()

        # 添加一些随机噪声模拟真实摄像头
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    def _update_physics(self) -> None:
        """更新物理状态"""
        # 模拟物理更新
        pass

    def add_human_target(self) -> bool:
        """
        添加人体目标到仿真环境
        人体尺寸：0.6m×0.4m×1.7m
        初始位置：3m正前方

        Returns:
            bool: 添加是否成功
        """
        try:
            logger.info("正在添加人体目标...")
            
            # 设置人体尺寸为 0.6m×0.4m×1.7m
            self.human_target.dimensions = np.array([0.6, 0.4, 1.7])
            
            # 设置初始位置为 3m 正前方（Y轴正方向）
            self.human_target.position = np.array([0.0, 3.0, 0.85])  # z=0.85 为人体高度的一半
            
            # 设置初始朝向为 180°（面向机器人）
            self.human_target.orientation = np.array([0.0, 180.0, 0.0])
            
            # 激活人体目标
            self.human_target.activate()

            logger.info(
                f"人体目标添加成功：尺寸={self.human_target.dimensions}m, "
                f"初始位置={self.human_target.position}m, 初始朝向={self.human_target.orientation}°"
            )
            return True

        except Exception as e:
            logger.error(f"添加人体目标失败: {str(e)}")
            return False

    def human_movement(self) -> None:
        """
        控制人体目标移动
        人体以 0.2m/s 的速度移动，每隔 2-5 秒随机转向 ±10°
        运动范围限制在 ±5m 内，同时确保不穿模
        """
        if not self.human_target.is_active:
            return

        current_time = time.time()
        dt = 1.0 / self.camera_fps  # 时间步长

        # 检查是否需要转向
        if current_time - self.human_target.last_turn_time >= self.human_target.turn_interval:
            # 随机转向 ±10°
            turn_angle = np.random.uniform(-10.0, 10.0)
            self.human_target.orientation[2] += turn_angle
            
            # 更新转向时间和间隔
            self.human_target.last_turn_time = current_time
            self.human_target.turn_interval = np.random.uniform(2.0, 5.0)
            
            logger.debug(
                f"人体转向: {turn_angle:.1f}°, "
                f"当前朝向: {self.human_target.orientation[2]:.1f}°"
            )

        # 计算移动方向（基于当前朝向）
        yaw = np.radians(self.human_target.orientation[2])
        direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])

        # 计算新位置（匀速 0.2m/s）
        new_position = self.human_target.position + direction * 0.2 * dt

        # 检查是否超出运动范围（±5m）
        distance_from_origin = np.sqrt(new_position[0]**2 + new_position[1]**2)
        if distance_from_origin > 5.0:
            # 超出范围，朝向原点方向
            to_origin = -self.human_target.position
            to_origin[2] = 0  # 保持z坐标不变
            to_origin = to_origin / np.linalg.norm(to_origin)

            # 计算新的朝向
            new_yaw = np.degrees(np.arctan2(to_origin[0], to_origin[1]))
            self.human_target.orientation[2] = new_yaw

            # 重新计算新位置
            yaw = np.radians(new_yaw)
            direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])
            new_position = self.human_target.position + direction * 0.2 * dt

            logger.debug(
                f"人体超出运动范围，调整朝向: {new_yaw:.1f}°, "
                f"当前位置: {self.human_target.position[:2]}"
            )

        # 检查是否与机器人碰撞（穿模）
        if self.human_target.check_collision(self.robot.position, self.robot.dimensions):
            # 发生碰撞，调整位置避免穿模
            to_robot = self.robot.position - self.human_target.position
            to_robot[2] = 0  # 保持z坐标不变
            to_robot = to_robot / np.linalg.norm(to_robot)

            # 将人体移离机器人
            min_distance = 0.5  # 最小安全距离
            new_position = self.robot.position - to_robot * min_distance
            new_position[2] = 0.85  # 保持人体高度

            # 调整朝向，远离机器人
            new_yaw = np.degrees(np.arctan2(to_robot[0], to_robot[1]))
            self.human_target.orientation[2] = new_yaw

            logger.warning(
                f"检测到碰撞，调整人体位置避免穿模: {new_position[:2]}"
            )

        # 更新人体位置和速度
        self.human_target.position = new_position
        self.human_target.velocity = direction * 0.2

    def get_state(self) -> Dict[str, Any]:
        """
        获取当前环境状态

        Returns:
            Dict[str, Any]: 包含当前状态信息的字典
        """
        if not self.is_running:
            return {}

        return {
            'frame': self._get_camera_frame(),
            'frame_count': self.frame_count,
            'fps': 1.0 / (time.time() - self.last_frame_time) if time.time() > self.last_frame_time else 0.0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0.0,
            'robot_position': self.robot.position,
            'robot_orientation': self.robot.orientation,
            'joint_positions': self.robot.joint_positions,
            'joint_velocities': self.robot.joint_velocities
        }

    def save_scene_config(self, filename='env_config.json') -> bool:
        """
        保存场景参数到JSON文件

        Args:
            filename: 配置文件名，默认为'env_config.json'

        Returns:
            bool: 保存是否成功
        """
        try:
            # 构建场景参数字典
            scene_params = {
                'camera': {
                    'resolution': list(self.camera_resolution),
                    'fps': self.camera_fps,
                    'fov': 90.0,
                    'height': 1.2,
                    'pitch': -15.0
                },
                'robot': {
                    'model': self.robot.model,
                    'dimensions': list(self.robot.dimensions),
                    'initial_position': list(self.robot.position),
                    'initial_orientation': list(self.robot.orientation)
                },
                'human': {
                    'dimensions': list(self.human_target.dimensions),
                    'initial_position': list(self.human_target.position),
                    'initial_orientation': list(self.human_target.orientation),
                    'speed': self.human_target.speed,
                    'turn_angle': self.human_target.turn_angle,
                    'turn_interval': list(self.human_target.turn_interval_range),
                    'movement_range': self.human_target.movement_range
                },
                'environment': {
                    'gravity': [0.0, 0.0, -9.81],
                    'ground_level': 0.0
                }
            }

            # 如果有人体目标，更新其当前位置和朝向
            if self.human_target.is_active:
                scene_params['human']['current_position'] = list(self.human_target.position)
                scene_params['human']['current_orientation'] = list(self.human_target.orientation)
                scene_params['human']['current_velocity'] = list(self.human_target.velocity)

            # 保存到JSON文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(scene_params, f, indent=4, ensure_ascii=False)

            logger.info(f"场景参数已保存到 {filename}")
            return True

        except Exception as e:
            logger.error(f"保存场景参数失败: {str(e)}")
            return False

    def run_stability_test(self) -> bool:
        """
        运行稳定性测试

        Returns:
            bool: 测试是否成功完成（无崩溃）
        """
        if not self.is_initialized:
            logger.error("环境未初始化，无法运行测试")
            return False

        logger.info(f"开始稳定性测试，持续时间：{self.test_duration_seconds}秒")
        start_time = time.time()
        test_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"测试开始时间：{test_start_time}")

        if not self.start():
            logger.error("无法启动仿真环境")
            return False

        try:
            # 记录测试信息
            frame_count = 0
            fps_samples = []
            last_log_time = start_time

            while (time.time() - start_time) < self.test_duration_seconds:
                # 执行仿真步
                state = self.step()

                # 记录统计信息
                frame_count = state['frame_count']
                fps_samples.append(state['fps'])

                # 每秒记录一次状态
                if time.time() - last_log_time >= 1.0:
                    avg_fps = np.mean(fps_samples) if fps_samples else 0.0
                    min_fps = np.min(fps_samples) if fps_samples else 0.0
                    max_fps = np.max(fps_samples) if fps_samples else 0.0

                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"测试进度: {elapsed_time:.1f}s/{self.test_duration_seconds}s | "
                        f"帧数: {frame_count} | "
                        f"FPS: 当前={state['fps']:.1f}, 平均={avg_fps:.1f}, "
                        f"最小={min_fps:.1f}, 最大={max_fps:.1f}"
                    )

                    fps_samples = []

            # 测试完成
            test_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"测试结束时间：{test_end_time}")
            logger.info(f"稳定性测试完成，总帧数：{frame_count}")

            return True

        except Exception as e:
            logger.error(f"稳定性测试失败: {str(e)}")
            return False

        finally:
            # 关闭可视化
            if self.visualization_enabled:
                self.close_visualization()

            self.stop()
