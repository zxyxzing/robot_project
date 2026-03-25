#!/usr/bin/env python3
"""
人体目标模块
提供人体目标模型和运动控制功能
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from config import (
    HUMAN_DIMENSIONS, HUMAN_INITIAL_POSITION, HUMAN_INITIAL_ORIENTATION,
    HUMAN_SPEED, HUMAN_TURN_ANGLE, HUMAN_TURN_INTERVAL, HUMAN_MOVEMENT_RANGE,
    CAMERA_FPS, CAMERA_FOV, CAMERA_HEIGHT, CAMERA_PITCH, logger
)


class HumanTarget:
    """人体目标类"""

    def __init__(self):
        """初始化人体目标"""
        self.dimensions = np.array(HUMAN_DIMENSIONS)
        self.position = np.array(HUMAN_INITIAL_POSITION)
        self.orientation = np.array(HUMAN_INITIAL_ORIENTATION)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.speed = HUMAN_SPEED
        self.turn_angle = HUMAN_TURN_ANGLE
        self.turn_interval_range = HUMAN_TURN_INTERVAL
        self.movement_range = HUMAN_MOVEMENT_RANGE
        self.is_active = False
        self.last_turn_time = 0.0
        self.turn_interval = 0.0

    def activate(self) -> None:
        """激活人体目标"""
        self.is_active = True
        self.last_turn_time = time.time()
        self.turn_interval = np.random.uniform(*self.turn_interval_range)
        logger.info("人体目标已激活")

    def deactivate(self) -> None:
        """停用人体目标"""
        self.is_active = False
        logger.info("人体目标已停用")

    def update(self, robot_position: np.ndarray) -> None:
        """
        更新人体目标的位置和朝向
        人体以0.2m/s的速度移动，每隔2-5秒随机转向±10°
        运动范围限制在±5m内，同时确保不超出摄像头视野

        Args:
            robot_position: 机器人位置，用于碰撞检测
        """
        if not self.is_active:
            return

        current_time = time.time()
        dt = 1.0 / CAMERA_FPS  # 时间步长

        # 计算摄像头视野范围
        # 基于摄像头高度和俯仰角计算可见距离
        fov_rad = np.radians(CAMERA_FOV / 2)
        pitch_rad = np.radians(CAMERA_PITCH)

        # 计算可见距离范围（考虑摄像头俯仰角）
        min_distance = CAMERA_HEIGHT * np.tan(np.abs(pitch_rad) - fov_rad)
        max_distance = CAMERA_HEIGHT * np.tan(np.abs(pitch_rad) + fov_rad)

        # 确保最小距离为正
        min_distance = max(0.5, min_distance)  # 至少0.5m
        max_distance = min(self.movement_range, max_distance)  # 不超过运动范围

        # 计算可见角度范围（水平方向）
        # 假设摄像头朝向Y轴正方向，水平视场角为CAMERA_FOV
        visible_angle_range = np.radians(CAMERA_FOV)

        # 检查是否需要转向
        if current_time - self.last_turn_time >= self.turn_interval:
            # 计算当前人体相对于机器人的角度
            human_pos = self.position
            angle_to_human = np.arctan2(human_pos[0], human_pos[1])

            # 随机转向±10°，但限制在可见角度范围内
            turn_angle = np.random.uniform(-self.turn_angle, self.turn_angle)
            new_yaw = self.orientation[2] + turn_angle

            # 计算新朝向相对于机器人的角度
            new_yaw_rad = np.radians(new_yaw)

            # 检查新朝向是否会导致人体移出视野
            # 预测未来几步的位置
            prediction_steps = int(1.0 / dt)  # 预测1秒
            predicted_pos = human_pos.copy()
            predicted_yaw = new_yaw_rad

            will_exit_fov = False
            for _ in range(prediction_steps):
                predicted_pos += np.array([np.sin(predicted_yaw), np.cos(predicted_yaw), 0.0]) * self.speed * dt

                # 检查距离
                dist_from_robot = np.sqrt(predicted_pos[0]**2 + predicted_pos[1]**2)
                if dist_from_robot < min_distance or dist_from_robot > max_distance:
                    will_exit_fov = True
                    break

                # 检查角度
                angle_to_predicted = np.arctan2(predicted_pos[0], predicted_pos[1])
                if abs(angle_to_predicted) > visible_angle_range / 2:
                    will_exit_fov = True
                    break

            if will_exit_fov:
                # 如果会移出视野，调整转向角度使其朝向视野中心
                center_angle = 0.0  # 摄像头朝向Y轴正方向
                desired_yaw = center_angle + np.random.uniform(-10, 10)  # 朝向中心附近
                self.orientation[2] = desired_yaw

                logger.debug(
                    f"人体即将移出视野，调整朝向: {desired_yaw:.1f}°, "
                    f"可见角度范围: ±{np.degrees(visible_angle_range/2):.1f}°"
                )
            else:
                self.orientation[2] = new_yaw

            self.last_turn_time = current_time
            self.turn_interval = np.random.uniform(*self.turn_interval_range)

            logger.debug(
                f"人体转向: {turn_angle:.1f}°, "
                f"当前朝向: {self.orientation[2]:.1f}°"
            )

        # 计算移动方向（基于当前朝向）
        yaw = np.radians(self.orientation[2])
        direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])

        # 计算新位置
        new_position = self.position + direction * self.speed * dt

        # 检查是否超出运动范围（±5m）
        distance_from_origin = np.sqrt(new_position[0]**2 + new_position[1]**2)
        if distance_from_origin > self.movement_range:
            # 超出范围，朝向原点方向
            to_origin = -self.position
            to_origin[2] = 0  # 保持z坐标不变
            to_origin = to_origin / np.linalg.norm(to_origin)

            # 计算新的朝向
            new_yaw = np.degrees(np.arctan2(to_origin[0], to_origin[1]))
            self.orientation[2] = new_yaw

            # 重新计算新位置
            yaw = np.radians(new_yaw)
            direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])
            new_position = self.position + direction * self.speed * dt

            logger.debug(
                f"人体超出运动范围，调整朝向: {new_yaw:.1f}°, "
                f"当前位置: {self.position[:2]}"
            )

        # 检查是否超出摄像头视野
        dist_from_robot = np.sqrt(new_position[0]**2 + new_position[1]**2)
        angle_to_human = np.arctan2(new_position[0], new_position[1])

        if dist_from_robot < min_distance or dist_from_robot > max_distance:
            # 距离超出视野，调整位置
            clamped_dist = np.clip(dist_from_robot, min_distance, max_distance)
            new_position[0] = clamped_dist * np.sin(angle_to_human)
            new_position[1] = clamped_dist * np.cos(angle_to_human)

            logger.debug(
                f"人体距离超出视野，调整位置: {dist_from_robot:.2f}m -> {clamped_dist:.2f}m"
            )

        if abs(angle_to_human) > visible_angle_range / 2:
            # 角度超出视野，调整朝向
            clamped_angle = np.clip(angle_to_human, -visible_angle_range/2, visible_angle_range/2)
            self.orientation[2] = np.degrees(clamped_angle)

            # 重新计算移动方向
            yaw = np.radians(self.orientation[2])
            direction = np.array([np.sin(yaw), np.cos(yaw), 0.0])

            logger.debug(
                f"人体角度超出视野，调整朝向: {np.degrees(angle_to_human):.1f}° -> {np.degrees(clamped_angle):.1f}°"
            )

        # 更新位置和速度
        self.position = new_position
        self.velocity = direction * self.speed

    def check_collision(self, robot_position: np.ndarray, robot_dimensions: np.ndarray) -> bool:
        """
        检查是否与机器人发生碰撞（穿模）
        使用AABB（轴对齐包围盒）碰撞检测算法

        Args:
            robot_position: 机器人位置
            robot_dimensions: 机器人尺寸

        Returns:
            bool: 是否发生碰撞
        """
        # 人体包围盒
        human_min = self.position - self.dimensions / 2
        human_max = self.position + self.dimensions / 2

        # 机器人包围盒
        robot_min = robot_position - robot_dimensions / 2
        robot_max = robot_position + robot_dimensions / 2

        # 检查是否碰撞
        collision = (
            human_min[0] < robot_max[0] and human_max[0] > robot_min[0] and
            human_min[1] < robot_max[1] and human_max[1] > robot_min[1] and
            human_min[2] < robot_max[2] and human_max[2] > robot_min[2]
        )

        if collision:
            logger.warning(
                f"检测到碰撞！人体位置: {self.position}, "
                f"机器人位置: {robot_position}"
            )

        return collision

    def get_state(self) -> Dict[str, Any]:
        """
        获取人体目标状态

        Returns:
            Dict[str, Any]: 包含人体目标状态的字典
        """
        return {
            'dimensions': self.dimensions,
            'position': self.position,
            'orientation': self.orientation,
            'velocity': self.velocity,
            'speed': self.speed,
            'is_active': self.is_active
        }
