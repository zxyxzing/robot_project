#!/usr/bin/env python3
"""
机器人模块
提供机器人模型和状态管理功能
"""

import numpy as np
from typing import Dict, Any
from config import (
    ROBOT_MODEL, ROBOT_DIMENSIONS, ROBOT_INITIAL_POSITION, 
    ROBOT_INITIAL_ORIENTATION, ROBOT_JOINT_COUNT, logger
)


class Robot:
    """机器人类"""

    def __init__(self):
        """初始化机器人"""
        self.model = ROBOT_MODEL
        self.dimensions = np.array(ROBOT_DIMENSIONS)
        self.position = np.array(ROBOT_INITIAL_POSITION)
        self.orientation = np.array(ROBOT_INITIAL_ORIENTATION)
        self.joint_count = ROBOT_JOINT_COUNT
        self.actuators = [
            'FL_hip', 'FL_thigh', 'FL_calf',
            'FR_hip', 'FR_thigh', 'FR_calf',
            'RL_hip', 'RL_thigh', 'RL_calf',
            'RR_hip', 'RR_thigh', 'RR_calf'
        ]
        self.joint_positions = np.zeros(self.joint_count)
        self.joint_velocities = np.zeros(self.joint_count)

    def update(self) -> None:
        """更新机器人状态"""
        # 模拟机器人状态更新
        # 添加一些随机运动
        self.position += np.random.normal(0, 0.001, 3)
        self.joint_positions += np.random.normal(0, 0.01, self.joint_count)
        self.joint_velocities += np.random.normal(0, 0.1, self.joint_count)

    def get_state(self) -> Dict[str, Any]:
        """
        获取机器人状态

        Returns:
            Dict[str, Any]: 包含机器人状态的字典
        """
        return {
            'model': self.model,
            'dimensions': self.dimensions,
            'position': self.position,
            'orientation': self.orientation,
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities
        }

    def set_position(self, position: np.ndarray) -> None:
        """
        设置机器人位置

        Args:
            position: 新的位置坐标
        """
        self.position = position.copy()

    def set_orientation(self, orientation: np.ndarray) -> None:
        """
        设置机器人朝向

        Args:
            orientation: 新的朝向（欧拉角）
        """
        self.orientation = orientation.copy()
