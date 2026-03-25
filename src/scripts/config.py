#!/usr/bin/env python3
"""
配置模块
包含仿真环境的所有配置参数和日志设置
"""

import logging
from typing import Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sim_env.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 摄像头配置
CAMERA_RESOLUTION: Tuple[int, int] = (1920, 1080)
CAMERA_FPS: int = 200
CAMERA_FOV: float = 90.0  # 摄像头视场角（度）
CAMERA_HEIGHT: float = 1.2  # 摄像头高度（米）
CAMERA_PITCH: float = -15.0  # 摄像头俯仰角（度）

# 机器人配置
ROBOT_MODEL: str = 'Go2'
ROBOT_DIMENSIONS: Tuple[float, float, float] = (0.5, 0.5, 0.6)  # 机器人尺寸（长×宽×高）
ROBOT_INITIAL_POSITION: Tuple[float, float, float] = (0.0, 0.0, 0.3)  # 初始位置
ROBOT_INITIAL_ORIENTATION: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 初始朝向（欧拉角）
ROBOT_JOINT_COUNT: int = 12

# 人体配置
HUMAN_DIMENSIONS: Tuple[float, float, float] = (0.6, 0.4, 1.7)  # 人体尺寸（宽×深×高）
HUMAN_INITIAL_POSITION: Tuple[float, float, float] = (0.0, 3.0, 0.85)  # 初始位置
HUMAN_INITIAL_ORIENTATION: Tuple[float, float, float] = (0.0, 180.0, 0.0)  # 初始朝向
HUMAN_SPEED: float = 0.2  # 移动速度（m/s）
HUMAN_TURN_ANGLE: float = 10.0  # 最大转向角度（度）
HUMAN_TURN_INTERVAL: Tuple[float, float] = (2.0, 5.0)  # 转向间隔范围（秒）
HUMAN_MOVEMENT_RANGE: float = 5.0  # 运动范围（米）

# 环境配置
GRAVITY: Tuple[float, float, float] = (0.0, 0.0, -9.81)  # 重力加速度
GROUND_LEVEL: float = 0.0  # 地面高度

# 测试配置
TEST_DURATION_MINUTES: int = 30  # 测试持续时间（分钟）

# MuJoCo配置
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.warning("MuJoCo未安装，将使用模拟模式")
