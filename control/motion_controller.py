from typing import Dict, List
import numpy as np

class MotionController:
    """
    四足机器人人体跟随运动控制器（PI控制）
    符合团队接口规范，可被集成模块直接调用
    """
    def __init__(
        self,
        kp_dist=0.0000001,
        ki_dist=0.0000001,
        kp_x=0.0005,
        ki_x=0.00005,
        integral_limit=0.05
    ):
        """
        初始化PI控制器
        Args:
            kp_dist, ki_dist: 前后距离控制参数
            kp_x, ki_x: 左右偏移控制参数
            integral_limit: 积分项限幅，防止超调
        """
        # PI控制参数
        self.kp_dist = kp_dist
        self.ki_dist = ki_dist
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.integral_limit = integral_limit

        # 积分项初始化
        self.integral_dist = 0.0
        self.integral_x = 0.0

        # 速度限幅（任务要求）
        self.max_vx = 0.15   # 前后最大速度 (m/s)
        self.max_vy = 0.1    # 左右最大速度 (m/s)
        self.max_yaw_rate = 0.5  # 最大角速度 (rad/s)

    def reset_integral(self):
        """重置积分项（丢失目标/重新跟随场景调用）"""
        self.integral_dist = 0.0
        self.integral_x = 0.0

    def compute_command(self, target_info: Dict, current_pose: List) -> Dict:
        """
        控制指令计算方法（团队标准接口）
        Args:
            target_info: 视觉模块输出的目标信息字典
                必须包含 'norm_x' 和 'norm_y' 字段
            current_pose: 机器人当前位姿 [x, y, theta]（本阶段暂不使用）
        Returns:
            dict: 标准控制指令
                - 'velocity': [vx, vy] 线速度 (m/s)
                - 'yaw_rate': float 角速度 (rad/s)
        """
        # 1. 从视觉模块的输出中提取归一化坐标
        norm_x = target_info['norm_x']
        norm_y = target_info['norm_y']

        # 2. 前后速度 vx（控制前后跟随）
        p_dist = self.kp_dist * norm_y
        self.integral_dist = np.clip(
            self.integral_dist + self.ki_dist * norm_y * 0.01,
            -self.integral_limit, self.integral_limit
        )
        vx = p_dist + self.integral_dist

        # 3. 左右速度 vy（控制横向跟随）
        p_x = self.kp_x * norm_x
        self.integral_x = np.clip(
            self.integral_x + self.ki_x * norm_x * 0.01,
            -self.integral_limit, self.integral_limit
        )
        vy = p_x + self.integral_x

        # 4. 速度限幅保护
        vx = np.clip(vx, -self.max_vx, self.max_vx)
        vy = np.clip(vy, -self.max_vy, self.max_vy)

        # 5. 计算转向角速度（根据横向偏差控制转向）
        yaw_rate = -norm_x * 0.3
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # 6. 返回团队约定的标准指令格式
        return {
            'velocity': [vx, vy],
            'yaw_rate': yaw_rate
        }

    def stop(self) -> Dict:
        """停止机器人，输出零速度指令"""
        self.reset_integral()
        return {
            'velocity': [0.0, 0.0],
            'yaw_rate': 0.0
        }

    def close(self):
        """清理控制器资源"""
        self.reset_integral()

