#!/usr/bin/env python3
"""
可视化模块
提供仿真环境的可视化功能
"""

import numpy as np
import cv2
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle
from config import CAMERA_RESOLUTION, CAMERA_FPS, logger


class SimVisualizer:
    """仿真环境可视化类"""

    def __init__(self, env, use_opencv=False, use_mujoco=False):
        """
        初始化可视化器

        Args:
            env: Go2SimEnv实例
            use_opencv: 是否使用OpenCV进行可视化，默认为False使用matplotlib
            use_mujoco: 是否使用MuJoCo进行可视化，默认为False
        """
        self.env = env
        self.use_opencv = use_opencv
        self.use_mujoco = use_mujoco
        
        # Matplotlib相关
        self.fig = None
        self.camera_ax = None
        self.fps_ax = None
        self.position_ax = None
        self.joint_ax = None
        self.map_ax = None
        self.mujoco_ax = None
        self.camera_image = None
        self.fps_line = None
        self.position_lines = []
        self.joint_bars = []
        self.status_text = None
        self.ani = None
        self.robot_marker = None
        self.human_marker = None
        self.human_direction = None
        
        # 2D地图相关
        self.robot_patch = None
        self.human_patch = None
        self.robot_traj_line = None
        self.human_traj_line = None
        self.robot_trajectory = []
        self.human_trajectory = []
        self.max_trajectory_length = 100
        
        # OpenCV相关
        self.cv_window_name = "Go2 Simulation"
        self.map_size = 400
        self.map_scale = 20  # 像素/米
        
        # 数据存储
        self.fps_data = {'time': [], 'fps': []}
        self.position_data = {'time': [], 'x': [], 'y': [], 'z': []}
        
        # MuJoCo相关
        self.mujoco_view = None
        self.mujoco_renderer = None
        
        logger.info(f"可视化器初始化完成，使用: {'OpenCV' if use_opencv else 'matplotlib'}" + 
                   f", MuJoCo: {'启用' if use_mujoco else '禁用'}")

    def initialize(self) -> bool:
        """
        初始化可视化界面

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("正在初始化可视化界面...")

            # 根据可视化方式选择初始化方法
            if self.use_mujoco:
                return self._init_mujoco()
            elif self.use_opencv:
                return self._init_opencv()
            else:
                return self._init_matplotlib()

        except Exception as e:
            logger.error(f"可视化界面初始化失败: {str(e)}")
            return False

    def _init_matplotlib(self) -> bool:
        """
        初始化matplotlib可视化界面

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("正在初始化matplotlib可视化界面...")

            # 创建图形和子图布局
            self.fig = plt.figure(figsize=(18, 10))
            gs = GridSpec(3, 4, figure=self.fig)

            # 摄像头画面（左上，占2x2）
            self.camera_ax = self.fig.add_subplot(gs[0:2, 0:2])
            self.camera_ax.set_title('摄像头画面')
            self.camera_ax.axis('off')
            self.camera_image = self.camera_ax.imshow(
                np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
            )

            # 2D地图视图（右上，占2x2）
            self.map_ax = self.fig.add_subplot(gs[0:2, 2:4])
            self.map_ax.set_title('2D地图视图')
            self.map_ax.set_xlabel('X (米)')
            self.map_ax.set_ylabel('Y (米)')
            self.map_ax.set_xlim(-10, 10)
            self.map_ax.set_ylim(-10, 10)
            self.map_ax.grid(True)
            self.map_ax.set_aspect('equal')

            # 创建机器人矩形
            robot_width = self.env.robot.dimensions[0]
            robot_length = self.env.robot.dimensions[1]
            self.robot_patch = Rectangle(
                (0, 0), robot_width, robot_length,
                angle=0,
                facecolor='blue', alpha=0.7, label='机器人'
            )
            self.map_ax.add_patch(self.robot_patch)

            # 创建人体目标圆形
            human_radius = self.env.human_target.dimensions[0] / 2
            self.human_patch = Circle(
                (0, 0), human_radius,
                facecolor='red', alpha=0.7, label='人体目标'
            )
            self.map_ax.add_patch(self.human_patch)

            # 创建轨迹线
            self.robot_traj_line, = self.map_ax.plot([], [], 'b--', linewidth=1, alpha=0.5, label='机器人轨迹')
            self.human_traj_line, = self.map_ax.plot([], [], 'r--', linewidth=1, alpha=0.5, label='人体轨迹')
            self.map_ax.legend(loc='upper right')

            # FPS曲线（中下）
            self.fps_ax = self.fig.add_subplot(gs[2, 0:2])
            self.fps_ax.set_title('FPS')
            self.fps_ax.set_xlabel('时间（秒）')
            self.fps_ax.set_ylabel('FPS')
            self.fps_ax.grid(True)
            self.fps_line, = self.fps_ax.plot([], [], 'b-', linewidth=2)

            # 机器人位置曲线（右下）
            self.position_ax = self.fig.add_subplot(gs[2, 2:4])
            self.position_ax.set_title('机器人位置')
            self.position_ax.set_xlabel('时间（秒）')
            self.position_ax.set_ylabel('位置（米）')
            self.position_ax.grid(True)
            colors = ['r', 'g', 'b']
            labels = ['X', 'Y', 'Z']
            for i in range(3):
                line, = self.position_ax.plot([], [], color=colors[i], label=labels[i], linewidth=2)
                self.position_lines.append(line)
            self.position_ax.legend()

            # 在摄像头画面上添加人体和机器人的标记
            self.robot_marker, = self.camera_ax.plot([], [], 'ro', markersize=10, label='机器人')
            self.human_marker, = self.camera_ax.plot([], [], 'go', markersize=15, label='人体')
            self.human_direction, = self.camera_ax.plot([], [], 'g-', linewidth=2)
            self.camera_ax.legend(loc='upper right')

            # 状态文本显示
            self.status_text = self.fig.text(
                0.02, 0.02,
                '',
                fontsize=10,
                family='monospace',
                bbox=dict(facecolor='white', alpha=0.8)
            )

            plt.tight_layout()
            logger.info("matplotlib可视化界面初始化完成")
            return True

        except Exception as e:
            logger.error(f"matplotlib可视化界面初始化失败: {str(e)}")
            return False

    def _init_mujoco(self) -> bool:
        """
        初始化MuJoCo可视化界面

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("正在初始化MuJoCo可视化界面...")
            
            # 检查MuJoCo是否可用
            if not self.env.use_mujoco or self.env.mujoco_integration is None:
                logger.warning("MuJoCo未启用，切换到matplotlib模式")
                return self._init_matplotlib()
            
            # 初始化MuJoCo环境
            if not self.env.mujoco_integration.is_initialized:
                if not self.env.mujoco_integration.initialize():
                    logger.warning("MuJoCo初始化失败，切换到matplotlib模式")
                    return self._init_matplotlib()
            
            logger.info("MuJoCo可视化界面初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"MuJoCo可视化界面初始化失败: {str(e)}")
            logger.info("切换到matplotlib模式")
            return self._init_matplotlib()

    def update(self, frame=None):
        """
        更新可视化界面

        Args:
            frame: 帧号（可选，仅用于matplotlib动画）

        Returns:
            list: 更新的图形对象列表（仅用于matplotlib）
        """
        # 获取当前状态
        state = self.env.get_state()
        if not state:
            return [self.camera_image, self.fps_line] if not self.use_opencv else None

        # 根据可视化方式选择更新方法
        if self.use_mujoco and self.mujoco_renderer is not None:
            return self._update_mujoco(state)
        elif self.use_opencv:
            self._update_opencv(state)
            return None
        else:
            return self._update_matplotlib(state)

        # 更新摄像头画面
        self.camera_image.set_data(state['frame'])

        # 更新FPS曲线
        current_time = self.env.frame_count / CAMERA_FPS
        self.fps_data['time'].append(current_time)
        self.fps_data['fps'].append(state['fps'])

        # 只保留最近100个数据点
        if len(self.fps_data['time']) > 100:
            self.fps_data['time'].pop(0)
            self.fps_data['fps'].pop(0)

        self.fps_line.set_data(self.fps_data['time'], self.fps_data['fps'])
        self.fps_ax.set_xlim(max(0, current_time - 10), current_time + 1)
        self.fps_ax.set_ylim(0, max(100, max(self.fps_data['fps']) * 1.1))

        # 更新机器人位置曲线
        position = state['robot_position']
        self.position_data['time'].append(current_time)
        self.position_data['x'].append(position[0])
        self.position_data['y'].append(position[1])
        self.position_data['z'].append(position[2])

        # 只保留最近100个数据点
        if len(self.position_data['time']) > 100:
            self.position_data['time'].pop(0)
            self.position_data['x'].pop(0)
            self.position_data['y'].pop(0)
            self.position_data['z'].pop(0)

        self.position_lines[0].set_data(self.position_data['time'], self.position_data['x'])
        self.position_lines[1].set_data(self.position_data['time'], self.position_data['y'])
        self.position_lines[2].set_data(self.position_data['time'], self.position_data['z'])
        self.position_ax.set_xlim(max(0, current_time - 10), current_time + 1)

        # 计算Y轴范围
        all_positions = (self.position_data['x'] + 
                        self.position_data['y'] + 
                        self.position_data['z'])
        if all_positions:
            min_pos = min(all_positions)
            max_pos = max(all_positions)
            self.position_ax.set_ylim(min_pos - 0.1, max_pos + 0.1)

        # 更新关节角度
        joint_positions = state['joint_positions']
        for i, bar in enumerate(self.joint_bars):
            bar.set_height(joint_positions[i])

        # 更新状态文本
        status_text = (
            f"运行时间: {current_time:.1f}s\n"
            f"帧数: {state['frame_count']}\n"
            f"当前FPS: {state['fps']:.1f}\n"
            f"平均FPS: {state['avg_fps']:.1f}\n"
            f"机器人位置: X={position[0]:.3f}, Y={position[1]:.3f}, Z={position[2]:.3f}m"
        )

        # 添加人体目标信息
        if self.env.human_target is not None and self.env.human_target['is_active']:
            human_pos = self.env.human_target['position']
            human_vel = self.env.human_target['velocity']
            human_speed = np.linalg.norm(human_vel)
            status_text += (
                f"\n人体位置: X={human_pos[0]:.3f}, Y={human_pos[1]:.3f}, Z={human_pos[2]:.3f}m"
                f"\n人体速度: {human_speed:.3f}m/s"
            )

            # 更新人体和机器人的标记
            # 将3D位置映射到2D图像坐标（简单投影）
            # 假设摄像头位于原点，朝向Y轴正方向
            scale = 100  # 缩放因子
            center_x = CAMERA_RESOLUTION[0] / 2
            center_y = CAMERA_RESOLUTION[1] / 2

            # 机器人标记
            robot_x = center_x + position[0] * scale
            robot_y = center_y - position[1] * scale
            self.robot_marker.set_data([robot_x], [robot_y])

            # 人体标记
            human_x = center_x + human_pos[0] * scale
            human_y = center_y - human_pos[1] * scale
            self.human_marker.set_data([human_x], [human_y])

            # 人体朝向指示线
            yaw = np.radians(self.env.human_target['orientation'][2])
            direction_length = 0.5  # 指示线长度（米）
            end_x = human_x + np.sin(yaw) * direction_length * scale
            end_y = human_y - np.cos(yaw) * direction_length * scale
            self.human_direction.set_data([human_x, end_x], [human_y, end_y])

        self.status_text.set_text(status_text)

        return [self.camera_image, self.fps_line] + self.position_lines + list(self.joint_bars) + [self.robot_marker, self.human_marker, self.human_direction]

    def start(self):
        """启动可视化"""
        try:
            logger.info("正在启动可视化...")
            
            if self.use_mujoco and self.env.mujoco_integration is not None:
                # MuJoCo模式：使用MuJoCo查看器
                def mujoco_update_callback():
                    """MuJoCo更新回调函数"""
                    self.env.step()
                
                self.env.mujoco_integration.launch_viewer(
                    mujoco_update_callback,
                    self.env.camera_fps
                )
            elif self.use_opencv:
                # OpenCV模式：直接运行循环
                while self.env.is_running:
                    self.update()
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
            else:
                # matplotlib模式：使用动画
                self.ani = animation.FuncAnimation(
                    self.fig,
                    self.update,
                    interval=1000 // CAMERA_FPS,
                    blit=True
                )
                plt.show()
            logger.info("可视化已关闭")
        except Exception as e:
            logger.error(f"可视化启动失败: {str(e)}")

    def _update_matplotlib(self, state):
        """
        更新matplotlib可视化界面

        Args:
            state: 当前状态字典

        Returns:
            list: 更新的图形对象列表
        """
        # 更新摄像头画面
        self.camera_image.set_data(state['frame'])

        # 更新2D地图
        self._update_2d_map(state)

        # 更新FPS曲线
        current_time = self.env.frame_count / CAMERA_FPS
        self.fps_data['time'].append(current_time)
        self.fps_data['fps'].append(state['fps'])

        # 只保留最近100个数据点
        if len(self.fps_data['time']) > 100:
            self.fps_data['time'].pop(0)
            self.fps_data['fps'].pop(0)

        self.fps_line.set_data(self.fps_data['time'], self.fps_data['fps'])
        self.fps_ax.set_xlim(max(0, current_time - 10), current_time + 1)
        self.fps_ax.set_ylim(0, max(100, max(self.fps_data['fps']) * 1.1))

        # 更新机器人位置曲线
        position = state['robot_position']
        self.position_data['time'].append(current_time)
        self.position_data['x'].append(position[0])
        self.position_data['y'].append(position[1])
        self.position_data['z'].append(position[2])

        # 只保留最近100个数据点
        if len(self.position_data['time']) > 100:
            self.position_data['time'].pop(0)
            self.position_data['x'].pop(0)
            self.position_data['y'].pop(0)
            self.position_data['z'].pop(0)

        self.position_lines[0].set_data(self.position_data['time'], self.position_data['x'])
        self.position_lines[1].set_data(self.position_data['time'], self.position_data['y'])
        self.position_lines[2].set_data(self.position_data['time'], self.position_data['z'])
        self.position_ax.set_xlim(max(0, current_time - 10), current_time + 1)

        # 计算Y轴范围
        all_positions = (self.position_data['x'] +
                        self.position_data['y'] +
                        self.position_data['z'])
        if all_positions:
            min_pos = min(all_positions)
            max_pos = max(all_positions)
            self.position_ax.set_ylim(min_pos - 0.1, max_pos + 0.1)

        # 更新状态文本
        status_text = self._generate_status_text(state, current_time, position)
        self.status_text.set_text(status_text)

        return [self.camera_image, self.fps_line] + self.position_lines + [self.robot_marker, self.human_marker, self.human_direction]

    def _update_2d_map(self, state):
        """
        更新2D地图视图

        Args:
            state: 当前状态字典
        """
        # 更新机器人位置
        robot_width = self.env.robot.dimensions[0]
        robot_length = self.env.robot.dimensions[1]
        robot_x = state['robot_position'][0]
        robot_y = state['robot_position'][1]
        robot_yaw = state['robot_orientation'][2]
        
        self.robot_patch.set_xy((robot_x - robot_width/2, robot_y - robot_length/2))
        self.robot_patch.angle = np.degrees(robot_yaw)

        # 更新人体目标位置
        human_radius = self.env.human_target.dimensions[0] / 2
        human_pos = self.env.human_target.position[:2]
        self.human_patch.center = human_pos

        # 更新轨迹
        self.robot_trajectory.append([robot_x, robot_y])
        self.human_trajectory.append(human_pos)

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

    def _update_opencv(self, state):
        """
        使用OpenCV更新可视化界面

        Args:
            state: 当前状态字典
        """
        # 创建组合画面
        combined = self._create_opencv_display(state)
        
        # 显示画面
        cv2.imshow(self.cv_window_name, combined)
        cv2.waitKey(1)

    def _create_opencv_display(self, state):
        """
        创建OpenCV显示画面

        Args:
            state: 当前状态字典

        Returns:
            np.ndarray: 组合后的显示画面
        """
        # 创建空白画布
        display_width = 1200
        display_height = 800
        combined = np.ones((display_height, display_width, 3), dtype=np.uint8) * 255

        # 左侧：摄像头画面 (800x600)
        camera_img = cv2.resize(state['frame'], (800, 600))
        combined[0:600, 0:800] = camera_img

        # 右侧上部：2D地图 (400x400)
        map_img = self._create_2d_map_opencv(state)
        combined[0:400, 800:1200] = map_img

        # 右侧下部：状态信息 (400x200)
        status_img = self._create_status_display_opencv(state)
        combined[600:800, 800:1200] = status_img

        # 添加标签
        cv2.putText(combined, "Camera View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "2D Map", (810, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Status", (810, 630), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return combined

    def _create_2d_map_opencv(self, state):
        """
        创建OpenCV格式的2D地图

        Args:
            state: 当前状态字典

        Returns:
            np.ndarray: 2D地图图像
        """
        # 创建空白画布
        map_size = 400
        scale = 20  # 像素/米
        center = map_size // 2
        map_img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255

        # 绘制网格
        for i in range(-10, 11):
            pos = int(center + i * scale)
            cv2.line(map_img, (pos, 0), (pos, map_size), (200, 200, 200), 1)
            cv2.line(map_img, (0, pos), (map_size, pos), (200, 200, 200), 1)

        # 绘制机器人
        robot_x = int(center + state['robot_position'][0] * scale)
        robot_y = int(center + state['robot_position'][1] * scale)
        robot_width = int(self.env.robot.dimensions[0] * scale)
        robot_length = int(self.env.robot.dimensions[1] * scale)
        robot_yaw = state['robot_orientation'][2]

        # 计算机器人的四个角点
        cos_a, sin_a = np.cos(robot_yaw), np.sin(robot_yaw)
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
            rotated_corners.append((robot_x + rx, robot_y + ry))

        cv2.fillPoly(map_img, [np.array(rotated_corners)], (255, 0, 0))

        # 绘制人体目标
        human_pos = self.env.human_target.position[:2]
        human_x = int(center + human_pos[0] * scale)
        human_y = int(center + human_pos[1] * scale)
        human_radius = int(self.env.human_target.dimensions[0] / 2 * scale)
        cv2.circle(map_img, (human_x, human_y), human_radius, (0, 0, 255), -1)

        # 绘制轨迹
        if len(self.robot_trajectory) > 1:
            traj_points = [
                (int(center + x * scale), int(center + y * scale))
                for x, y in self.robot_trajectory
            ]
            cv2.polylines(map_img, [np.array(traj_points)], False, (255, 0, 0), 2)

        if len(self.human_trajectory) > 1:
            traj_points = [
                (int(center + x * scale), int(center + y * scale))
                for x, y in self.human_trajectory
            ]
            cv2.polylines(map_img, [np.array(traj_points)], False, (0, 0, 255), 2)

        return map_img

    def _create_status_display_opencv(self, state):
        """
        创建OpenCV格式的状态显示

        Args:
            state: 当前状态字典

        Returns:
            np.ndarray: 状态显示图像
        """
        # 创建空白画布
        status_img = np.ones((200, 400, 3), dtype=np.uint8) * 255

        # 准备状态文本
        current_time = self.env.frame_count / CAMERA_FPS
        position = state['robot_position']
        status_text = self._generate_status_text(state, current_time, position)

        # 绘制状态文本
        y_offset = 30
        for line in status_text.split('\n'):
            cv2.putText(status_img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 20

        return status_img

    def _generate_status_text(self, state, current_time, position):
        """
        生成状态文本

        Args:
            state: 当前状态字典
            current_time: 当前时间
            position: 机器人位置

        Returns:
            str: 状态文本
        """
        status_text = (
            f"运行时间: {current_time:.1f}s\n"
            f"帧数: {state['frame_count']}\n"
            f"当前FPS: {state['fps']:.1f}\n"
            f"平均FPS: {state['avg_fps']:.1f}\n"
            f"机器人位置: X={position[0]:.3f}, Y={position[1]:.3f}, Z={position[2]:.3f}m"
        )

        # 添加人体目标信息
        if self.env.human_target.is_active:
            human_pos = self.env.human_target.position
            human_vel = self.env.human_target.velocity
            human_speed = np.linalg.norm(human_vel)
            status_text += (
                f"\n人体位置: X={human_pos[0]:.3f}, Y={human_pos[1]:.3f}, Z={human_pos[2]:.3f}m"
                f"\n人体速度: {human_speed:.3f}m/s"
            )

        return status_text

    def _update_mujoco(self, state):
        """
        使用MuJoCo更新可视化界面

        Args:
            state: 当前状态字典

        Returns:
            list: 更新的图形对象列表
        """
        try:
            # 更新MuJoCo仿真状态
            if self.env.mujoco_integration is not None:
                self.env.mujoco_integration.step()
                
                # 更新MuJoCo中的机器人位置
                self.env.mujoco_integration.update_body_position(
                    "robot",
                    self.env.robot.position
                )
                
                # 更新MuJoCo中的人体位置
                if self.env.human_target.is_active:
                    self.env.mujoco_integration.update_body_position(
                        "human",
                        self.env.human_target.position
                    )
            
            # 使用matplotlib更新其他视图
            return self._update_matplotlib(state)
            
        except Exception as e:
            logger.error(f"MuJoCo更新失败: {str(e)}")
            # 出错时切换到matplotlib模式
            return self._update_matplotlib(state)
