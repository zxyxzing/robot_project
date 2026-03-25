#!/usr/bin/env python3
"""
Go2机器人控制程序
在MuJoCo仿真平台中控制Go2机器人
"""

import os
import time
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('go2_control.log'),
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


class Go2Controller:
    """Go2机器人控制类"""

    def __init__(self, model_path: str):
        """
        初始化Go2控制器

        Args:
            model_path: Go2机器人的MuJoCo模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.data = None
        self.viewer = None
        self.is_running = False

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

            # 输出模型信息
            logger.info(f"模型加载成功: {self.model.nbody}个body, {self.model.njnt}个关节, {self.model.nu}个执行器")

            # 输出body名称
            logger.info("Body列表:")
            for i in range(self.model.nbody):
                logger.info(f"  {i}: {self.model.body(i).name}")

            logger.info("MuJoCo环境初始化成功")
            return True

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def step(self) -> None:
        """执行一步仿真"""
        if not self.is_running:
            return

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
        logger.info("开始Go2仿真")

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
    # Go2和人体模型路径
    model_path = "/home/yuan/dog/venv_yolo_follow/src/model/go2_human_follow_new.xml"

    # 创建Go2控制器
    controller = Go2Controller(model_path)
    # 运行仿真
    controller.run()


if __name__ == "__main__":
    main()
