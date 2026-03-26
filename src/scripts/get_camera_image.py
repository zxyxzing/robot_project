#!/usr/bin/env python3
"""
从机器狗摄像头获取图像的程序
使用MuJoCo仿真环境中的摄像头获取图像并显示
"""

import os
import cv2
import numpy as np
import time
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('camera_capture.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入MuJoCo
try:
    import mujoco
    MUJOCO_AVAILABLE = True
    logger.info("MuJoCo已安装")
except ImportError:
    MUJOCO_AVAILABLE = False
    logger.error("MuJoCo未安装，程序无法运行")


class CameraCapture:
    """摄像头捕获类"""

    def __init__(self, model_path: str, camera_name: str = "go2_head"):
        """
        初始化摄像头捕获

        Args:
            model_path: MuJoCo模型文件路径
            camera_name: 摄像头名称，默认为"go2_head"
        """
        self.model_path = model_path
        self.camera_name = camera_name
        self.model = None
        self.data = None
        self.renderer = None
        self.camera_id = None
        self.is_initialized = False

        # 摄像头参数
        self.width = 640
        self.height = 480
        self.fps = 30

    def initialize(self) -> bool:
        """
        初始化摄像头

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
            logger.info(f"加载模型: {self.model_path}")
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)

            # 创建渲染器
            self.renderer = mujoco.Renderer(self.model)

            # 获取摄像头ID
            self.camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self.camera_name
            )

            if self.camera_id < 0:
                logger.error(f"未找到摄像头: {self.camera_name}")
                return False

            logger.info(f"摄像头初始化成功: {self.camera_name} (ID: {self.camera_id})")
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像

        Returns:
            np.ndarray: 捕获的图像，如果失败则返回None
        """
        if not self.is_initialized:
            logger.error("摄像头未初始化")
            return None

        try:
            # 更新仿真
            mujoco.mj_step(self.model, self.data)

            # 渲染图像
            self.renderer.update_scene(self.data, camera=self.camera_name)
            image = self.renderer.render()

            # 转换为RGB格式（MuJoCo默认为RGB，但OpenCV使用BGR）
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image

        except Exception as e:
            logger.error(f"捕获图像失败: {str(e)}")
            return None

    def get_single_frame(self) -> Optional[np.ndarray]:
        """
        获取单帧图像（适合其他程序调用）

        Returns:
            np.ndarray: 捕获的图像，如果失败则返回None
        """
        if not self.is_initialized:
            if not self.initialize():
                return None

        image = self.capture_frame()
        if image is not None:
            image = cv2.resize(image, (self.width, self.height))
        return image

    def run(self, duration: int = 60, save_images: bool = False):
        """
        运行摄像头捕获

        Args:
            duration: 运行时长（秒）
            save_images: 是否保存图像到文件
        """
        if not self.initialize():
            logger.error("初始化失败，无法运行")
            return

        logger.info(f"开始捕获图像，时长: {duration}秒")
        start_time = time.time()
        frame_count = 0

        # 创建显示窗口
        cv2.namedWindow(f"Camera: {self.camera_name}", cv2.WINDOW_NORMAL)

        # 创建保存目录
        if save_images:
            save_dir = "captured_images"
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"图像将保存到: {save_dir}")

        try:
            while time.time() - start_time < duration:
                # 捕获图像
                image = self.capture_frame()

                if image is not None:
                    # 调整图像大小
                    image = cv2.resize(image, (self.width, self.height))

                    # 显示图像
                    cv2.imshow(f"Camera: {self.camera_name}", image)

                    # 保存图像
                    if save_images:
                        image_path = os.path.join(
                            save_dir,
                            f"frame_{frame_count:06d}.jpg"
                        )
                        cv2.imwrite(image_path, image)

                    frame_count += 1

                    # 计算FPS
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    logger.info(f"帧: {frame_count}, FPS: {fps:.2f}")

                # 控制帧率
                key = cv2.waitKey(int(1000 / self.fps))
                if key == 27:  # ESC键退出
                    logger.info("用户按ESC键退出")
                    break

        except KeyboardInterrupt:
            logger.info("用户中断程序")
        except Exception as e:
            logger.error(f"运行出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            cv2.destroyAllWindows()
            logger.info(f"程序结束，共捕获 {frame_count} 帧")


def main():
    """主函数"""
    # 模型文件路径
    model_path = "/home/yuan/dog/venv_yolo_follow/src/model/go2_human_follow_new.xml"

    # 创建摄像头捕获器
    camera = CameraCapture(model_path, camera_name="go2_head")

    # 运行摄像头捕获（运行60秒，不保存图像）
    camera.run(duration=60, save_images=False)


if __name__ == "__main__":
    main()
