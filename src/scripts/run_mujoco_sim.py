#!/usr/bin/env python3
"""
MuJoCo仿真运行示例
演示如何使用MuJoCo仿真平台进行可视化
"""

import sys
import os

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_env import Go2SimEnv
from visualizer import SimVisualizer
from config import logger


def main():
    """主函数"""
    try:
        # 创建仿真环境，启用MuJoCo
        logger.info("创建仿真环境...")
        env = Go2SimEnv(
            use_mujoco=True,  # 启用MuJoCo仿真平台
            mujoco_model_path=None  # 使用内置模型，也可以指定XML文件路径
        )

        # 初始化环境
        if not env.initialize():
            logger.error("环境初始化失败")
            return

        # 添加人体目标
        if not env.add_human_target():
            logger.error("添加人体目标失败")
            return

        # 创建可视化器，启用MuJoCo模式
        logger.info("创建可视化器...")
        visualizer = SimVisualizer(
            env=env,
            use_mujoco=True  # 启用MuJoCo可视化
        )

        # 初始化可视化器
        if not visualizer.initialize():
            logger.error("可视化器初始化失败")
            return

        # 启动环境
        if not env.start():
            logger.error("环境启动失败")
            return

        # 启动可视化
        logger.info("启动可视化...")
        visualizer.start()

        logger.info("仿真结束")

    except KeyboardInterrupt:
        logger.info("用户中断，停止仿真")
    except Exception as e:
        logger.error(f"仿真运行出错: {str(e)}")
    finally:
        # 清理资源
        if 'env' in locals():
            env.stop()


if __name__ == "__main__":
    main()
