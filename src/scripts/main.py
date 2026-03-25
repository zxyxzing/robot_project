#!/usr/bin/env python3
"""
Go2仿真环境主程序
提供命令行接口和程序入口
"""

import sys
from datetime import datetime
from config import MUJOCO_AVAILABLE, logger
from simulation_env import Go2SimEnv
from visualizer import SimVisualizer


def main():
    """主函数"""
    # 检查命令行参数，决定是否启用可视化
    enable_visualization = "--visual" in sys.argv or "-v" in sys.argv or True  # 默认启用可视化

    # 检查是否使用MuJoCo
    use_mujoco = "--mujoco" in sys.argv or "-m" in sys.argv or MUJOCO_AVAILABLE  # 默认使用MuJoCo（如果可用）
    mujoco_model_path = None

    # 检查是否指定了MuJoCo模型文件
    if "--model" in sys.argv:
        try:
            model_idx = sys.argv.index("--model")
            if model_idx + 1 < len(sys.argv):
                mujoco_model_path = sys.argv[model_idx + 1]
        except ValueError:
            pass
    
    # 如果没有指定模型文件，使用Go2机器人的MJCF模型
    if mujoco_model_path is None:
        mujoco_model_path = "/home/yuan/dog/venv_yolo_follow/src/model/go2_mujoco.xml"

    logger.info("Go2仿真环境稳定性测试程序启动")
    if enable_visualization:
        logger.info("可视化模式已启用")
    else:
        logger.info("非可视化模式（使用 --visual 或 -v 参数启用可视化）")

    if use_mujoco:
        if MUJOCO_AVAILABLE:
            logger.info("MuJoCo模式已启用")
            if mujoco_model_path:
                logger.info(f"使用MuJoCo模型文件: {mujoco_model_path}")
            else:
                logger.info("使用内置MuJoCo模型")
        else:
            logger.warning("MuJoCo未安装，将使用模拟模式")
            use_mujoco = False

    # 创建仿真环境
    env = Go2SimEnv(
        camera_resolution=(1920, 1080),
        camera_fps=200,
        test_duration_minutes=30,
        use_mujoco=use_mujoco,
        mujoco_model_path=mujoco_model_path
    )

    # 初始化环境
    if not env.initialize():
        logger.error("环境初始化失败，程序退出")
        return

    # 添加人体目标
    if not env.add_human_target():
        logger.error("添加人体目标失败，程序退出")
        return

    if enable_visualization:
        if env.use_mujoco:
            # MuJoCo可视化模式
            # 启动环境
            if not env.start():
                logger.error("无法启动仿真环境")
                return

            # 启动MuJoCo查看器
            env.mujoco_integration.launch_viewer(None, env.camera_fps)

            # 停止环境
            env.stop()

            # 保存场景参数
            env.save_scene_config()
        else:
            # Matplotlib可视化模式
            visualizer = SimVisualizer(env)
            visualizer.initialize()

            # 启动环境
            if not env.start():
                logger.error("无法启动仿真环境")
                return

            # 启动可视化
            visualizer.start()

            # 停止环境
            env.stop()

            # 保存场景参数
            env.save_scene_config()
    else:
        # 非可视化模式，运行稳定性测试
        success = env.run_stability_test()

        if success:
            logger.info("稳定性测试成功完成，环境运行稳定")
        else:
            logger.error("稳定性测试失败")

        # 保存场景参数
        env.save_scene_config()

    logger.info("程序结束")


if __name__ == "__main__":
    main()
