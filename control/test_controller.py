from motion_controller import MotionController

def test_controller():
    # 1. 初始化控制器
    controller = MotionController()

    # 2. 模拟视觉模块的输出格式
    test_target_info = {
        'norm_x': 0.02,
        'norm_y': -0.01,
        'confidence': 0.87
    }

    # 模拟机器人当前位姿（示例）
    test_current_pose = [0.0, 0.0, 0.0]

    # 3. 调用控制指令方法
    command = controller.compute_command(test_target_info, test_current_pose)

    # 4. 打印输出，验证格式
    print("=== 控制指令输出 ===")
    print(f"线速度: {command['velocity']}")
    print(f"角速度: {command['yaw_rate']}")

    # 5. 测试停止方法
    stop_command = controller.stop()
    print("\n=== 停止指令输出 ===")
    print(f"线速度: {stop_command['velocity']}")
    print(f"角速度: {stop_command['yaw_rate']}")

if __name__ == "__main__":
    test_controller()

