from simple_pi_controller import SimplePIController
import numpy as np

class Move:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0

    def move(self, vx, vy):
        """发送速度指令到机器人"""
        self.vx = vx
        self.vy = vy
        print(f"[Move] 发送速度: vx={vx:.4f}m/s, vy={vy:.4f}m/s")

class SportControl:
    def __init__(self):
        # 初始化PI控制器
        self.pi_controller = SimplePIController()
        # 初始化运动控制接口
        self.move_interface = Move()
        self.dt = 0.01

    def control_step(self, human_info):
        """
        主控制接口：接收human_info，输出速度指令
        :param human_info: 视觉模块输出的字典
        :return: vx, vy
        """
        # 调用PI控制器计算速度
        vx, vy = self.pi_controller.calculate_from_human_info(human_info, self.dt)
        # 发送速度指令
        self.move_interface.move(vx, vy)
        return vx, vy

if __name__ == "__main__":
    # 测试：模拟视觉输出的human_info
    test_human_info = {
        'norm_x': 0.02,
        'norm_y': -0.01,
        'norm_size': 0.15,
        'center_x': 960.5,
        'center_y': 540.3,
        'width': 200,
        'height': 400,
        'confidence': 0.87,
        'timestamp': 1234567
    }

    controller = SportControl()
    vx, vy = controller.control_step(test_human_info)
    print(f"[Test] 输入norm_x={test_human_info['norm_x']}, norm_y={test_human_info['norm_y']}")
    print(f"[Test] 输出vx={vx:.4f}, vy={vy:.4f}")
