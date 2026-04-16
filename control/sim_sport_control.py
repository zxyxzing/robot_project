from simple_pi_controller import SimplePIController
import numpy as np

class Move:
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0

    def move(self, vx, vy):
        self.vx = vx
        self.vy = vy
        print(f"Move -> vx={vx:.4f}, vy={vy:.4f}")

class SportControl:
    def __init__(self):
        self.pi = SimplePIController()
        self.move = Move()
        self.dt = 0.01

    def control(self, human_pos, robot_pos):
        dist = np.linalg.norm(np.array(human_pos) - np.array(robot_pos))
        dist_error = dist - self.pi.d_target
        x_error = human_pos[1] - robot_pos[1]

        vx, vy = self.pi.calculate_speed(dist_error, x_error, self.dt)
        self.move.move(vx, vy)
        return vx, vy

if __name__ == "__main__":
    ctrl = SportControl()
    ctrl.control([1.0, 0.2], [0.0, 0.0])

