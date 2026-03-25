# Go2仿真环境模块化设计

## 模块结构

本项目将Go2仿真环境按功能拆分为多个模块，每个模块负责特定的功能：

### 1. config.py
配置模块，包含所有配置参数和日志设置：
- 摄像头配置（分辨率、帧率、视场角等）
- 机器人配置（尺寸、初始位置、朝向等）
- 人体目标配置（尺寸、速度、转向角度等）
- 环境配置（重力、地面高度等）
- 测试配置（测试持续时间等）
- MuJoCo配置

### 2. robot.py
机器人类，提供机器人模型和状态管理功能：
- 机器人初始化
- 机器人状态更新
- 机器人位置和朝向设置
- 机器人状态获取

### 3. human_target.py
人体目标类，提供人体目标模型和运动控制功能：
- 人体目标初始化
- 人体目标激活和停用
- 人体目标位置和朝向更新
- 碰撞检测
- 摄像头视野约束

### 4. mujoco_integration.py
MuJoCo集成类，提供MuJoCo仿真平台的集成功能：
- MuJoCo环境初始化
- MuJoCo模型加载和创建
- MuJoCo仿真步执行
- 物体位置更新和获取
- MuJoCo查看器启动

### 5. visualizer.py
可视化类，提供仿真环境的可视化功能：
- 可视化界面初始化
- 摄像头画面显示
- FPS曲线显示
- 机器人位置曲线显示
- 关节角度显示
- 人体和机器人标记显示

### 6. simulation_env.py
仿真环境主类，提供仿真环境的核心功能：
- 仿真环境初始化
- 摄像头初始化
- 物理引擎初始化
- 仿真环境启动和停止
- 仿真步执行
- 人体目标添加
- 场景参数保存
- 稳定性测试

### 7. main.py
主程序，提供命令行接口和程序入口：
- 命令行参数解析
- 仿真环境创建
- 可视化模式选择
- MuJoCo模式选择
- 程序流程控制

## 使用方法

### 1. 基本使用

#### 可视化模式（Matplotlib）
```bash
python main.py --visual
# 或
python main.py -v
```

#### MuJoCo可视化模式
```bash
python main.py --visual --mujoco
# 或
python main.py -v -m
```

#### 使用自定义MuJoCo模型
```bash
python main.py --visual --mujoco --model path/to/your_model.xml
```

#### 非可视化模式（稳定性测试）
```bash
python main.py
```

### 2. 模块导入

您可以在自己的代码中导入这些模块：

```python
from config import CAMERA_RESOLUTION, CAMERA_FPS, logger
from robot import Robot
from human_target import HumanTarget
from mujoco_integration import MuJoCoIntegration
from visualizer import SimVisualizer
from simulation_env import Go2SimEnv

# 创建仿真环境
env = Go2SimEnv(
    camera_resolution=(1920, 1080),
    camera_fps=200,
    test_duration_minutes=30,
    use_mujoco=False
)

# 初始化环境
env.initialize()

# 添加人体目标
env.add_human_target()

# 启动环境
env.start()

# 执行仿真步
state = env.step()

# 停止环境
env.stop()

# 保存场景参数
env.save_scene_config()
```

### 3. 配置修改

您可以直接修改config.py中的配置参数：

```python
# 修改摄像头分辨率
CAMERA_RESOLUTION = (1280, 720)

# 修改摄像头帧率
CAMERA_FPS = 60

# 修改人体速度
HUMAN_SPEED = 0.3

# 修改人体运动范围
HUMAN_MOVEMENT_RANGE = 8.0
```

## 功能特性

1. **模块化设计**：每个模块负责特定功能，代码结构清晰
2. **灵活配置**：所有配置参数集中在config.py中，便于修改
3. **多种可视化**：支持Matplotlib和MuJoCo两种可视化方式
4. **人体目标**：支持人体目标的添加、运动控制和碰撞检测
5. **摄像头视野约束**：人体运动始终在摄像头视野范围内
6. **场景参数保存**：支持将场景参数保存到JSON文件

## 依赖项

- numpy
- matplotlib
- mujoco（可选，用于MuJoCo模式）

## 注意事项

1. 如果要使用MuJoCo模式，需要先安装MuJoCo：
   ```bash
   pip install mujoco
   ```

2. 程序运行时会生成sim_env.log日志文件和env_config.json配置文件

3. 可视化模式下，可以通过鼠标交互来调整视角（MuJoCo模式）

4. 碰撞检测使用AABB（轴对齐包围盒）算法，可能会有一定的误差
