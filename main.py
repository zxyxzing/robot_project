import mujoco
import numpy as np
import glfw
from mujoco import viewer

# 创建一个空的 MuJoCo 模型
xml = """
<mujoco>
  <worldbody>
    <geom type="box" size="1 1 1" rgba="0.8 0.2 0.2 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 启动渲染器
with viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()# 在 main.py 最后加一行：
print("程序运行结束，按回车键退出...")
input()
