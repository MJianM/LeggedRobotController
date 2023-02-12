# LeggedRobotController
基于convexMPC算法的多足机器人运动控制器，使用Python复现了MIT Chteeah3的控制框架。
搭建了基于Mujoco的多足机器人方仿真环境。

## Requirements
  * OSQP
  * QPOASES
  * pybind11
  * mujoco
  * mujoco-python-viewer
## 使用说明
MPC算法使用C语言实现，采用pybind11搭建python接口。由于不同平台环境存在差异，使用前需要自行编译生成对应库文件。编译方式可参考convex_MPC中的CMakeLists文件。
