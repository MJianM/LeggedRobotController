import numpy as np

import mujoco
import mujoco_viewer

from MPC_Controller.common.RobotDefinition import RobotType, Robot
from MPC_Controller.common.LegController import LegController
from MPC_Controller.StateEstimator import StateEstimator
from MPC_Controller.DesiredStateCommand import DesiredStateCommand
from MPC_Controller.Parameters import Parameters
from MPC_Controller.controlFSM.controlFSM import ControlFSM
from MPC_Controller.convex_MPC.MPCController import ConvexMPCLocomotion
from MPC_Controller.utils import DTYPE

class SimData():
    def __init__(self, legNum) -> None:
        self.states = dict()
        self.legNum = legNum
        self.states["qPos"] = np.zeros(self.legNum*3, dtype=DTYPE)
        self.states["qVel"] = np.zeros(self.legNum*3, dtype=DTYPE)
        self.states["bAngVel"] = np.zeros(3, dtype=DTYPE)
        self.states["bLinVel"] = np.zeros(3, dtype=DTYPE)
        self.states["bQuat"] = np.zeros(4, dtype=DTYPE)
        self.states["trq"] = np.zeros(self.legNum*3, dtype=DTYPE)

    def fromMujoco(self, simData:mujoco.MjData):
        self.states["qPos"] = simData.qpos[7:]
        self.states["qVel"] = simData.qvel[6:]
        self.states["bAngVel"] = simData.qvel[3:6]
        self.states["bLinVel"] = simData.qvel[0:3]
        self.states["bQuat"] = simData.qpos[3:7]
    def toMujoco(self, simData:mujoco.MjData):
        # tmp = self.states['trq']/200
        # print(type(tmp))
        # print(tmp)
        simData.ctrl[0:] = self.states['trq']/200


if __name__ == '__main__':

    # mujoco simulation data
    model = mujoco.MjModel.from_xml_path("./model/littledog.xml")
    print("-----------------------------------------------------------")
    print("[SIMULATION INFO] Printing....")
    print("timestep:",model.opt.timestep)

    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # robot
    robot = Robot(RobotType.LITTLE_DOG)
    # leg controller
    legController = LegController(robot)
    # state estimator
    stateEstimator = StateEstimator(robot) 
    # desired command
    desiredCommand = DesiredStateCommand()
    desiredCommand.reset()

    # FSM
    controlFSM = ControlFSM(robot, stateEstimator, legController, desiredCommand)

    # bridge between simulator and controller
    dataBridge = SimData(robot._legNum)


    # simulate and render
    for _ in range(10000):
        if viewer.is_alive:

            # update states
            dataBridge.fromMujoco(data)
            legController.updateData(dataBridge.states)
            legController.zeroCommand()
            stateEstimator.update(dataBridge.states)

            # FSM control
            controlFSM.runFSM()

            # send torque command to simulator
            dataBridge.states['trq'] = legController.updateCommand()
            dataBridge.toMujoco(data)

            # print('---------------------------------------------------------')
            # print("[MUJOCO INFO] Printing....")
            # print("ctrl:",data.ctrl)
            # print("actuator_force:",data.actuator_force)

            # simulation
            mujoco.mj_step(model, data)
            viewer.render()
        else:
            break

    # close
    viewer.close()