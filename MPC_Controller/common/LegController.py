import math
import numpy as np
from math import sin, cos
from MPC_Controller.Parameters import Parameters
from MPC_Controller.common.RobotDefinition import Robot
from MPC_Controller.utils import DTYPE

class LegControllerCommand:
    def __init__(self):
        self.tauFeedForward = np.zeros((3,1), dtype=DTYPE)
        self.forceFeedForward = np.zeros((3,1), dtype=DTYPE)

        self.qDes = np.zeros((3,1), dtype=DTYPE)
        self.qdDes = np.zeros((3,1), dtype=DTYPE)
        self.pDes = np.zeros((3,1), dtype=DTYPE)
        self.vDes = np.zeros((3,1), dtype=DTYPE)

        self.kpCartesian = np.zeros((3,3), dtype=DTYPE)
        self.kdCartesian = np.zeros((3,3), dtype=DTYPE)
        self.kpJoint = np.zeros((3,3), dtype=DTYPE)
        self.kdJoint = np.zeros((3,3), dtype=DTYPE)

    def zero(self):
        """
        Zero the leg command so the leg will not output torque
        """
        self.tauFeedForward.fill(0)
        self.forceFeedForward.fill(0)

        self.qDes.fill(0)
        self.qdDes.fill(0)
        self.pDes.fill(0)
        self.vDes.fill(0)

        self.kpCartesian.fill(0)
        self.kdCartesian.fill(0)
        self.kpJoint.fill(0)
        self.kdJoint.fill(0)

class LegControllerData:
    def __init__(self):
        self.q = np.zeros((3,1), dtype=DTYPE)
        self.qd = np.zeros((3,1), dtype=DTYPE)

        self.p = np.zeros((3,1), dtype=DTYPE)
        self.v = np.zeros((3,1), dtype=DTYPE)

        self.J = np.zeros((3,3), dtype=DTYPE)
        # self.tauEstimate = np.zeros((3,1), dtype=DTYPE)

    def zero(self):

        self.q.fill(0)
        self.qd.fill(0)

        self.p.fill(0)
        self.v.fill(0)

        self.J.fill(0)
        # self.tauEstimate.fill(0)

    def setRobot(self, rob:Robot):
        self.robot = rob

class LegController:

    def __init__(self, rob:Robot):
        self.commands = [LegControllerCommand() for _ in range(rob._legNum)]
        self.datas = [LegControllerData() for _ in range(rob._legNum)]
        # self._legsEnabled = False
        self._maxTorque = 0.0
        self._robot = rob
        for data in self.datas:
            data.setRobot(self._robot)

    def zeroCommand(self):
        """
        Zero all leg commands.  This should be run *before* any control code, so if
        the control code is confused and doesn't change the leg command, the legs
        won't remember the last command.
        """
        for cmd in self.commands:
            cmd.zero()

    def setMaxTorque(self, tau:float):
        self._maxTorque = tau     

    def updateData(self, dof_states):
        """
        update leg data from simulator
        """
        # ! update q, qd, J, p and v here
        for leg in range(self._robot._legNum):
            # q and qd
            self.datas[leg].q[:, 0] = dof_states["qPos"][3*leg:3*leg+3]
            self.datas[leg].qd[:, 0] = dof_states["qVel"][3*leg:3*leg+3]
            # J and p
            self.computeLegJacobianAndPosition(leg)
            # v
            self.datas[leg].v = self.datas[leg].J @ self.datas[leg].qd

    def updateCommand(self): 
        """
        update leg commands for simulator
        """
        # ! update joint PD gain, leg enable, feedforward torque and estimate torque
        legTorques = np.zeros(3*self._robot._legNum, dtype=DTYPE)

        for leg in range(self._robot._legNum):
            # MPC -> f_ff -R^T-> forceFeedForward
            # force feedforward + cartesian PD
            footForce = self.commands[leg].forceFeedForward \
                        + self.commands[leg].kpCartesian @ (self.commands[leg].pDes - self.datas[leg].p) \
                        + self.commands[leg].kdCartesian @ (self.commands[leg].vDes - self.datas[leg].v)

            # tau feedforward + torque
            legTorque = self.commands[leg].tauFeedForward + self.datas[leg].J.T @ footForce

            # joint PD control
            legTorque += self.commands[leg].kpJoint @ (self.commands[leg].qDes - self.datas[leg].q)
            legTorque += self.commands[leg].kdJoint @ (self.commands[leg].qdDes - self.datas[leg].qd)

            legTorques[leg*3:(leg+1)*3] = legTorque.flatten()
        
        # print("leg 0 effort %.3f %.3f %.3f"%(legTorques[0], legTorques[1], legTorques[2]))
        return legTorques

    def computeLegJacobianAndPosition(self, leg:int):
        """
        return J and p
        """
        a = self._robot._hipLinkLength
        b = self._robot._kneeLinkLength
        t0 = self.datas[leg].q[0]
        t1 = self.datas[leg].q[1]
        t2 = self.datas[leg].q[2]

        s0 = sin(t0)
        s1 = sin(t1)
        s2 = sin(t2)
        c0 = cos(t0)
        c1 = cos(t1)
        c2 = cos(t2)
        s12 = sin(t1+t2)
        c12 = cos(t1+t2)

        x_tip_sag = - a * s1 - b * s12
        z_tip_sag = - a * c1 - b * c12

        self.datas[leg].p[0] = x_tip_sag
        self.datas[leg].p[1] = - z_tip_sag * s0
        self.datas[leg].p[2] = z_tip_sag * c0

        self.datas[leg].J[0,0] = 0.0
        self.datas[leg].J[0,1] = - a * c1 - b * c12
        self.datas[leg].J[0,2] = - b * c12
        self.datas[leg].J[1,0] = c0 * (a * c1 + b * c12)
        self.datas[leg].J[1,1] = s0 * (- a * s1 - b * s12)
        self.datas[leg].J[1,2] = s0 * (- b * s12)
        self.datas[leg].J[2,0] = s0 * (a * c1 + b * c12)
        self.datas[leg].J[2,1] = c0 * (a * s1 + b * s12)
        self.datas[leg].J[2,2] = c0 * (b * s12)


    def inverseKinematics(self, pDes, bendInfo:bool):
        '''
            pDes: ndarray (3,1) in the hip frame
            benInfo:bool 
            return: ndarray (3,1)
        '''
        # t1 > 0 , t2 < 0 for bend in. bend_info=True
        # t1 < 0 , t2 > 0 for bend out. bend_info=False
        x = pDes[0,0]
        y = pDes[1,0]
        z = pDes[2,0]
        len_thigh = 0.25
        len_calf = 0.25

        theta0 = math.atan(-y/z)

        z_tip_sag = math.sqrt(z*z+y*y)
        cos_shank = (z_tip_sag**2 + x**2 - len_thigh**2 - len_calf**2)/(2*len_thigh*len_calf)
        if cos_shank>1:
            cos_shank = 1
        if cos_shank<-1:
            cos_shank = -1

        if bendInfo == True:
            theta2 = - math.acos(cos_shank)
        else:
            theta2 = math.acos(cos_shank)

        cos_beta = (z_tip_sag**2 + x**2 + len_thigh**2 - len_calf**2)/(2*len_thigh*math.sqrt(z_tip_sag**2 + x**2))
        if cos_beta>1:
            cos_beta = 1
        if cos_beta<-1:
            cos_beta = -1
        beta = math.acos(cos_beta)

        alpha = math.atan(x/z_tip_sag)
        if x>0:
            if bendInfo == False:
                theta1 = -alpha - beta
            else:
                theta1 = -alpha + beta
        else:
            if bendInfo == False:
                theta1 = -alpha - beta
            else:
                theta1 = -alpha + beta

        return np.array([theta0, theta1, theta2], dtype=DTYPE).reshape((3,1))

