# import math
import time
import sys
import copy
import numpy as np
# import MPC_Controller.convex_MPC.mpc_osqp as mpc
from MPC_Controller.common.RobotDefinition import RobotType
from MPC_Controller.Parameters import Parameters
from MPC_Controller.convex_MPC.GaitScheduler import GaitScheduler
from MPC_Controller.DesiredStateCommand import DesiredStateCommand
from MPC_Controller.controlFSM.ControlFSMData import ControlFSMData
from MPC_Controller.common.FootSwingTrajectory import FootSwingTrajectory
from MPC_Controller.utils import DTYPE
from MPC_Controller.math_utils.orientation_tools import coordinateRotation, CoordinateAxis
from MPC_Controller.Logger import Logger
try:
    from MPC_Controller.convex_MPC import mpc_osqp as mpc
except:
    print("mpc module can not be imported.")
    sys.exit()


class ConvexMPCLocomotion:
    def __init__(self, _dt:float, _iterationsBetweenMPC:int):
        self.iterationsBetweenMPC = int(_iterationsBetweenMPC)
        self.horizonLength = 10 # a fixed number for all mpc gait
        self.dt = _dt  # simulation time per iteration
        self.dtMPC = self.dt * self.iterationsBetweenMPC

        self.gait = GaitScheduler(self.horizonLength,
                            np.array([0, 5, 0, 5, 0, 5], dtype=DTYPE),
                            np.array([5, 5, 5, 5, 5, 5], dtype=DTYPE), "33-Walking"
                            )

        # self.default_iterations_between_mpc = self.iterationsBetweenMPC
        print("[Convex MPC] dt: %.3f iterations: %d, dtMPC: %.3f" % (self.dt, self.iterationsBetweenMPC, self.dtMPC))
        
        self.firstRun = True
        self.iterationCounter = 0

        self.legNum:int = None
        

        self.f_ff:np.ndarray = None
        self.foot_positions:np.ndarray = None

        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0

        self.footSwingTrajectories:list = None

        self.Kp:np.ndarray = None
        self.Kp_stance:np.ndarray = None
        self.Kd:np.ndarray = None
        self.Kd_stance:np.ndarray = None

        self.logger = Logger("logs/")

    def initialize(self, data:ControlFSMData):
        if Parameters.cmpc_alpha > 1e-4:
            print("Alpha was set too high (" + str(Parameters.cmpc_alpha) + ") adjust to 1e-5\n")
            Parameters.cmpc_alpha = 1e-5

        if Parameters.cmpc_enable_log:
            # flush last log
            if not self.logger.is_empty():
                self.logger.flush_logging()
            # start new logs
            self.logger.start_logging()

        self.iterationCounter = 0
        self.legNum = data._robot._legNum
        self.f_ff = np.zeros((self.legNum,3,1), dtype=DTYPE)
        self.foot_positions = np.zeros((self.legNum,3,1), dtype=DTYPE)
        self.footSwingTrajectories = [FootSwingTrajectory() for _ in range(self.legNum)]

        self._cpp_mpc = mpc.ConvexMpc(data._robot._bodyMass,
                                    list(data._robot._bodyInertia),
                                    data._robot._legNum,
                                    self.horizonLength,
                                    self.dtMPC,
                                    Parameters.cmpc_alpha,
                                    mpc.OSQP)

        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0
        self.firstRun = True

    def solveMPC(self, data:ControlFSMData):
        '''
            
            solve MPC in yaw aligned world frame
        '''
        timer = time.time()
        
        mpc_weight = data._robot._mpc_weights
        com_position = list(data._stateEstimator.result.position.flatten()) # ground frame
        com_velocity = data._stateEstimator.world_R_yaw_frame.T @ data._stateEstimator.result.vWorld
        # com_velocity = data._stateEstimator.ground_R_body_frame @ data._stateEstimator.result.vBody
        com_velocity = list(com_velocity.flatten())
        gravity_projection_vec = [0,0,1]
        # attention :  ZYX format in google MPC solver / XYZ format in normal 
        # so we should transfer it.
        com_roll_pitch_yaw = data._stateEstimator.result.rpy_yaw.flatten()
        # com_roll_pitch_yaw = list(com_roll_pitch_yaw[::-1])

        com_angular_velocity = data._stateEstimator.world_R_yaw_frame.T @ data._stateEstimator.result.omegaWorld
        # com_angular_velocity = data._stateEstimator.ground_R_body_frame @ data._stateEstimator.result.omegaBody
        com_angular_velocity = list(com_angular_velocity.flatten())

        mpcTable = self.gait.getMpcTable()

        footPositionsBaseFrame = list(self.foot_positions.flatten())

        desired_com_position = [0.,0., data._robot._bodyHeight]
        desired_com_velocity = [self._x_vel_des, self._y_vel_des, 0.]
        desired_com_rpy = [0.,0.,0.]
        desired_com_angular_velocity = [0.,0.,self._yaw_turn_rate]


        print("-----------------------------------------------------------")
        print("[MPC INFO] Printing MPC Info....")
        print("ITER:", self.iterationCounter)
        print("weight:",mpc_weight)
        print("com_position:",com_position)
        print("com_vel:",com_velocity)
        print("com_rpy:",com_roll_pitch_yaw)
        print("gravity_vec:",gravity_projection_vec)
        print("com_angvel:",com_angular_velocity)
        print("mpcTable:",mpcTable)
        print("footpos:",footPositionsBaseFrame)        

        predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
            mpc_weight, # mpc weights list(12,)
            com_position, # com_position (set x y to 0.0)
            com_velocity, # com_velocity
            com_roll_pitch_yaw, # com_roll_pitch_yaw (set yaw to 0.0)
            gravity_projection_vec,  # Normal Vector of ground
            com_angular_velocity, # com_angular_velocity
            mpcTable,  # Foot contact states
            footPositionsBaseFrame,  # foot_positions_base_frame
            data._robot._friction_coeffs,  # foot_friction_coeffs
            desired_com_position,  # desired_com_position
            desired_com_velocity,  # desired_com_velocity
            desired_com_rpy,  # desired_com_roll_pitch_yaw
            desired_com_angular_velocity  # desired_com_angular_velocity
        )

        total_force_body:np.ndarray = np.zeros((3,1),dtype=DTYPE)
        total_torque_body = np.zeros((3,1),dtype=DTYPE)

        body_R_yaw = data._stateEstimator.result.rBody.T @ data._stateEstimator.world_R_yaw_frame
        for leg in range(data._robot._legNum):
            self.f_ff[leg] = body_R_yaw @ \
                np.array(predicted_contact_forces[leg*3:leg*3+3],dtype=DTYPE).reshape((3,1))
            total_force_body -= self.f_ff[leg]
            total_torque_body += np.cross(self.foot_positions[leg].flatten(), -self.f_ff[leg].flatten()).reshape((3,1))


        print("-----------------------------------------------------------")
        print("[MPC INFO] Printing MPC Info....")
        print("ITER:", self.iterationCounter)
        print("FRC:",self.f_ff.flatten())

        print("TOTAL_F:",(data._stateEstimator.result.rBody @ total_force_body).flatten())
        print("TOTAL_T:",(data._stateEstimator.result.rBody @ total_torque_body).flatten())

        print("RPY:", data._stateEstimator.result.rpy.flatten())
        # print("QUAT:",data._stateEstimator.result.orientation.w,
        #             data._stateEstimator.result.orientation.x,
        #             data._stateEstimator.result.orientation.y,
        #             data._stateEstimator.result.orientation.z)
        print("GAIT:",self.gait.contactFlag.flatten())

        if Parameters.cmpc_print_solver_time:
            print("MPC Update Time %.3f s\n"%(time.time()-timer))     

        log_data_frame = dict(
            COM_RPY = com_roll_pitch_yaw, # COM_RPY
            COM_POS = com_position, # COM_POS
            COM_ANG = com_angular_velocity, # COM_ANG
            COM_VEL = com_velocity, # COM_VEL
            DES_RPY = desired_com_rpy, # DES_RPY
            DES_POS = desired_com_position, # DES_POS
            DES_ANG = desired_com_angular_velocity, # DES_ANG
            DES_VEL = desired_com_velocity, # DES_VEL
            MPC_GRF = predicted_contact_forces[:data._robot._legNum*3], # MPC_GRF
            # MPC_LOS = mpc_state_loss+mpc_torque_loss, # MPC_LOS
            MPC_WEI = mpc_weight, # MPC_WEI
            TIM_STA = self.iterationCounter # TIM_STA
        )
        if Parameters.cmpc_enable_log:
            self.logger.update_logging(log_data_frame)

    def recomputerTiming(self, iterations_per_mpc:int):
        self.iterationsBetweenMPC = iterations_per_mpc
        self.dtMPC = self.dt*iterations_per_mpc

    # before running, we should update stateEstimator, legData
    def run(self, data:ControlFSMData):

        # desired command in ground frame
        self._x_vel_des = data._desiredStateCommand.x_vel_cmd
        self._y_vel_des = data._desiredStateCommand.y_vel_cmd
        self._yaw_turn_rate = data._desiredStateCommand.yaw_turn_rate

        # update foot position in base frame
        for leg in range(self.legNum):
            self.foot_positions[leg] = data._robot.getHipLocation(leg) + data._legController.datas[leg].p

        # * first time initialization
        if self.firstRun:
            self.firstRun = False
            data._stateEstimator._init_contact_history(self.foot_positions)
            for i in range(self.legNum):
                self.footSwingTrajectories[i].setHeight(0.05)
                self.footSwingTrajectories[i].setInitialPosition(self.foot_positions[i])
                self.footSwingTrajectories[i].setFinalPosition(self.foot_positions[i])

        if Parameters.flat_ground:
            data._stateEstimator._update_com_position_ground_frame(self.foot_positions)
            data._stateEstimator._update_contact_history(self.foot_positions)
        else:
            data._stateEstimator._compute_ground_normal_and_com_position(self.foot_positions)

        # update gait status
        swingBeginFlag = self.gait.update(self.iterationCounter, self.iterationsBetweenMPC)
        data._stateEstimator.setContactPhase(self.gait.contactFlag)

        # solve MPC
        if self.iterationCounter % self.iterationsBetweenMPC == 0:
            self.solveMPC(data)

        # counter
        self.iterationCounter += 1

        # swing trajectory planning
        for leg in range(self.legNum):
            if swingBeginFlag[leg]:
                # leg begin swinging
                self.footSwingTrajectories[leg].setInitialPosition(copy.copy(self.foot_positions[leg]))
                # foothold heuristic planning
                stanceTime = self.gait.getStanceTime(leg)*self.dtMPC
                hipPos = data._robot.getHipLocation(leg)
                hipPos = coordinateRotation(CoordinateAxis.Z, self._yaw_turn_rate * stanceTime / 2) @ hipPos
                footHold = hipPos - data._stateEstimator.result.position

                vx = data._stateEstimator.result.vBody[0,0]
                vy = data._stateEstimator.result.vBody[1,0]
                height = data._stateEstimator.result.position[2,0]
                xRel = vx * stanceTime * (0.5 + Parameters.cmpc_bonus_swing) \
                    + 0.03 * (vx - self._x_vel_des) \
                    + (0.5 * height / 9.81) * (vy * self._yaw_turn_rate)
                yRel = vy * stanceTime * (0.5 + Parameters.cmpc_bonus_swing) \
                    + 0.03 * (vy - self._y_vel_des) \
                    + (0.5 * height / 9.81) * (vx * self._yaw_turn_rate)
                maxRel = 0.3
                xRel = min(max(xRel, -maxRel), maxRel)
                yRel = min(max(yRel, -maxRel), maxRel)
                footHold[0,0] += xRel
                footHold[1,0] += yRel

                self.footSwingTrajectories[leg].setFinalPosition(footHold)
                self.footSwingTrajectories[leg].setHeight(height/3)

        self.Kp = np.array([50, 0, 0, 0, 50, 0, 0, 0, 30], dtype=DTYPE).reshape((3,3))
        self.Kp_stance = np.zeros_like(self.Kp)

        self.Kd = np.array([7, 0, 0, 0, 7, 0, 0, 0, 7], dtype=DTYPE).reshape((3,3))
        self.Kd_stance = self.Kd

        for leg in range(self.legNum):
            if self.gait.swingState[leg]:
                # swing leg
                assert self.gait.stanceState[leg] == 0
                swingTime = (self.gait.nSegment - self.gait.durations[leg]) * self.dtMPC
                self.footSwingTrajectories[leg].computeSwingTrajectoryBezier(self.gait.swingState[leg],swingTime)
                pDes = self.footSwingTrajectories[leg].getPosition()
                vDes = self.footSwingTrajectories[leg].getVelocity()

                pDes = pDes - data._robot.getHipLocation(leg)
                if pDes[2,0] > 0.0:
                    a = pDes
                qDes = data._legController.inverseKinematics(pDes, True)

                print("-----------------------------------------------------")
                print("[LEG CONTROL] Printing ...")
                print("leg:",leg)
                print('pDes:',pDes.flatten())
                print("qDes:", qDes.flatten())
                print("q:", data._legController.datas[leg].q.flatten())

                np.copyto(data._legController.commands[leg].qDes, qDes, casting="same_kind")
                np.copyto(data._legController.commands[leg].vDes, vDes, casting="same_kind")
                np.copyto(data._legController.commands[leg].kpJoint, self.Kp, casting="same_kind")
                np.copyto(data._legController.commands[leg].kdCartesian, self.Kd, casting="same_kind")

            else:
                # stance leg
                assert self.gait.swingState[leg] == 0
                np.copyto(data._legController.commands[leg].forceFeedForward, self.f_ff[leg], casting="same_kind")
                # np.copyto(data._legController.commands[leg].kdJoint, np.identity(3, dtype=DTYPE)*0.2, casting=CASTING)

        # print("-----------------------------------------------------------")
        # print("[MPC INFO] Printing MPC Info....")
        # print("ITER:", self.iterationCounter)
        # print("FRC:",self.f_ff.flatten())
        # print("RPY:", data._stateEstimator.result.rpy.flatten())
        # print("QUAT:",data._stateEstimator.result.orientation.w,
        #             data._stateEstimator.result.orientation.x,
        #             data._stateEstimator.result.orientation.y,
        #             data._stateEstimator.result.orientation.z)