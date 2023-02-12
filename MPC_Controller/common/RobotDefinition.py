from enum import Enum, auto
import numpy as np
from MPC_Controller.utils import DTYPE

# Data structure containing parameters for quadruped robot
class RobotType(Enum): # python 中的枚举类型，用一个类继承Enum
    MINI_CHEETAH = auto()
    LITTLE_DOG = auto()
    QINGZHUI = auto()

class Robot:

    def __init__(self, robotype:RobotType):

        if robotype is RobotType.LITTLE_DOG:
            self._abadLinkLength = 0.0802
            self._hipLinkLength = 0.25
            self._kneeLinkLength = 0.25
            self._bodyMass = 22.2
            # self._bodyInertia = np.array([0.078016, 0, 0,
			# 	 0, 0.42789, 0,
			# 	 0, 0, 0.48331])
            self._bodyInertia = np.array([2.8097, 0, 0,
				 0, 2.5396, 0,
				 0, 0, 0.5977])            
            self._bodyHeight = 0.4

            self._kneeLinkY_offset = 0.0 
            self._abadLocation = np.array([
                0.33, -0.05, 0.0,   # rf
                0.0, -0.19025, 0.0, # rm
                -0.33, -0.05, 0.0,  # rb
                -0.33, 0.05, 0.0,   # lb
                0.0, 0.19025, 0.0,  # lm
                0.33, 0.05, 0.0     # lf
            ], dtype=DTYPE).reshape((6,3))
            self._legNum = 6
            self._bodyName = "trunk"

            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = [1.0, 1.5, 0.0,
                                 0.0, 0.0, 5,
                                 0.0, 0.0, 0.1,
                                 1.0, 1.0, 0.1,
                                 0.0]


        elif robotype is RobotType.MINI_CHEETAH:
            self._abadLinkLength = 0.062
            self._hipLinkLength = 0.209
            self._kneeLinkLength = 0.195
            self._kneeLinkY_offset = 0.004
            self._abadLocation = np.array([
                0.19, -0.049, 0.0,
                -0.19, -0.049, 0.0,
                -0.19, 0.049, 0.0,
                0.19, 0.049, 0.0
            ], dtype=DTYPE).reshape((4,3))
            self._bodyName = "body"
            self._bodyMass = 3.3 * 3
            self._bodyInertia = np.array([0.011253, 0, 0, 
                                      0, 0.036203, 0, 
                                      0, 0, 0.042673]) * 10
            self._bodyHeight = 0.29
            self._legNum = 4
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = [0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1, 0]
        
        else:
            raise Exception("Invalid RobotType")
            
        self._robotType = robotype

    def getHipLocation(self, leg:int):
        """
        Get location of the hip for the given leg in robot base frame
        leg index: rf, rm, rb, lb, lm, lf
        """
        assert leg >= 0 and leg < self._legNum
        pHip = self._abadLocation[leg,:].reshape((3,1))
        return pHip