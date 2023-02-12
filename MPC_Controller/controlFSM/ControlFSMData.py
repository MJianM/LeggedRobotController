from MPC_Controller.common.RobotDefinition import Robot, RobotType
from MPC_Controller.common.LegController import LegController
# from MPC_Controller.Parameters import Parameters
from MPC_Controller.StateEstimator import StateEstimator
from MPC_Controller.DesiredStateCommand import DesiredStateCommand

class ControlFSMData:
    def __init__(self):
        self._robot:Robot = None
        self._stateEstimator:StateEstimator = None
        self._legController:LegController = None
        self._desiredStateCommand:DesiredStateCommand = None

        self._iterationCounter:int = 0
        # self._gaitScheduler:GaitScheduler = None
        # self.userParameters:Parameters = None


