import numpy as np
from MPC_Controller.utils import DTYPE
from MPC_Controller.controlFSM.FSM_State import FSM_State
from MPC_Controller.Parameters import FSM_StateName
from MPC_Controller.controlFSM.ControlFSMData import ControlFSMData

class FSM_State_StandStill(FSM_State):
    def __init__(self, _controlFSMData: ControlFSMData):
        super().__init__(_controlFSMData, FSM_StateName.StandStill, "StandStill")
        self.nextStateName = FSM_StateName.MPCWalking

    def onEnter(self):
        pass

    def run(self):
        
        legNum = self._data._robot._legNum
        for leg in range(legNum):
            qDes = np.array([0., 0.5, -0.9], dtype=DTYPE).reshape((3,1))
            qVel = np.zeros_like(qDes, dtype=DTYPE)

            self.jointPDControl(leg, qDes, qVel)


    def onExit(self):
        pass

    def checkTransition(self):
        if self._data._iterationCounter > 200:
            return FSM_StateName.MPCWalking
        else:
            return self.stateName

    def transition(self):
        
        return True

