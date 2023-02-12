import numpy as np
from MPC_Controller.utils import DTYPE
from MPC_Controller.controlFSM.FSM_State import FSM_State
from MPC_Controller.Parameters import FSM_StateName
from MPC_Controller.controlFSM.ControlFSMData import ControlFSMData

class FSM_State_Recovery(FSM_State):
    def __init__(self, _controlFSMData: ControlFSMData):
        super().__init__(_controlFSMData, FSM_StateName.Recovery, "Recovery")

    def onEnter(self):
        pass

    def run(self):
        pass

    def onExit(self):
        pass

        