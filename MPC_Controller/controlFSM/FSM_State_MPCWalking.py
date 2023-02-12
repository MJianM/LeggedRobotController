import numpy as np
from MPC_Controller.utils import DTYPE
from MPC_Controller.controlFSM.FSM_State import FSM_State
from MPC_Controller.Parameters import FSM_StateName
from MPC_Controller.controlFSM.ControlFSMData import ControlFSMData
from MPC_Controller.convex_MPC.MPCController import ConvexMPCLocomotion
from MPC_Controller.Parameters import Parameters

class FSM_State_MPCWalking(FSM_State):
    def __init__(self, _controlFSMData: ControlFSMData):
        super().__init__(_controlFSMData, FSM_StateName.MPCWalking, "MPCWalking")

        self.controller = ConvexMPCLocomotion(0.005, 10) # 200Hz底层控制器， 20Hz MPC, 10步Horizon
        self.controller.initialize(self._data)

    def onEnter(self):
        pass

    def run(self):
        self.controller.run(self._data)

    def onExit(self):
        if Parameters.cmpc_enable_log:
            if not self.controller.logger.is_empty():
                self.controller.logger.flush_logging()
        print("MPC end.")

    def checkTransition(self):
        return self.stateName
    
    def transition(self):
        return True


        