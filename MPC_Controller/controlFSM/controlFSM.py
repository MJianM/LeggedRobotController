from MPC_Controller.common.RobotDefinition import Robot
from MPC_Controller.common.LegController import LegController
from MPC_Controller.StateEstimator import StateEstimator
from MPC_Controller.DesiredStateCommand import DesiredStateCommand
from MPC_Controller.controlFSM.ControlFSMData import ControlFSMData
from MPC_Controller.Parameters import Parameters,FSM_StateName,FSM_OperatingMode

from MPC_Controller.controlFSM.FSM_State import FSM_State
from MPC_Controller.controlFSM.FSM_State_StandStill import FSM_State_StandStill
from MPC_Controller.controlFSM.FSM_State_Recovery import FSM_State_Recovery
from MPC_Controller.controlFSM.FSM_State_MPCWalking import FSM_State_MPCWalking

class FSMStatesList():
    def __init__(self) -> None:
        self.StandStill = None
        self.MPCWalking = None
        self.Recovery = None
        self.Invalid = None


class ControlFSM:
    def __init__(self,
                 _quadruped:Robot,
                 _stateEstimator:StateEstimator,
                 _legController:LegController,
                 _desiredStateCommand:DesiredStateCommand):
        self.data = ControlFSMData()
        self.data._robot = _quadruped
        self.data._stateEstimator = _stateEstimator
        self.data._legController = _legController
        self.data._desiredStateCommand = _desiredStateCommand

        self.statesList = FSMStatesList()
        self.statesList.Invalid = None
        self.statesList.StandStill = FSM_State_StandStill(self.data)
        self.statesList.MPCWalking = FSM_State_MPCWalking(self.data)
        self.statesList.Recovery = FSM_State_Recovery(self.data)


        # FSM state information
        self.currentState:FSM_State = None
        self.nextState:FSM_State = None
        self.nextStateName:FSM_StateName = None
        
        self.transitionDone = False
        self.printIter = 0
        # self.printNum = int(1000/(Parameters.controller_dt*100)) # N*(0.01s) in simulation time
        self.printNum = 1
        self.iter = 0

        # ! may need a SafetyChecker
        # self.safetyChecker = 

        self.initialize()

    def initialize(self):
        # Initialize a new FSM State with the control data
        if Parameters.control_mode is FSM_StateName.StandStill:
            self.currentState = self.statesList.StandStill
        elif Parameters.control_mode is FSM_StateName.MPCWalking:
            self.currentState = self.statesList.MPCWalking
        elif Parameters.control_mode is FSM_StateName.Recovery:
            self.currentState = self.statesList.Recovery
        else:
            raise Exception("Invalid initial FSM state!")
        
            
        # Enter the new current state cleanly
        self.currentState.onEnter()
        # Initialize to not be in transition
        self.nextState = self.currentState
        # Initialize FSM mode to normal operation
        self.operatingMode = Parameters.operating_mode

    def runFSM(self):
        # Check the robot state for safe operation
        # operatingMode = safetyPreCheck();

        if self.operatingMode is FSM_OperatingMode.TEST:
            self.currentState.run()

        # Run normal controls if no transition is detected
        elif self.operatingMode is FSM_OperatingMode.NORMAL:
            # Check the current state for any transition
            nextStateName = self.currentState.checkTransition()

            # Detect a commanded transition
            if nextStateName != self.currentState.stateName:
                # Set the FSM operating mode to transitioning
                self.operatingMode = FSM_OperatingMode.TRANSITION
                # Get the next FSM State by name
                self.nextState = self.getNextState(nextStateName)
                # Print transition initialized info
                self.printInfo(1)
            else:
                # Run the iteration for the current state normally
                self.currentState.run()

        # Run the transition code while transition is occuring
        elif self.operatingMode is FSM_OperatingMode.TRANSITION:
            self.transitionDone = self.currentState.transition()

            # TODO Check the robot state for safe operation
            # safetyPostCheck()

            # Run the state transition
            if self.transitionDone:
                # Exit the current state cleanly
                self.currentState.onExit()

                # Print finalizing transition info
                self.printInfo(2)

                # Complete the transition
                self.currentState = self.nextState

                # Enter the new current state cleanly
                self.currentState.onEnter()

                # Return the FSM to normal operation mode
                self.operatingMode = FSM_OperatingMode.NORMAL
        else:
            raise NotImplementedError
            # TODO Check the robot state for safe operation
            # safetyPostCheck()
        

        # TODO if ESTOP
        # self.currentState = self.statesList.passive
        # self.currentState.onEnter()
        # nextStateName = self.currentState.stateName

        # Print the current state of the FSM
        self.printInfo(0) 
        self.iter += 1
        self.data._iterationCounter += 1


    def getNextState(self, stateName:FSM_StateName):
        """
        * Returns the approptiate next FSM State when commanded.
        *
        * @param  next commanded enumerated state name
        * @return next FSM state
        """
        if stateName is FSM_StateName.Invalid:
            return self.statesList.Invalid
        elif stateName is FSM_StateName.MPCWalking:
            return self.statesList.MPCWalking
        elif stateName is FSM_StateName.StandStill:
            return self.statesList.StandStill
        elif stateName is FSM_StateName.Recovery:
            return self.statesList.Recovery
        else:
            return self.statesList.Invalid


    def printInfo(self, opt:int):
        """
        * Prints Control FSM info at regular intervals and on important events
        * such as transition initializations and finalizations. Separate function
        * to not clutter the actual code.
        *
        * @param printing mode option for regular or an event
        """
        if not Parameters.FSM_print_info:
            return
            
        if opt == 0:
            self.printIter += 1
            if self.printIter == self.printNum:
                print("---------------------------------------------------------")
                print("[CONTROL FSM] Printing FSM Info...")
                print("Iteration: " + str(self.iter))
                if self.operatingMode is FSM_OperatingMode.NORMAL:
                    print("Operating Mode:: NORMAL in "+self.currentState.stateString)
                elif self.operatingMode is FSM_OperatingMode.TRANSITION:
                    print("Operating Mode: TRANSITIONING from "+self.currentState.stateString+" to "+self.nextState.stateString)

                self.printIter = 0
        
        # Initializing FSM State transition
        elif opt == 1:
            print("[CONTROL FSM] Transition initialized from "+
                  self.currentState.stateString + " to " +
                  self.nextState.stateString)

        # Finalizing FSM State transition
        elif opt == 2:
            print("[CONTROL FSM] Transition finalizing from "+
                  self.currentState.stateString + " to " +
                  self.nextState.stateString)