from enum import Enum, auto

class FSM_StateName(Enum):
    StandStill = auto()
    Walking = auto()
    Invalid = auto()
    Recovery = auto()
    MPCWalking = auto()

class FSM_OperatingMode(Enum):
    TEST = auto()
    NORMAL = auto()
    TRANSITION = auto()


class Parameters:
    cmpc_x_drag = 3.0

    cmpc_bonus_swing = 0.0 # 启发式落脚点选取时参数

    cmpc_alpha = 1e-5 # MPC中的控制能量权重

    cmpc_print_solver_time = False
    cmpc_print_update_time = False
    cmpc_print_states = False
    cmpc_enable_log = False


    # cmpc_py_solver = 1 # 0 cvxopt, 1 osqp
    # cmpc_solver_type = 2 # 0 mit py solver, 1 mit cpp solver, 2 google cpp solver
    # # cmpc_gait = GaitType.TROT # 1 bound, 2 pronk, 3 pace, 4 stand, else trot



    flat_ground = False

    # * [-1, 1] -> [a, b] => [-1, 1] * (b-a)/2 + (b+a)/2
    MPC_param_scale = [4, 4, 4,     # 1-9
                       20, 20, 20,  # 30-70
                       1, 1, 1,     # 0-2
                       1, 1, 1]     # 0-2
    
    MPC_param_const = [5, 5, 5,
                       50,50,50,
                       1, 1, 1,
                       1, 1, 1]

    
    controller_dt = 0.005 # in sec

    locomotionUnsafe = False # global indicator for switching contorl mode
    FSM_check_safety = False




    FSM_print_info = True
    control_mode:FSM_StateName = FSM_StateName.StandStill
    operating_mode:FSM_OperatingMode = FSM_OperatingMode.NORMAL # TEST / NORMAL




    