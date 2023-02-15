import numpy as np
import scipy
from MPC_Controller.Parameters import Parameters
from MPC_Controller.common.RobotDefinition import Robot
# from MPC_Controller.math_utils.moving_window_filter import MovingWindowFilter
from MPC_Controller.utils import DTYPE
from MPC_Controller.math_utils.orientation_tools import quat_to_rot, quat_to_rpy,rot_to_rpy, get_rot_from_normals, rpy_to_rot, Quaternion

class StateEstimate:
    def __init__(self):
        self.position = np.zeros((3,1), dtype=DTYPE)
        self.vWorld = np.zeros((3,1), dtype=DTYPE)
        self.omegaWorld = np.zeros((3,1), dtype=DTYPE)
        self.orientation = Quaternion(1, 0, 0, 0)

        self.rBody = np.zeros((3,3), dtype=DTYPE) # rotation of body in the world frame
        self.rpy = np.zeros((3,1), dtype=DTYPE)  # rpy of body in the world frame
        self.rpy_yaw = np.zeros((3,1),dtype=DTYPE) # rpy of body in the yaw aligned world frame
        self.rpy_ground = np.zeros((3,1), dtype=DTYPE) # rpy of body in yaw aligned ground frame

        # ground normal in the world frame
        self.ground_normal_world = np.array([0,0,1], dtype=DTYPE) 
        # ground normal in the yaw frame
        self.ground_normal_yaw = np.array([0,0,1], dtype=DTYPE)

        self.vBody = np.zeros((3,1), dtype=DTYPE)
        self.omegaBody = np.zeros((3,1), dtype=DTYPE)

class StateEstimator:

    def __init__(self, rob:Robot):
        self.result = StateEstimate()
        self._robot = rob
        # self._ground_normal_filter = MovingWindowFilter(window_size=10)
        # self._velocity_filter = MovingWindowFilter(window_size=60)
        self._phase = np.zeros((self._robot._legNum,1), dtype=DTYPE)
        self._contactPhase = self._phase
        self._foot_contact_history:np.ndarray = None  # (legNum,3) store foot positions

        # rotation from ground frame to base frame
        self.ground_R_body_frame:np.ndarray = None
        self.world_R_yaw_frame:np.ndarray = None
        self.body_height:float = self._robot._bodyHeight
        self.result.position[2] = self.body_height

    def reset(self):
        self.result = StateEstimate()
        self._phase = np.zeros((self._robot._legNum,1), dtype=DTYPE)
        self._contactPhase = self._phase
        self._foot_contact_history:np.ndarray = None
        self.ground_R_body_frame:np.ndarray = None
        self.world_R_yaw_frame:np.ndarray = None
        self.body_height = self._robot._bodyHeight
        self.result.position[2] = self.body_height

    def setContactPhase(self, phase:np.ndarray):
        '''

        '''
        self._contactPhase = phase

    def getResult(self):
        return self.result

    def update(self, body_states:dict):
        self.result.orientation.w = body_states["bQuat"][0]
        self.result.orientation.x = body_states["bQuat"][1]
        self.result.orientation.y = body_states["bQuat"][2]
        self.result.orientation.z = body_states["bQuat"][3]

        self.result.vWorld[0, 0] = body_states["bLinVel"][0]
        self.result.vWorld[1, 0] = body_states["bLinVel"][1]
        self.result.vWorld[2, 0] = body_states["bLinVel"][2]
        self.result.omegaWorld[0, 0] = body_states["bAngVel"][0]
        self.result.omegaWorld[1, 0] = body_states["bAngVel"][1]
        self.result.omegaWorld[2, 0] = body_states["bAngVel"][2]

        # frame transformation

        # body rotation in the world frame
        self.result.rBody = quat_to_rot(self.result.orientation)
        # RPY of body in the world frame
        self.result.rpy = quat_to_rpy(self.result.orientation)
        # body linear velocity w.r.t. base frame 
        self.result.vBody = self.result.rBody.T @ self.result.vWorld
        # body angular velocity w.r.t. base frame
        self.result.omegaBody = self.result.rBody.T @ self.result.omegaWorld

        world_R_yaw_frame = rpy_to_rot([0,0,self.result.rpy[2]])
        self.world_R_yaw_frame = world_R_yaw_frame
        yaw_R_ground_frame = get_rot_from_normals(np.array([0,0,1], dtype=DTYPE),
                                                    self.result.ground_normal_yaw)
        self.ground_R_body_frame = yaw_R_ground_frame.T @ world_R_yaw_frame.T @ self.result.rBody

        # RPY of body in yaw aligned world frame
        self.result.rpy_yaw = rot_to_rpy(self.world_R_yaw_frame.T @ self.result.rBody)
        # RPY of body in yaw aligned ground frame
        self.result.rpy_ground = rot_to_rpy(self.ground_R_body_frame)

    def _init_contact_history(self, foot_positions:np.ndarray):
        self._foot_contact_history = foot_positions.copy()
        self._foot_contact_history[:, 2] = - self.body_height

    def _update_contact_history(self, foot_positions:np.ndarray):
        foot_positions_ = foot_positions.copy()
        for leg_id in range(self._robot._legNum):
            if self._contactPhase[leg_id]:
                self._foot_contact_history[leg_id] = foot_positions_[leg_id]

    def _update_com_position_ground_frame(self, foot_positions:np.ndarray):
        '''
        foot_positions : (legNum,3) in the base frame
        '''
        foot_contacts = self._contactPhase.flatten()
        if np.sum(foot_contacts) == 0:
            return np.array((0, 0, self.body_height))
        else:
            foot_positions_ground_frame = (foot_positions.reshape((self._robot._legNum,3)).dot(self.ground_R_body_frame.T))
            foot_heights = -foot_positions_ground_frame[:, 2]

        height_in_ground_frame = np.sum(foot_heights * foot_contacts) / np.sum(foot_contacts)
        self.result.position[2] = height_in_ground_frame

    def _compute_ground_normal_and_com_position(self, foot_positions:np.ndarray):
        """
        Computes the surface orientation in robot frame based on foot positions.
        Solves a least squares problem, see the following paper for details:
        https://ieeexplore.ieee.org/document/7354099

        a * x + b * y + c * z = 1
        (a,b,c) is the normal vector of the plane.

        foot_positions : (legNum,3) in the base frame
        """
        self._update_com_position_ground_frame(foot_positions)
        self._update_contact_history(foot_positions)

        contact_foot_positions = self._foot_contact_history.reshape((self._robot._legNum,3)) # reshape from (legNum,3,1) to (legNum,3)
        normal_vec = scipy.linalg.lstsq(contact_foot_positions, np.ones(self._robot._legNum, dtype=DTYPE))[0]
        # numpy lstsq not support float16 or less
        # normal_vec = np.linalg.lstsq(contact_foot_positions.astype(np.float32), 
        #                              np.ones(4, dtype=np.float32), rcond=None)[0]
        normal_vec /= np.linalg.norm(normal_vec)
        if normal_vec[2] < 0:
            normal_vec = -normal_vec

        # _ground_normal = self._ground_normal_filter.calculate_average(normal_vec)
        _ground_normal = normal_vec
        _ground_normal /= np.linalg.norm(_ground_normal)

        # ground normal in world frame and yaw aligned frame
        self.result.ground_normal_world = self.result.rBody @ _ground_normal
        world_R_yaw_frame = rpy_to_rot([0,0,self.result.rpy[2]])
        self.result.ground_normal_yaw = world_R_yaw_frame.T @ self.result.ground_normal_world