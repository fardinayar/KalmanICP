"""
Constant Acceleration Extended Kalman Filter (CAEKF) for ICP pose smoothing.

This module implements an EKF that smooths ICP pose measurements using
a constant acceleration motion model. The filter helps reduce noise in ICP
measurements and provides smooth pose estimates for SLAM applications.

The CAEKF maintains global pose state vectors for translation and rotation:
- Translation: position and velocity in world frame
- Rotation: global rotation matrix and angular velocity

Key Features:
- Constant acceleration motion model for both translation and rotation
- Global rotation representation using rotation matrices
- Extended Kalman Filter for handling nonlinear rotation dynamics
- Smoothing of noisy ICP measurements
- Integration with pose-only EKF for enhanced SLAM performance
"""

import numpy as np
from dataclasses import dataclass
from ..utils.geometry import so3_exp, so3_log, skew, T_from_R_t


@dataclass
class CAEKFConfig:
    """Configuration parameters for the Constant Acceleration Kalman Filter.
    
    The CAKF uses separate noise models for translation and rotation components
    to account for different measurement characteristics and motion patterns.
    
    Attributes:
        q_acc: Process noise spectral density for translational acceleration (m^2/s^3)
        r_pos: Measurement noise standard deviation for position (m)
        p0_pos: Initial position standard deviation (m)
        p0_vel: Initial velocity standard deviation (m/s)
        q_rot_acc: Process noise spectral density for rotational acceleration (rad^2/s^3)
        r_rot: Measurement noise standard deviation for rotation small-angles (rad)
        p0_rot: Initial rotation standard deviation (rad)
        p0_rot_vel: Initial rotational velocity standard deviation (rad/s)
    """
    q_acc: float            # process noise spectral density (acceleration) [m^2 s^-3]
    r_pos: float            # measurement noise std for position [m]
    p0_pos: float           # initial std for position [m]
    p0_vel: float           # initial std for velocity [m/s]
    q_rot_acc: float = 0.1  # rotational acceleration spectral density [rad^2 s^-3]
    r_rot: float = 0.02     # measurement noise std for rotation small-angles [rad]
    p0_rot: float = 0.2     # initial std for rotation [rad]
    p0_rot_vel: float = 0.2 # initial std for rot velocity [rad/s]


class ConstantAccelerationKF:
    """Constant Acceleration Kalman Filter for ICP pose smoothing.
    
    This filter smooths noisy ICP pose measurements using a constant acceleration
    motion model. It maintains separate state vectors for translation and rotation
    components, allowing for different noise characteristics and motion patterns.
    
    State Representation (nominal):
        - R: global rotation matrix (3x3) in world frame
        - x = [p(3), v(3), w(3)] where:
          - p: position in world frame (m)
          - v: velocity in world frame (m/s)
          - w: angular velocity in body frame (rad/s)
    
    Error-State (EKF):
        - delta_x = [dtheta(3), dp(3), dv(3), dw(3)]
          where dtheta is a small-angle rotation error applied multiplicatively to R
    
    Motion Model:
        - Translation: constant velocity with white acceleration noise
        - Rotation: constant angular velocity with white angular-acceleration noise
    
    The filter is designed to work with the pose-only EKF by providing smoothed
    pose estimates that can be used as measurements or corrections.
    """
    
    def __init__(self, cfg: CAEKFConfig):
        """Initialize the Constant Acceleration Kalman Filter.
        
        Args:
            cfg: Configuration parameters for the filter
        """
        self.cfg = cfg
        
        # Nominal states: global rotation R and vector x = [p, v, w]
        self.R = np.eye(3)
        self.x = np.zeros(9)
        
        # Error-state covariance over [dtheta(3), dp(3), dv(3), dw(3)]
        self.P = np.diag([
            cfg.p0_rot**2, cfg.p0_rot**2, cfg.p0_rot**2,           # rotation error variance
            cfg.p0_pos**2, cfg.p0_pos**2, cfg.p0_pos**2,           # position variance
            cfg.p0_vel**2, cfg.p0_vel**2, cfg.p0_vel**2,           # velocity variance
            cfg.p0_rot_vel**2, cfg.p0_rot_vel**2, cfg.p0_rot_vel**2,  # angular velocity variance
        ])
        self.initialized = False

    def initialize(self, p_world: np.ndarray, R_world: np.ndarray = None):
        """Initialize the filter state with initial position and global rotation.
        
        Args:
            p_world: Initial position in world frame (3,)
            R_world: Initial global rotation matrix (3x3). If None, identity is used.
        """
        self.x[:3] = p_world.reshape(3)  # position
        self.x[3:6] = 0.0                # velocity
        self.x[6:9] = 0.0                # angular velocity
        self.R = (R_world.copy() if R_world is not None else np.eye(3))
        self.initialized = True

    def _Q_cv6(self, dt: float, q_acc: float):
        """Compute discrete process noise for constant-velocity model in 3D.
        
        This method computes the discrete-time process noise covariance matrix
        for a constant-velocity model with white acceleration noise. The model
        assumes constant acceleration over the time interval dt.
        
        The process noise matrix has the form:
            Q = q_acc * [Q_pos_pos*I, Q_pos_vel*I]
                     [Q_pos_vel*I, Q_vel_vel*I]
        
        where:
            Q_pos_pos = dt^3 / 3
            Q_pos_vel = dt^2 / 2  
            Q_vel_vel = dt
        
        Args:
            dt: Time step (s)
            q_acc: Acceleration noise spectral density (m^2/s^3 or rad^2/s^3)
            
        Returns:
            6x6 process noise covariance matrix
        """
        dt2 = dt * dt
        Q_pos_pos = (dt**3) / 3.0
        Q_pos_vel = (dt2) / 2.0
        Q_vel_vel = dt
        I3 = np.eye(3)
        Q6 = q_acc * np.block([
            [Q_pos_pos * I3, Q_pos_vel * I3],
            [Q_pos_vel * I3, Q_vel_vel * I3],
        ])
        return Q6

    def _F_Q(self, dt: float):
        """Build error-state transition F and process noise Q for [dtheta, dp, dv, dw]."""
        I3 = np.eye(3)
        F = np.eye(12)
        Q = np.zeros((12, 12))
        
        # Rotation error kinematics: dtheta' = dtheta + dt * dw
        F[:3, 9:12] = dt * I3
        
        # Translation error kinematics: dp' = dp + dt * dv; dv' = dv
        F[3:6, 6:9] = dt * I3
        
        # Process noise blocks using CV discretization
        # For rotation error and angular velocity: use q_rot_acc
        Q_rot = self._Q_cv6(dt, self.cfg.q_rot_acc)
        # For translation position and velocity: use q_acc
        Q_trans = self._Q_cv6(dt, self.cfg.q_acc)
        
        # Place translational process noise into [dp, dv]
        Q[3:9, 3:9] = Q_trans
        
        # Place rotational process noise into [dtheta, dw]
        Q[:3, :3] = Q_rot[:3, :3]
        Q[:3, 9:12] = Q_rot[:3, 3:6]
        Q[9:12, :3] = Q_rot[3:6, :3]
        Q[9:12, 9:12] = Q_rot[3:6, 3:6]
        
        return F, Q

    def predict(self, dt: float):
        """Propagate nominal state (R, p, v, w) and error covariance forward by dt."""
        if not self.initialized:
            return
        I3 = np.eye(3)
        
        # Unpack states
        p = self.x[:3]
        v = self.x[3:6]
        w = self.x[6:9]
        
        # Nominal kinematics
        # Rotation: R_new = R * exp(w * dt)
        self.R = self.R @ so3_exp(w * dt)
        # Translation: constant velocity model
        self.x[:3] = p + v * dt
        self.x[3:6] = v
        self.x[6:9] = w
        
        # Error-state propagation
        F, Q = self._F_Q(dt)
        self.P = F @ self.P @ F.T + Q

    def update_position(self, z_pos_world: np.ndarray):
        """Correct position using world-frame position measurement z = p."""
        if not self.initialized:
            self.initialize(z_pos_world)
            return
        # Residual in measurement space
        y = z_pos_world.reshape(3) - self.x[:3]
        
        # Measurement Jacobian H maps error-state to residual: y ≈ H * delta_x
        H = np.zeros((3, 12))
        H[:, 3:6] = np.eye(3)  # dp
        Rm = (self.cfg.r_pos ** 2) * np.eye(3)
        
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ y  # [dtheta, dp, dv, dw]
        
        # Apply correction to nominal states
        dtheta = delta[:3]
        dp = delta[3:6]
        dv = delta[6:9]
        dw = delta[9:12]
        
        self.R = self.R @ so3_exp(dtheta)
        self.x[:3] += dp
        self.x[3:6] += dv
        self.x[6:9] += dw
        
        # Covariance update (Joseph form)
        I = np.eye(12)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T

    def update_rotation_global(self, R_meas_world: np.ndarray):
        """Correct global rotation using a rotation matrix measurement in world frame.
        
        Residual r = log(R_pred^T * R_meas) (small-angle in body frame).
        """
        if not self.initialized:
            # Initialize with zero position and measured rotation
            self.initialize(np.zeros(3), R_meas_world)
            return
        # Compute rotational residual (3,)
        R_err = self.R.T @ R_meas_world
        y = so3_log(R_err)
        
        # Measurement Jacobian for rotation: y ≈ I * dtheta
        H = np.zeros((3, 12))
        H[:, :3] = np.eye(3)
        Rm = (self.cfg.r_rot ** 2) * np.eye(3)
        
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        delta = K @ y
        
        dtheta = delta[:3]
        dp = delta[3:6]
        dv = delta[6:9]
        dw = delta[9:12]
        
        self.R = self.R @ so3_exp(dtheta)
        self.x[:3] += dp
        self.x[3:6] += dv
        self.x[6:9] += dw
        
        I = np.eye(12)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T

    def get_position(self) -> np.ndarray:
        """Get current position estimate (world frame)."""
        return self.x[:3].copy()

    def get_rotation_matrix(self) -> np.ndarray:
        """Get current global rotation matrix (world frame)."""
        return self.R.copy()
