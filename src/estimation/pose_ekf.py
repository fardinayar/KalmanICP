"""
Pose Extended Kalman Filter (EKF) for IMU-based SLAM.

This module implements a pose-only EKF that estimates the 6DOF pose (position and orientation)
of a vehicle using IMU measurements and LiDAR-based pose corrections from ICP.

The filter maintains:
- State: [position (3), orientation (3)] in world frame
- Process model: IMU integration with gravity compensation
- Measurement model: LiDAR pose corrections from ICP
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from ..utils.geometry import so3_exp, so3_log, skew, T_from_R_t, invert_se3, adjoint


@dataclass
class PoseEKFConfig:
    """Configuration parameters for the Pose EKF.
    
    Attributes:
        sigma_g: Gyroscope noise standard deviation (rad/sqrt(s)) in body frame
        sigma_a: Accelerometer noise standard deviation (m/s^2/sqrt(s)) in body frame
        gravity: Gravity vector in world frame (m/s^2)
        gyro_bias: Static gyroscope bias to subtract (rad/s) in body frame
        accel_bias: Static accelerometer bias to subtract (m/s^2) in body frame
        P0_diag: Initial error state covariance diagonal [dtheta(3), dp(3)]
        use_joseph_form: Whether to use Joseph form for covariance update
        min_dt: Minimum time step to avoid numerical issues (s)
        max_dt: Maximum time step for single prediction (s)
    """
    sigma_g: float
    sigma_a: float
    gravity: np.ndarray  # shape (3,)
    gyro_bias: np.ndarray  # shape (3,)
    accel_bias: np.ndarray  # shape (3,)
    P0_diag: np.ndarray = None  # shape (6,) - will default to 1e-3 if None
    use_joseph_form: bool = True
    min_dt: float = 1e-6
    max_dt: float = 1.0


class PoseEKF:
    """Pose Extended Kalman Filter for 6DOF pose estimation.
    
    This EKF estimates the vehicle's pose (position and orientation) in world frame
    using IMU measurements for prediction and LiDAR-based pose corrections for updates.
    
    State representation:
        - Position: p_w (3x1) in world frame
        - Orientation: R_wi (3x3) rotation matrix from IMU to world frame
        - Error state: [dtheta (3), dp (3)] where dtheta is small-angle rotation error
    
    Process model:
        - IMU integration with gravity compensation
        - Constant acceleration assumption over short time intervals
    
    Measurement model:
        - LiDAR pose corrections from ICP matching
    """
    
    def __init__(self, R0: np.ndarray, p0: np.ndarray, config: PoseEKFConfig):
        """Initialize the Pose EKF.
        
        Args:
            R0: Initial rotation matrix (3x3) from IMU to world frame
            p0: Initial position (3x1) in world frame
            config: EKF configuration parameters
        """
        self.R = R0.copy()  # Rotation matrix R_wi (IMU to world)
        self.p = p0.reshape(3).copy()  # Position in world frame
        
        # Initialize error state covariance
        if config.P0_diag is not None:
            self.P = np.diag(config.P0_diag)
        else:
            self.P = np.eye(6) * 1e-3  # Default covariance
        
        self.cfg = config

    def get_T_wi(self) -> np.ndarray:
        """Get the current pose as a 4x4 transformation matrix.
        
        Returns:
            4x4 transformation matrix T_wi (IMU to world frame)
        """
        return T_from_R_t(self.R, self.p)

    def predict(self, omega_m: np.ndarray, a_m: np.ndarray, dt: float) -> None:
        """Predict step using IMU measurements.
        
        This method integrates IMU measurements to update the pose estimate and
        propagates the error state covariance. The process model assumes constant
        acceleration over the time interval dt.
        
        Args:
            omega_m: Measured angular velocity (3x1) in body frame (rad/s)
            a_m: Measured acceleration (3x1) in body frame (m/s^2)
            dt: Time step (s)
        """
        # Validate time step
        dt = max(self.cfg.min_dt, min(dt, self.cfg.max_dt))
        
        # Apply bias correction to IMU measurements
        omega = omega_m.reshape(3) - self.cfg.gyro_bias.reshape(3)
        acc_body = a_m.reshape(3) - self.cfg.accel_bias.reshape(3)

        # Orientation update: R_new = R_old * exp(omega * dt)
        # This integrates the angular velocity to update the rotation matrix
        # Left multiplication: body frame rotation applied to current world-to-body transform
        dR = so3_exp(omega * dt)
        self.R = self.R @ dR

        # Position update: double integrate acceleration with gravity compensation
        # a_world = R_wi * a_body + g_world
        # p_new = p_old + 0.5 * a_world * dt^2 (constant acceleration assumption)
        a_world = self.R @ acc_body + self.cfg.gravity.reshape(3)
        self.p = self.p + 0.5 * a_world * (dt * dt)

        # Covariance propagation for error state [dtheta, dp]
        # This implements the error state transition for IMU integration
        F = np.zeros((6, 6))  # State transition matrix
        G = np.zeros((6, 6))  # Noise mapping matrix
        
        # Error state: [dtheta (3), dp (3)]
        # F: Jacobian of state transition w.r.t. error state
        # For pose-only EKF with constant acceleration assumption:
        F[:3, :3] = np.eye(3)  # dtheta/dtheta (orientation error propagates)
        F[3:, :3] = -self.R @ skew(acc_body) * 0.5 * (dt * dt)  # dp/dtheta (acceleration coupling)
        F[3:, 3:] = np.eye(3)  # dp/dp (position error propagates)

        # Noise mapping: [gyro(3), accel(3)] into [dtheta(3), dp(3)]
        # Gyro noise affects orientation error directly
        G[:3, :3] = np.eye(3) * dt
        # Accel noise affects position error through rotation to world frame
        # For constant acceleration: dp = 0.5 * R * a_body * dt^2
        G[3:, 3:] = self.R * 0.5 * (dt * dt)

        # Process noise covariance (continuous-time)
        Qc = np.zeros((6, 6))
        Qc[:3, :3] = (self.cfg.sigma_g ** 2) * np.eye(3)  # Gyro noise
        Qc[3:, 3:] = (self.cfg.sigma_a ** 2) * np.eye(3)  # Accel noise

        # Discrete-time process noise
        Qd = G @ Qc @ G.T
        Phi = np.eye(6) + F * dt  # State transition matrix (discrete)
        self.P = Phi @ self.P @ Phi.T + Qd  # Covariance update

    def update_with_icp_pose(self, T_wl_meas: np.ndarray, T_il: np.ndarray, R_icp: np.ndarray) -> None:
        """Update step using LiDAR pose measurement from ICP.
        
        This method corrects the pose estimate using a LiDAR pose measurement obtained
        from ICP matching. The measurement model relates the LiDAR pose to the IMU pose
        through the known calibration transform T_il.
        
        Measurement Model:
            z = T_wl_meas (measured LiDAR pose in world frame)
            h(x) = T_wi(x) * T_il (predicted LiDAR pose from IMU pose)
            where T_wi is the current IMU-to-world transform and T_il is LiDAR-to-IMU calibration
            
        The innovation is computed as: r = log(T_wl_meas^(-1) * T_wl_pred)
        This represents the pose error between measurement and prediction.
        
        Args:
            T_wl_meas: Measured LiDAR pose (4x4) in world frame from ICP
            T_il: Calibration transform (4x4) from LiDAR to IMU frame  
            R_icp: Measurement noise covariance (6x6) for [x,y,z,roll,pitch,yaw]
        """
        # Get current IMU pose estimate
        T_wi = self.get_T_wi()  # Current IMU-to-world transform
        
        # Predict LiDAR pose from current IMU pose: T_wl_pred = T_wi * T_il
        # This transforms from LiDAR frame to world frame via IMU frame
        T_wl_pred = T_wi @ T_il
        
        # Compute pose error: T_err = T_wl_meas^(-1) * T_wl_pred
        # This gives the relative transform from measured to predicted pose
        T_err = invert_se3(T_wl_meas) @ T_wl_pred
        
        # Convert pose error to innovation vector [dt, dtheta] (6x1)
        # This represents the pose correction needed in the error state
        r = np.zeros(6)
        R_err = T_err[:3, :3]  # Rotation error matrix
        t_err = T_err[:3, 3]   # Translation error vector
        r[:3] = t_err          # Translation innovation (m)
        r[3:] = so3_log(R_err) # Rotation innovation (rad) - small angle approximation

        # Measurement Jacobian H: dh/dx where x = [dtheta, dp] (error state)
        # For pose measurements, the Jacobian relates pose errors to state errors
        H = np.zeros((6, 6))
        H[:3, :3] = np.zeros((3, 3))  # Translation w.r.t. rotation error (no direct coupling)
        H[:3, 3:] = np.eye(3)         # Translation w.r.t. position error (1:1 mapping)
        H[3:, :3] = np.eye(3)         # Rotation w.r.t. rotation error (1:1 mapping)  
        H[3:, 3:] = np.zeros((3, 3))  # Rotation w.r.t. position error (no direct coupling)

        # Kalman filter update equations
        S = H @ self.P @ H.T + R_icp  # Innovation covariance (6x6)
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain (6x6)
        dx = K @ (-r)  # State correction (negative innovation)

        # Apply state correction to error state [dtheta, dp]
        dtheta = dx[:3]  # Rotation correction (rad)
        dp = dx[3:]      # Position correction (m)

        # Update state: R_new = R_old * exp(dtheta), p_new = p_old + dp
        # Note: dtheta is applied in body frame (left multiplication)
        self.R = self.R @ so3_exp(dtheta)
        self.p = self.p + dp

        # Update covariance using Joseph form for numerical stability
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_icp @ K.T
