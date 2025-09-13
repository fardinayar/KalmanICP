"""
Geometric utility functions for SLAM.

This module provides essential geometric operations for 3D transformations,
rotation representations, and Lie group operations commonly used in SLAM
and robotics applications.

Key functionality:
- SO(3) operations: rotation matrix exponentials and logarithms
- SE(3) operations: rigid body transformation exponentials and logarithms
- Skew-symmetric matrix operations for rotation representations
- Coordinate frame transformations and conversions
- KITTI dataset format utilities
"""

import numpy as np
from typing import Tuple


def skew(vector: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector.
    
    The skew-symmetric matrix is used to represent cross products as matrix
    multiplications and is fundamental to rotation representations in SO(3).
    
    For vector v = [vx, vy, vz], the skew-symmetric matrix is:
        [0,  -vz,  vy]
        [vz,  0,  -vx]
        [-vy, vx,  0 ]
    
    This satisfies: skew(v) * w = v × w (cross product)
    
    Args:
        vector: 3D vector (3,) or array that can be reshaped to (3,)
        
    Returns:
        3x3 skew-symmetric matrix
    """
    v = vector.reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) to SO(3): axis-angle to rotation matrix.
    
    Converts an axis-angle representation (3D vector) to a rotation matrix
    using the Rodrigues' rotation formula. This is the exponential map
    from the Lie algebra so(3) to the Lie group SO(3).
    
    Formula: R = I + sin(θ)/θ * K + (1-cos(θ))/θ² * K²
    where K = skew(axis), θ = ||phi||, axis = phi/θ
    
    Args:
        phi: Axis-angle representation (3,) in radians
        
    Returns:
        3x3 rotation matrix in SO(3)
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        # Small angle approximation: R ≈ I + skew(phi)
        return np.eye(3) + skew(phi)
    axis = phi / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) to so(3): rotation matrix to axis-angle.
    
    Converts a rotation matrix to its axis-angle representation.
    This is the logarithmic map from the Lie group SO(3) to the Lie algebra so(3).
    
    The axis-angle representation is: phi = θ * axis
    where θ is the rotation angle and axis is the unit rotation axis.
    
    Args:
        R: 3x3 rotation matrix in SO(3)
        
    Returns:
        Axis-angle representation (3,) in radians
    """
    cos_theta = max(-1.0, min(1.0, (np.trace(R) - 1.0) * 0.5))
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        # Small angle case: return zero vector
        return np.zeros(3)
    # Extract rotation axis from skew-symmetric part
    w = (1.0 / (2.0 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * w


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) to SE(3): twist to rigid body transformation.
    
    Converts a 6D twist vector (translation + rotation) to a 4x4 homogeneous
    transformation matrix. This is the exponential map from the Lie algebra
    se(3) to the Lie group SE(3).
    
    The twist vector is: xi = [rho, phi] where:
    - rho (3,): translation component
    - phi (3,): rotation component (axis-angle)
    
    Args:
        xi: 6D twist vector [rho(3), phi(3)]
        
    Returns:
        4x4 homogeneous transformation matrix in SE(3)
    """
    rho = xi[:3]  # Translation component
    phi = xi[3:]  # Rotation component (axis-angle)
    R = so3_exp(phi)  # Convert rotation to matrix
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        # Small angle case: V ≈ I
        V = np.eye(3)
    else:
        # V matrix for translation component
        K = skew(phi / theta)
        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / (theta) * (K @ K)
        )
    t = V @ rho  # Apply V matrix to translation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithmic map from SE(3) to se(3): rigid body transformation to twist.
    
    Converts a 4x4 homogeneous transformation matrix to its 6D twist representation.
    This is the logarithmic map from the Lie group SE(3) to the Lie algebra se(3).
    
    The twist vector is: xi = [rho, phi] where:
    - rho (3,): translation component
    - phi (3,): rotation component (axis-angle)
    
    Args:
        T: 4x4 homogeneous transformation matrix in SE(3)
        
    Returns:
        6D twist vector [rho(3), phi(3)]
    """
    R = T[:3, :3]  # Rotation matrix
    t = T[:3, 3]   # Translation vector
    phi = so3_log(R)  # Convert rotation to axis-angle
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        # Small angle case: V_inv ≈ I
        V_inv = np.eye(3)
    else:
        # Inverse V matrix for translation component
        # Using normalized axis K = skew(phi/theta):
        # V_inv = I - 1/2 K + (1 - (theta/2) * cot(theta/2)) * K^2
        K = skew(phi / theta)
        half_theta = 0.5 * theta
        cot_term = (half_theta / np.tan(half_theta))  # equals (theta/2) * cot(theta/2)
        V_inv = (
            np.eye(3)
            - 0.5 * theta * K
            + (1.0 - cot_term) * (K @ K)
        )
    rho = V_inv @ t  # Extract translation component
    return np.hstack([rho, phi])


def adjoint(T: np.ndarray) -> np.ndarray:
    """Compute the adjoint matrix of a transformation.
    
    The adjoint matrix is used to transform twists between coordinate frames.
    It's essential for Lie group operations and coordinate transformations.
    
    For transformation T = [R, t; 0, 1], the adjoint is:
        Ad(T) = [R,    0  ]
                [skew(t)*R, R]
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        6x6 adjoint matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(t) @ R
    return Ad


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix.
    
    For transformation T = [R, t; 0, 1], the inverse is:
        T^(-1) = [R^T, -R^T*t; 0, 1]
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        4x4 inverse transformation matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """Compose two transformations: T_a_c = T_a_b * T_b_c.
    
    This function chains two transformations together. The result represents
    the transformation from frame C to frame A via frame B.
    
    Args:
        T_a_b: Transformation from frame B to frame A (4x4)
        T_b_c: Transformation from frame C to frame B (4x4)
        
    Returns:
        Transformation from frame C to frame A (4x4)
    """
    return T_a_b @ T_b_c


def R_from_rpy_deg(rpy_deg: Tuple[float, float, float]) -> np.ndarray:
    """Create rotation matrix from roll-pitch-yaw angles in degrees.
    
    The rotation is composed as: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    This follows the ZYX Euler angle convention (yaw-pitch-roll).
    
    Args:
        rpy_deg: Roll, pitch, yaw angles in degrees (r, p, y)
        
    Returns:
        3x3 rotation matrix
    """
    r, p, y = [np.deg2rad(x) for x in rpy_deg]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def T_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 transformation matrix from rotation and translation.
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def write_kitti_poses_txt(file_path: str, Twb_list: np.ndarray) -> None:
    """Write trajectory poses to KITTI format text file.
    
    The KITTI format stores each pose as a 3x4 matrix flattened row-major:
    [R11 R12 R13 t1]
    [R21 R22 R23 t2]  ->  [R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3]
    [R31 R32 R33 t3]
    
    Args:
        file_path: Output file path
        Twb_list: Array of 4x4 transformation matrices (N, 4, 4)
    """
    with open(file_path, 'w') as f:
        for T in Twb_list:
            R = T[:3, :3]
            t = T[:3, 3]
            M = np.hstack([R, t.reshape(3, 1)])
            row = M.reshape(-1)  # row-major flatten (3x4)
            f.write(' '.join([f"{v:.9f}" for v in row]) + '\n')
