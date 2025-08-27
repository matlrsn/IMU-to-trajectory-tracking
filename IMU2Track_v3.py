# %% ====== CONTEXT AND SOURCES ======
# ====================================
# This script implements a sensor fusion algorithm (based on Madgwick's AHRS)
# to estimate 3D orientation and position from IMU data.
# It processes raw sensor inputs, applies filtering, and outputs orientation
# (quaternion + Euler angles), velocity, and position over time.
# Core math steps include computing the gradient from the Jacobian and residual
# normalizing update vectors, and integrating to track motion.
# ====================================
# Sources:
# Madgwick S.O.H., "An efficient orientation filter for inertial and magnetic sensor arrays" (2010),
# related open-source implementations.
# Documentation and mathematical explanation for implementation:
# https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
# %% ====== IMPORTS =======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from collections import deque

# %% ====== IMU DATA processing and post processing ======


@dataclass
class IMUConfig:
    """Configuration parameters for IMU processing"""

    # file name of data to be processed, replace with your actual file name
    data_file: str = 'data/100hz_final_test_001.xlsx'

    # Sampling parameters
    sample_rate: float = 100.0             # Sample rate (IMU frames rate) [Hz]

    # Bias estimation parameters
    bias_estimation_window: int = 8400     # number of samples for initial bias estimation

    # Filter parameters
    madgwick_beta: float = 0.015           # Madgwick filter gain (0.041 as default)
    acc_lpf_cutoff: float = 25.0           # [Hz] - accelerometer low-pass filter
    gyro_hpf_cutoff: float = 0.20          # [Hz] - gyroscope high-pass filter
    MARG: bool = False                      # Set to true if you want mag data to be integrated

    # Gravity parameters
    gravity_magnitude: float = 9.81        # [m/s²]
    gravity_alignment_samples: int = 8400  # number of samples for initial gravity alignment

    # Motion constraints
    max_velocity: float = 2.0              # [m/s] - maximum reasonable velocity
    max_acceleration: float = 15.0         # [m/s²] - maximum reasonable acceleration

    # Advanced parameters
    outlier_threshold: float = 3.0         # standard deviations for outlier detection
    smoothing_window: int = 10             # number of samples for temporal smoothing


class BiasEstimator:
    """Estimates and removes sensor biases"""

    def __init__(self, config: IMUConfig):
        self.config = config             # loads actual configuration
        self.acc_bias = np.zeros(3)      # initializes acceleration bias as a zero numpy array
        self.gyro_bias = np.zeros(3)     # initializes gyro bias as a zero numpy array
        self.is_initialized = False      # bias not yet initialized

    def initialize_bias(self, acc_data: np.ndarray, gyro_data: np.ndarray) -> None:
        """Initialize bias from stationary data at the beginning"""

        # Checks if number of samples desired for calibration do not exceed the number of samples available in the dataset
        n_samples = min(self.config.bias_estimation_window, len(acc_data))

        # Estimate gyroscope bias on each axis (axis = 0 is computed mean column wise)
        self.gyro_bias = np.mean(gyro_data[:n_samples], axis=0)           # average gyro value over n samples

        # Estimate accelerometer bias: mean of stationary data, corrected norm ≈ gravity
        acc_mean = np.mean(acc_data[:n_samples], axis=0)

        # Compute norm of mean acceleration
        acc_mean_norm = np.linalg.norm(acc_mean)
        if acc_mean_norm > 0:
            # Scale the mean to make its norm equal to gravity magnitude
            scale_factor = self.config.gravity_magnitude / acc_mean_norm
            # Bias is the difference between raw mean and scaled mean (which has norm = g)
            self.acc_bias = acc_mean - (acc_mean * scale_factor)
        else:
            self.acc_bias = np.zeros(3)  # Fallback if norm is zero (rare)

        self.is_initialized = True                                        # bias is now initialized

    def correct_measurements(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bias correction to measurements"""
        if not self.is_initialized:
            return acc, gyro                     # if bias is not initialized, return actual acc and gyro values

        acc_corrected = acc - self.acc_bias      # compute corrected acceleration values (actual values - bias)
        gyro_corrected = gyro - self.gyro_bias   # compute corrected gyroscopic values (actual values - bias)

        return acc_corrected, gyro_corrected


class MadgwickFilter:
    """Enhanced Madgwick AHRS filter implementation"""

    def __init__(self, config: IMUConfig):
        self.config = config                     # load actual congifuration
        self.beta = config.madgwick_beta         # Madgwick filter gain
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # initial quaternion [w, x, y, z]

    @staticmethod
    def _quaternion_product(a, b):
        """Defines the logic of a product of two quaternions a and b"""
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

    @staticmethod
    def _quaternion_conjugate(q):
        """Defines the logic of the conjugate of a quaternion"""
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    def update(self, acc: np.ndarray, gyro: np.ndarray,
               mag: np.ndarray, dt: float = 1.0/100.0) -> np.ndarray:
        """Update orientation estimate"""
        # Normalize accelerometer measurement
        if np.linalg.norm(acc) == 0:
            return self.q

        acc = acc / np.linalg.norm(acc)

        # Normalize magnetometer measurement
        if np.linalg.norm(mag) == 0:
            return self.q

        mag = mag / np.linalg.norm(mag)

        # Extract quaternion components
        q0, q1, q2, q3 = self.q

        # Compute reference direction of Earth's magnetic field from the current quaternion
        h = self._quaternion_product(
            self.q,
            self._quaternion_product(np.array([0, *mag]), self._quaternion_conjugate(self.q))
            )
        bx = np.sqrt(h[1]**2 + h[2]**2)            # Northward component of earth magnetic field (weighted with eastward component)
        bz = h[3]                                  # Northward component of earth magnetic field

        # Gradient descent algorithm corrective step
        # Accelerometer objective function
        f_acc = np.array([
            2*(q1*q3 - q0*q2) - acc[0],
            2*(q0*q1 + q2*q3) - acc[1],
            2*(0.5 - q1**2 - q2**2) - acc[2]
        ])

        # Magnetometer objective function
        f_mag = np.array([
            2*bx*(0.5 - q2**2 - q3**2) + 2*bz*(q1*q3 - q0*q2) - mag[0],
            2*bx*(q1*q2 - q0*q3) + 2*bz*(q0*q1 + q2*q3) - mag[1],
            2*bx*(q0*q2 + q1*q3) + 2*bz*(0.5 - q1**2 - q2**2) - mag[2]
        ])

        f = np.concatenate([f_acc, f_mag])          # Merge the two objective functions

        # Accelerometer Jacobian matrix
        j_acc = np.array([
            [-2*q2, 2*q3, -2*q0, 2*q1],
            [2*q1, 2*q0, 2*q3, 2*q2],
            [0, -4*q1, -4*q2, 0]
        ])

        # Magnetometer Jacobian matrix
        j_mag = np.array([
            [-2*bz*q2, 2*bz*q3, -4*bx*q2 - 2*bz*q0, -4*bx*q3 + 2*bz*q1],
            [-2*bx*q3 + 2*bz*q1, 2*bx*q2 + 2*bz*q0, 2*bx*q1 + 2*bz*q3, -2*bx*q0 + 2*bz*q2],
            [2*bx*q2, 2*bx*q3 - 4*bz*q1, 2*bx*q0 - 4*bz*q2, 2*bx*q1]
        ])

        J = np.vstack([j_acc, j_mag])               # Merge the two Jacobian matrices

        step = J.T.dot(f)                           # Gradient direction
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step = step / step_norm                 # Keep only one gradient direction

        # Rate of change of quaternion from gyroscope
        q_dot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],  # w component rate
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],   # x component rate
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],   # y component rate
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]    # z component rate
        ])

        # Apply feedback
        q_dot = q_dot - self.beta * step

        # Integrate to update quaternion estimate
        self.q = self.q + q_dot * dt
        self.q = self.q / np.linalg.norm(self.q)    # normalize

        return self.q.copy()

    # If no mag data is available, use this function instead
    def update_imu_only(self, acc: np.ndarray, gyro: np.ndarray, dt: float = 1.0/100.0) -> np.ndarray:
        """Update using only accelerometer and gyroscope (fallback method)"""

        # Normalize accelerometer measurement
        acc_norm = np.linalg.norm(acc)
        if acc_norm == 0:
            return self.q
        acc = acc / acc_norm

        # Extract quaternion components
        q0, q1, q2, q3 = self.q

        # Accelerometer objective function
        f = np.array([
            2*(q1*q3 - q0*q2) - acc[0],
            2*(q0*q1 + q2*q3) - acc[1],
            2*(0.5 - q1**2 - q2**2) - acc[2]
        ])

        # Accelerometer Jacobian matrix
        J = np.array([
            [-2*q2, 2*q3, -2*q0, 2*q1],
            [2*q1, 2*q0, 2*q3, 2*q2],
            [0, -4*q1, -4*q2, 0]
        ])

        # Gradient descent step
        step = J.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step = step / step_norm

        # Rate of change of quaternion from gyroscope
        q_dot_gyro = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])

        # Apply feedback
        q_dot = q_dot_gyro - self.beta * step

        # Integrate quaternion
        self.q = self.q + q_dot * dt
        self.q = self.q / np.linalg.norm(self.q)

        return self.q.copy()


class MotionConstraints:
    """Apply kinematic constraints to reduce drift"""

    def __init__(self, config: IMUConfig):
        self.config = config                                                  # loads actual configuration
        self.velocity_history = deque(maxlen=10)                              # size 10 buffer for velocity values

    def constrain_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Apply velocity constraints"""
        # Limit maximum velocity
        vel_magnitude = np.linalg.norm(velocity)                              # computes velocity norm
        if vel_magnitude > self.config.max_velocity:
            velocity = velocity * (self.config.max_velocity / vel_magnitude)  # corrects vel with normalized ratio

        # Store in history
        self.velocity_history.append(velocity.copy())

        return velocity

    def detect_outliers(self, acceleration: np.ndarray) -> bool:
        """Detect acceleration outliers"""
        acc_magnitude = np.linalg.norm(acceleration)                          # computes acceleration norm
        return acc_magnitude > self.config.max_acceleration                   # returns acc norm only under max acc


class IMUTracker:
    """IMU-based position tracking class with drift compensation"""

    def __init__(self, config: IMUConfig):
        self.config = config                                 # loads actual configuration
        self.bias_estimator = BiasEstimator(config)          # loads bias estimator class
        self.madgwick = MadgwickFilter(config)               # loads Madgwick filter class
        self.motion_constraints = MotionConstraints(config)  # loads motion constraints class
        self.sample_index = 0                                # keep track of current sample

        # State vectors
        self.position = np.zeros(3)                          # initial position state vector (numpy zeros array)
        self.velocity = np.zeros(3)                          # initial velocity state vector (numpy zeros array)
        self.acceleration = np.zeros(3)                      # initial acceleration state vector (numpy zeros array)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])    # [w, x, y, z] initial quaternion

        # History for smoothing and analysis
        self.position_history = []                           # position history list
        self.velocity_history = []                           # velocity history list
        self.acceleration_history = []                       # acceleration history list
        self.orientation_history = []                        # orientation history list

        # Gravity vector in world frame ([0, 0, -g])
        self.gravity_world = np.array([0.0, 0.0, -self.config.gravity_magnitude])

    def initialize(self, acc_data: np.ndarray, gyro_data: np.ndarray,
                   mag_data: Optional[np.ndarray] = None) -> None:
        """Initialize the tracker with initial data"""
        # Initialize bias estimation
        self.bias_estimator.initialize_bias(acc_data, gyro_data)

        # Initialize orientation from first few samples
        acc_mean = np.mean(acc_data[:self.config.gravity_alignment_samples], axis=0)
        acc_mean_corrected, _ = self.bias_estimator.correct_measurements(acc_mean, np.zeros(3))

        # Step 1: Align with gravity (as before)
        gravity_body = acc_mean_corrected / np.linalg.norm(acc_mean_corrected)
        gravity_world = np.array([0, 0, -1])

        # Calculate initial rotation to align body frame with world frame (gravity alignment)
        if np.allclose(gravity_body, gravity_world):                                # checks if vectors are equal
            gravity_alignment_q = np.array([1.0, 0.0, 0.0, 0.0])                    # if equal, no rotation needed
        else:
            cross = np.cross(gravity_body, gravity_world)                           # computes rotation axis
            dot = np.dot(gravity_body, gravity_world)                               # rotation angle
            if np.linalg.norm(cross) > 1e-6:                                        # if cross is non zero (vectors not //)
                axis = cross / np.linalg.norm(cross)                                # normalize cross
                angle = np.arccos(np.clip(dot, -1, 1))                              # angle between vectors
                gravity_alignment_q = self._axis_angle_to_quaternion(axis, angle)   # convert axis angle to quaternion
            else:
                if dot > 0:  # parallel: no rotation needed
                    gravity_alignment_q = np.array([1.0, 0.0, 0.0, 0.0])
                else:  # anti-parallel: 180-degree rotation (around x-axis, arbitrary perpendicular)
                    gravity_alignment_q = np.array([0.0, 1.0, 0.0, 0.0])            # quaternion for 180 deg around x

                # Align initial orientation with gravity
                gravity_body = acc_mean_corrected / np.linalg.norm(acc_mean_corrected)  # normalized gravity in body frame
                gravity_world = np.array([0, 0, -1])                                    # normalized gravity in world frame

        # Step 2: Align x axis with magnetic north (if magnetometer data is available)
        if mag_data is not None and self.config.MARG:
            print("Performing magnetic north alignment...")

            # Get mean magnetometer reading
            mag_mean = np.mean(mag_data[:self.config.gravity_alignment_samples], axis=0)
            mag_mean_norm = np.linalg.norm(mag_mean)

            if mag_mean_norm > 1e-6:                                    # Check if magnetometer data is valid
                mag_body_normalized = mag_mean / mag_mean_norm

                # Transform magnetometer reading to the gravity-aligned frame
                R_gravity = self._quaternion_to_rotation_matrix(gravity_alignment_q)
                mag_gravity_aligned = R_gravity @ mag_body_normalized

                # Project magnetometer vector onto horizontal plane (remove Z component)
                mag_horizontal = np.array([mag_gravity_aligned[0], mag_gravity_aligned[1], 0.0])
                mag_horizontal_norm = np.linalg.norm(mag_horizontal)

                if mag_horizontal_norm > 1e-6:                          # Check if horizontal component exists
                    mag_horizontal_normalized = mag_horizontal / mag_horizontal_norm

                    # For NED convention: X points North, Y points East
                    magnetic_north_world = np.array([1.0, 0.0, 0.0])

                    # Calculate rotation angle around Z-axis to align with magnetic north
                    cos_yaw = np.dot(mag_horizontal_normalized, magnetic_north_world)
                    sin_yaw = np.cross(mag_horizontal_normalized, magnetic_north_world)[2]  # Z-component of cross product

                    yaw_angle = np.arctan2(sin_yaw, cos_yaw)

                    # Create quaternion for yaw rotation around Z-axis
                    yaw_q = self._axis_angle_to_quaternion(np.array([0, 0, 1]), yaw_angle)

                    # Combine gravity alignment and magnetic alignment
                    self.orientation = MadgwickFilter._quaternion_product(yaw_q, gravity_alignment_q)

                    print(f"Initial yaw correction: {np.degrees(yaw_angle):.2f} degrees")
                else:
                    print("Warning: Magnetometer horizontal component too small, using gravity alignment only")
                    self.orientation = gravity_alignment_q
            else:
                print("Warning: Invalid magnetometer data, using gravity alignment only")
                self.orientation = gravity_alignment_q
        else:
            print("No magnetometer data available, using gravity alignment only")
            self.orientation = gravity_alignment_q

        print("Initial orientation quaternion:")
        print(f"[{self.orientation[0]:.4f}, {self.orientation[1]:.4f}, {self.orientation[2]:.4f}, {self.orientation[3]:.4f}]")

        # Convert to Euler angles for verification
        initial_euler = self.get_euler_angles(self.orientation)
        print(f"Initial Euler angles - Yaw: {initial_euler[0]:.2f}°, Pitch: {initial_euler[1]:.2f}°, Roll: {initial_euler[2]:.2f}°")

        self.madgwick.q = self.orientation.copy()

    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle representation to quaternion (w, x, y, z)"""
        half_angle = angle / 2             # quaternions represent rotations using half-angles
        sin_half = np.sin(half_angle)      # scale angles
        # w = cos(angle/2) => scalar part ; [x, y, z] = axis * sin(angle/2) => vector part along rotation axis
        return np.array([np.cos(half_angle), axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half])

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])

    @staticmethod
    def norm_q(q: np.ndarray) -> float:
        """Returns the norm of a quaternion"""
        return float(np.linalg.norm(q))

    def process_sample(self, acc_raw: np.ndarray, gyro_raw: np.ndarray,
                       mag_raw: Optional[np.ndarray] = None, dt: float = 1.0/100.0) -> Dict[str, Any]:
        """Process a single IMU sample"""

        # Bias correction
        acc_corrected, gyro_corrected = self.bias_estimator.correct_measurements(acc_raw, gyro_raw)

        # Only update Madgwick after bias window
        if self.sample_index >= self.config.bias_estimation_window:
            if self.config.MARG:
                self.orientation = self.madgwick.update(acc_corrected, gyro_corrected, mag_raw, dt)
            else:
                self.orientation = self.madgwick.update_imu_only(acc_corrected, gyro_corrected, dt)

            # Transform acceleration to world frame
            R_body_to_world = self._quaternion_to_rotation_matrix(self.orientation)
            acc_world = R_body_to_world @ acc_corrected

            # Remove gravity
            acc_world_no_gravity = acc_world + self.gravity_world             # gravity_world is negative (hence the + sign)

            # Check for outliers
            is_outlier = self.motion_constraints.detect_outliers(acc_world_no_gravity)

            if not is_outlier:
                self.acceleration = acc_world_no_gravity
            # else keep previous acceleration

            # Integrate acceleration to velocity using trapezoidal rule (semi-implicit Euler)
            self.velocity += self.acceleration * dt

            # Apply velocity constraints
            self.velocity = self.motion_constraints.constrain_velocity(self.velocity)

            # Integrate velocity to position
            self.position += self.velocity * dt
        else:
            # Before Madgwick starts, keep initial orientation
            self.orientation = self.madgwick.q.copy()
            self.acceleration = np.zeros(3)

        self.sample_index += 1  # increment counter

        # Store history
        self.position_history.append(self.position.copy())                # append to position history list
        self.velocity_history.append(self.velocity.copy())                # append to velocity history list
        self.acceleration_history.append(self.acceleration.copy())        # append to acceleration history list
        self.orientation_history.append(self.orientation.copy())

        euler_angles = self.get_euler_angles(self.orientation)            # euler angle conversion from quaternions

        # returns data to be added to the dataframe
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'orientation': self.orientation.copy(),
            '|q|': IMUTracker.norm_q(self.orientation),
            'euler': euler_angles
        }

    def process_data(self, acc_data: np.ndarray, gyro_data: np.ndarray,
                     mag_data: Optional[np.ndarray] = None,
                     time_data: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Process complete IMU dataset"""

        if time_data is None:
            # if no time data, make it using data, dataset length and sample rate
            time_data = np.arange(len(acc_data)) / self.config.sample_rate

        # Initialize tracker
        self.initialize(acc_data, gyro_data, mag_data)

        results = []                                                        # list to store processed data results

        for i in range(len(acc_data)):
            dt = 1.0/self.config.sample_rate if i == 0 else time_data[i] - time_data[i-1]

            mag_sample = mag_data[i] if mag_data is not None else None      # extract mag data sample if available
            result = self.process_sample(acc_data[i], gyro_data[i], mag_sample, dt)

            # Add time and sample index
            result['time'] = time_data[i]
            result['sample'] = i

            results.append(result)                                          # add to the results list

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Expand vector columns
        for col in ['position', 'velocity', 'acceleration', 'orientation', 'euler']:
            if col == 'orientation':
                df_results[['qw', 'qx', 'qy', 'qz']] = pd.DataFrame(df_results[col].tolist())
            elif col == 'euler':
                df_results[['yaw', 'pitch', 'roll']] = pd.DataFrame(df_results[col].tolist())
            else:
                suffix = ['x', 'y', 'z']
                df_results[[f'{col}_{s}' for s in suffix]] = pd.DataFrame(df_results[col].tolist())

        # Drop the original columns after expansion
        df_results.drop(['position', 'velocity', 'acceleration', 'orientation', 'euler'], axis=1, inplace=True)

        return df_results

    def get_euler_angles(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion (w, x, y, z) to Euler angles (yaw, pitch, roll) in degrees
        Using aerospace convention: Roll(X), Pitch(Y), Yaw(Z)
        - Roll: positive when banking right (CCW from front view)
        - Pitch: positive when nose up (CCW from right view)
        - Yaw: positive when turning CCW (from above view)
        """
        # Convert to scipy format (x, y, z, w)
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
        rotation = R.from_quat(q_scipy)

        # Use extrinsic XYZ rotation (equivalent to intrinsic ZYX)
        # This gives us the standard aerospace sequence: Yaw -> Pitch -> Roll
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)

        return np.array([yaw, pitch, roll])

    def plot_results(self, df_results: pd.DataFrame) -> None:
        """Plot tracking results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))                    # 1 main plot, 3 subplots, figure size 15x10

        # 3D trajectory
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(df_results['position_x'], df_results['position_y'], df_results['position_z'], label="Trajectory")

        # Mark start point (green)
        ax.scatter(df_results['position_x'].iloc[0],
                   df_results['position_y'].iloc[0],
                   df_results['position_z'].iloc[0],
                   color='green', s=50, label='Start')

        # Mark end point (red)
        ax.scatter(df_results['position_x'].iloc[-1],
                   df_results['position_y'].iloc[-1],
                   df_results['position_z'].iloc[-1],
                   color='red', s=50, label='End')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()

        # Velocity components

        for comp in ['x', 'y', 'z']:
            axes[0, 1].plot(df_results['time'], df_results[f'velocity_{comp}'], label=f'V{comp}')
            axes[0, 1].plot(df_results['time'],
                            df_results[f'velocity_{comp}'].rolling(200).mean(),
                            label=f'V{comp} MA({200})', linestyle='--')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Position components
        axes[1, 0].plot(df_results['time'], df_results['position_x'], label='X')
        axes[1, 0].plot(df_results['time'], df_results['position_y'], label='Y')
        axes[1, 0].plot(df_results['time'], df_results['position_z'], label='Z')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].set_title('Position Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        for angle in ['yaw', 'roll', 'pitch']:
            axes[1, 1].plot(df_results['time'], df_results[angle], label=angle)
            axes[1, 1].plot(df_results['time'],
                            df_results[angle].rolling(200).mean(),
                            label=f'{angle} MA({200})', linestyle='--')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Euler angles (°)')
        axes[1, 1].set_title('Euler angles components')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


def process_imu_data(file_path: str, config: Optional[IMUConfig] = None) -> pd.DataFrame:
    """
    Process IMU data with batch filtering - MAIN PROCESSING FUNCTION
    This applies filters to the entire dataset before processing
    """

    if config is None:
        config = IMUConfig()

    print("Loading IMU data...")
    # Read data
    df = pd.read_excel(file_path)

    # Extract IMU data
    acc_cols = ['acc_x(m/s²)', 'acc_y(m/s²)', 'acc_z(m/s²)']
    gyro_cols = ['ω_x(rad/s)', 'ω_y(rad/s)', 'ω_z(rad/s)']
    mag_cols = ['mag_x(µT)', 'mag_y(µT)', 'mag_z(µT)']

    # Check for required columns
    required_cols = acc_cols + gyro_cols
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    print(f"Found {len(df)} samples")

    # Clean data
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"After cleaning: {len(df)} samples")

    # Extract data arrays
    acc_data = df[acc_cols].values
    gyro_data = df[gyro_cols].values
    mag_data = df[mag_cols].values if all(col in df.columns for col in mag_cols) else None

    print("Applying batch filtering...")

    # BATCH FILTERING
    nyquist = config.sample_rate / 2

    # Filter accelerometer data (low-pass filter to remove high-frequency noise)
    if config.acc_lpf_cutoff < nyquist:
        print(f"Applying accelerometer low-pass filter (cutoff: {config.acc_lpf_cutoff} Hz)")
        acc_lpf_b, acc_lpf_a = butter(4, config.acc_lpf_cutoff / nyquist, 'low')
        for i in range(3):  # Apply to each axis (x, y, z)
            acc_data[:, i] = filtfilt(acc_lpf_b, acc_lpf_a, acc_data[:, i])
    else:
        print("Accelerometer low-pass filter disabled (cutoff frequency too high)")

    # Filter gyroscope data (high-pass filter to remove bias drift)
    if config.gyro_hpf_cutoff > 0 and config.gyro_hpf_cutoff < nyquist:
        print(f"Applying gyroscope high-pass filter (cutoff: {config.gyro_hpf_cutoff} Hz)")
        gyro_hpf_b, gyro_hpf_a = butter(4, config.gyro_hpf_cutoff / nyquist, 'high')
        for i in range(3):  # Apply to each axis (x, y, z)
            gyro_data[:, i] = filtfilt(gyro_hpf_b, gyro_hpf_a, gyro_data[:, i])
    else:
        print("Gyroscope high-pass filter disabled (cutoff frequency invalid)")

    # Filter magnetometer data if available (optional low-pass filtering)
    if mag_data is not None and config.acc_lpf_cutoff < nyquist:
        print("Applying magnetometer low-pass filter")
        for i in range(3):  # Apply to each axis (x, y, z)
            mag_data[:, i] = filtfilt(acc_lpf_b, acc_lpf_a, mag_data[:, i])

    # Create time vector
    if 'timestamp(s)' in df.columns:
        time_data = df['timestamp(s)'].values
        print("Using timestamp data from file")
    else:
        time_data = np.arange(len(df)) / config.sample_rate
        print(f"Creating synthetic time vector (sample rate: {config.sample_rate} Hz)")

    print("Processing filtered data...")

    # Create tracker and process filtered data
    # Note: We don't need internal filtering anymore since data is pre-filtered
    tracker = IMUTracker(config)
    results = tracker.process_data(acc_data, gyro_data, mag_data, time_data)

    print("Processing complete!")
    print(f"Final trajectory length: {np.linalg.norm(results[['position_x', 'position_y', 'position_z']].iloc[-1].values):.3f} m")

    # Plot results
    tracker.plot_results(results)

    return results


# Usage example:
config = IMUConfig(sample_rate=100.0)
results = process_imu_data(config.data_file, config)
