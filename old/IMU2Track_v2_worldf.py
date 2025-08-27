# Uses RK4 for acceleration and speed integration
# Uses quaternions for angular velocity integration
#%% IMPORTS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from ahrs.filters import Madgwick # not used here, implemented in the GSAT S/W

#%% CONSTANTS AND CONFIG (data preparation)

# Path to Excel file
current_dir = 'C:/Users/mathi/OneDrive/Documents/Toulouse/Stages_Travail/eNova/G_SAT/IMU2Track'
#file_name = 'IMU2Track-RotX_v2.xlsx'
#file_name = 'test.xlsx'
#file_name = 'test_marche.xlsx'
#file_name = 'drop_test.xlsx'
file_name = 'gpa_latence_2_003.xlsx'
#file_name = 'random_data.xlsx'
file_path = os.path.join(current_dir, file_name)

# Read Excel file
df = pd.read_excel(file_path)

# Remove rows with NaNs in accelerometer or gyroscope data
#df = df.dropna(subset=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']).reset_index(drop=True)
df = df.dropna(subset=['acc_x(m/s²)', 'acc_y(m/s²)', 'acc_z(m/s²)', 'ω_x(rad/s)', 'ω_y(rad/s)', 'ω_z(rad/s)']).reset_index(drop=True)
# Column names
#gx_column, gy_column, gz_column = 'Gx', 'Gy', 'Gz'
gx_column, gy_column, gz_column = 'ω_x(rad/s)', 'ω_y(rad/s)', 'ω_z(rad/s)'
#ax_column, ay_column, az_column = 'Ax', 'Ay', 'Az'
ax_column, ay_column, az_column = 'acc_x(m/s²)', 'acc_y(m/s²)', 'acc_z(m/s²)'


# Check for columns
required_columns = [gx_column, gy_column, gz_column, ax_column, ay_column, az_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns.")

# Time vector
#time_interval = 1/800
time_interval = 1/10
#df['time'] = np.arange(0, len(df) * time_interval, time_interval)
df['time'] = np.arange(len(df)) * time_interval

df['dt'] = df['time'].diff().fillna(0)

# Init state vectors
velocity = np.zeros((len(df), 3))
position = np.zeros((len(df), 3))

# Quaternion init (w, x, y, z)
quaternions = np.zeros((len(df), 4))
quaternions[0] = np.array([1, 0, 0, 0])  # Initial quaternion (1+i0+j0+k0)

#%% MADGWICK implementation (test to see if there is less drift with filter)

gyro_data = df[[gx_column, gy_column, gz_column]].to_numpy()  # shape (N, 3)
acc_data = df[[ax_column, ay_column, az_column]].to_numpy()   # shape (N, 3)
mag_data = df[['mag_x(µT)', 'mag_y(µT)', 'mag_z(µT)']].to_numpy()

madgwick = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data, gain=0.08)
Q = madgwick.Q

def integrate_accel_to_position(df, quaternions, dt_column='dt',
                               acc_cols=('Ax', 'Ay', 'Az'),
                               gyr_cols=('Gx', 'Gy', 'Gz')):

    N = len(df)
    velocity = np.zeros((N, 3))
    position = np.zeros((N, 3))

    def body_to_world(acc_body, quat_wxyz):
        return R.from_quat(np.roll(quat_wxyz, -1)).apply(acc_body)  # w,x,y,z -> x,y,z,w for scipy

    # Initialize
    acc_body_0 = df.loc[0, list(acc_cols)].values
    acc_world_prev = body_to_world(acc_body_0, quaternions[0])
    acc_world_prev -= np.array([0, 0, 9.81])  # Remove gravity

    for i in range(1, N):
        dt = df[dt_column].iloc[i]

        acc_body_curr = df.loc[i, list(acc_cols)].values
        acc_world_curr = body_to_world(acc_body_curr, quaternions[i])
        acc_world_curr -= np.array([0, 0, 9.81])  # Remove gravity

        # RK4 integration velocity
        k1_v = acc_world_prev
        k2_v = 0.5 * (acc_world_prev + acc_world_curr)
        k3_v = k2_v
        k4_v = acc_world_curr
        velocity[i] = velocity[i-1] + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        # RK4 integration position
        v_prev = velocity[i-1]
        v_curr = velocity[i]
        k1_p = v_prev
        k2_p = 0.5 * (v_prev + v_curr)
        k3_p = k2_p
        k4_p = v_curr
        position[i] = position[i-1] + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

        acc_world_prev = acc_world_curr

    return velocity, position

velocity, position = integrate_accel_to_position(df, Q, dt_column='dt',
                                                acc_cols=(ax_column, ay_column, az_column),
                                                gyr_cols=(gx_column, gy_column, gz_column))

#%% MADGWICK GRAPHS

time = df['timestamp(s)'] if 'timestamp(s)' in df.columns else np.cumsum(df['dt'].values)

plt.figure(figsize=(10, 6))
plt.plot(time, velocity[:, 0], label='Vx')
plt.plot(time, velocity[:, 1], label='Vy')
plt.plot(time, velocity[:, 2], label='Vz')
plt.title("Velocity Components Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(position[:, 0], position[:, 1], position[:, 2], label='Trajectory')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Position Trajectory')
ax.legend()
plt.show()

#%% INTEGRATION FUNCTIONS

def body_to_world(acc_body, quat_wxyz):
    """
    acc_body  : numpy (3,)  – Ax,Ay,Az in sensor/body frame
    quat_wxyz : numpy (4,)  – current orientation quaternion (w,x,y,z)
    returns   : numpy (3,)  – acceleration expressed in world (ENU) axes
    """
    return R.from_quat(np.roll(quat_wxyz, -1)).apply(acc_body)  # convert to x,y,z,w for scipy

acc_body_0 = df.loc[0, [ax_column, ay_column, az_column]].values
acc_world_prev = body_to_world(acc_body_0, quaternions[0])

for i in range(1, len(df)):
    dt = df['dt'].iloc[i]

    # -----------------------------------------------------------------
    # 1) Orientation update
    # -----------------------------------------------------------------
    omega = df.loc[i, [gx_column, gy_column, gz_column]].values
    omega_mag = np.linalg.norm(omega)

    if omega_mag > 1e-8:
        theta = omega_mag * dt
        axis  = omega / omega_mag
        delta_q = R.from_rotvec(axis * theta).as_quat()   # x,y,z,w
        delta_q = np.roll(delta_q, 1)                     # w,x,y,z

        q_prev  = quaternions[i-1]
        w1, x1, y1, z1 = delta_q
        w2, x2, y2, z2 = q_prev
        q_new = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        quaternions[i] = q_new / np.linalg.norm(q_new)
    else:
        quaternions[i] = quaternions[i-1]

    acc_body_curr   = df.loc[i, [ax_column, ay_column, az_column]].values

    # Transform to world frame
    acc_world_curr  = body_to_world(acc_body_curr, quaternions[i])

    # Subtract gravity
    g_world = np.array([0, 0, 9.81])
    acc_world_curr -= g_world
    
    # ---- velocity ----
    k1_v = acc_world_prev
    k2_v = 0.5 * (acc_world_prev + acc_world_curr)
    k3_v = k2_v
    k4_v = acc_world_curr
    velocity[i] = velocity[i-1] + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    # ---- position ----
    v_prev = velocity[i-1]
    v_curr = velocity[i]
    k1_p = v_prev
    k2_p = 0.5 * (v_prev + v_curr)
    k3_p = k2_p
    k4_p = v_curr
    position[i] = position[i-1] + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

    acc_world_prev = acc_world_curr

#%% POST PROCESS

# Convert quaternion orientation to Euler angles (roll, pitch, yaw) for visualization
quaternions_xyzw = np.roll(quaternions, -1, axis=1)
rot_objs = R.from_quat(quaternions_xyzw)
euler_angles = rot_objs.as_euler('xyz', degrees=True)  # roll, pitch, yaw

# Store results
df['vx'], df['vy'], df['vz'] = velocity.T
df['x'], df['y'], df['z'] = position.T
df['roll'], df['pitch'], df['yaw'] = euler_angles.T

# Check data
print(df[['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']].head(20))

#%% GRAPHS
# Static plot
fig_static = plt.figure(figsize=(10, 8))
ax_static = fig_static.add_subplot(111, projection='3d')
ax_static.plot(df['x'], df['y'], df['z'])
ax_static.set_title('Static Trajectory')
ax_static.set_xlabel('X (m)')
ax_static.set_ylabel('Y (m)')
ax_static.set_zlabel('Z (m)')
plt.show()

# Plot roll, pitch, yaw vs time
plt.figure(figsize=(10, 8))
plt.plot(df['time'], df['roll'], label='Roll (deg)')
plt.plot(df['time'], df['pitch'], label='Pitch (deg)')
plt.plot(df['time'], df['yaw'], label='Yaw (deg)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Roll, Pitch, and Yaw over Time')
plt.legend()
plt.grid(True)
plt.show()

# 3D Orientation Visualization
fig_orientation = plt.figure(figsize=(10, 8))
ax_orientation = fig_orientation.add_subplot(111, projection='3d')
ax_orientation.set_xlim([-1.5, 1.5])
ax_orientation.set_ylim([-1.5, 1.5])
ax_orientation.set_zlim([-1.5, 1.5])
ax_orientation.set_title('3D Orientation (Roll-Pitch-Yaw)')
ax_orientation.set_xlabel('X')
ax_orientation.set_ylabel('Y')
ax_orientation.set_zlabel('Z')

origin = np.array([0, 0, 0])

quiver_obj = []

def update_orientation(frame):
    global quiver_obj

    # Remove previous quivers
    for artist in quiver_obj:
        artist.remove()
    quiver_obj = []

    # Euler angles
    rot = R.from_euler('xyz', [df['roll'].iloc[frame], df['pitch'].iloc[frame], df['yaw'].iloc[frame]], degrees=True)
    x_axis = rot.apply([1, 0, 0])
    y_axis = rot.apply([0, 1, 0])
    z_axis = rot.apply([0, 0, 1])

    # Draw new orientation frame
    qx = ax_orientation.quiver(*origin, *x_axis, color='r', label='X-axis' if frame == 0 else None)
    qy = ax_orientation.quiver(*origin, *y_axis, color='g', label='Y-axis' if frame == 0 else None)
    qz = ax_orientation.quiver(*origin, *z_axis, color='b', label='Z-axis' if frame == 0 else None)

    quiver_obj = [qx, qy, qz]
    return quiver_obj

# Animation update
ani_orientation = FuncAnimation(fig_orientation, update_orientation, frames=range(0, len(df), 1),
                                blit=False, interval=100, repeat=False)

plt.legend()
plt.show()
#%% GRAPHS
# ------------------------------------------------------------------
#  ANIMATION : trajectoire 3D + orientation 3D + R/P/Y temps réel
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7))
ax_traj = fig.add_subplot(121, projection='3d')
ax_ori  = fig.add_subplot(122, projection='3d')

# — Trajectoire (subplot gauche)
ax_traj.set_title('Trajectoire 3D', y=0.98, fontsize=15, fontweight='bold')
ax_traj.set_xlabel('X'); ax_traj.set_ylabel('Y'); ax_traj.set_zlabel('Z')
ax_traj.set_xlim(df['x'].min(), df['x'].max())
ax_traj.set_ylim(df['y'].min(), df['y'].max())
ax_traj.set_zlim(df['z'].min(), df['z'].max())
traj_line, = ax_traj.plot([], [], [], lw=2, color='steelblue')

# — Orientation (subplot droit)
ax_ori.set_title('Orientation IMU', y= 0.98, fontsize=15, fontweight='bold')
ax_ori.set_xlabel('X'); ax_ori.set_ylabel('Y'); ax_ori.set_zlabel('Z')
ax_ori.set_xlim([-1.2, 1.2]); ax_ori.set_ylim([-1.2, 1.2]); ax_ori.set_zlim([-1.2, 1.2])
origin = np.array([0, 0, 0])

arrow_elems = []                   # pour effacer/redessiner
rot_list = R.from_quat(quaternions_xyzw)

# — Texte R/P/Y (placé sur toute la figure, coin haut gauche)
txt_rpy = fig.text(0.48, 0.90, '', fontsize=12, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    global arrow_elems

    # --- ligne trajectoire ---
    traj_line.set_data(df['x'][:frame+1], df['y'][:frame+1])
    traj_line.set_3d_properties(df['z'][:frame+1])

    # --- repère orientation ---
    for art in arrow_elems:
        art.remove()
    arrow_elems = []
    rot = rot_list[frame]
    L = 0.9
    x_vec, y_vec, z_vec = rot.apply([[1,0,0], [0,1,0], [0,0,1]])
    qx = ax_ori.quiver(*origin, *(L*x_vec), color='r')
    qy = ax_ori.quiver(*origin, *(L*y_vec), color='g')
    qz = ax_ori.quiver(*origin, *(L*z_vec), color='b')
    arrow_elems = [qx, qy, qz]

    # --- texte roll/pitch/yaw ---
    roll  = df['roll'].iloc[frame]
    pitch = df['pitch'].iloc[frame]
    yaw   = df['yaw'].iloc[frame]
    txt_rpy.set_text(f"t = {df['time'].iloc[frame]:.2f} s\n"
                     f"Roll  : {roll:+6.1f}°\n"
                     f"Pitch : {pitch:+6.1f}°\n"
                     f"Yaw   : {yaw:+6.1f}°")

    # retourner tout ce qui change
    return [traj_line, txt_rpy] + arrow_elems

ani = FuncAnimation(fig, update,
                    frames=len(df), interval=120,
                    blit=False, repeat=False)

plt.tight_layout()
plt.show()
#%% GRAPHS
# ------------------------------------------------------------------
#  PLOT : Vx, Vy, Vz en fonction du temps
# ------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['vx'], label='Vx (m/s)', color='tab:blue')
plt.plot(df['time'], df['vy'], label='Vy (m/s)', color='tab:green')
plt.plot(df['time'], df['vz'], label='Vz (m/s)', color='tab:red')
plt.xlabel('Temps (s)')
plt.ylabel('Vitesse (m/s)')
plt.title('Composantes de vitesse en fonction du temps')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
