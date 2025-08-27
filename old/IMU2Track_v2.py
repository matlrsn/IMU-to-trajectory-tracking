# Uses RK4 for acceleration and speed integration
# Uses quaternions for angular velocity integration (useful for sensor fusion later)
# gravity compensation should now be directly integrated in the IMU S/W
# sensor fusion algorithm seems already implemented directly in g sat S/W

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

#%%
# Path to Excel file
current_dir = 'C:/Users/mathi/OneDrive/Documents/Toulouse/Stages_Travail/eNova/G_SAT/IMU2Track'
#file_name = 'IMU2Track-RotX_v2.xlsx'
file_name = 'drop_test.xlsx'
#file_name = 'test.xlsx'
#file_name = 'random_data.xlsx'
file_path = os.path.join(current_dir, file_name)

# Read Excel file
df = pd.read_excel(file_path)
# Remove rows with NaNs in accelerometer or gyroscope data
df = df.dropna(subset=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']).reset_index(drop=True)
#%%
# Column names
gx_column, gy_column, gz_column = 'Gx', 'Gy', 'Gz'
ax_column, ay_column, az_column = 'Ax', 'Ay', 'Az'

# Check for columns
#required_columns = [x_column, y_column, z_column, ax_column, ay_column, az_column, wx_column, wy_column, wz_column]
required_columns = [gx_column, gy_column, gz_column, ax_column, ay_column, az_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Missing required columns.")

# Time vector
time_interval = 1/200
#df['time'] = np.arange(0, len(df) * time_interval, time_interval)
# Activate for random data
df['time'] = np.arange(len(df)) * time_interval

df['dt'] = df['time'].diff().fillna(0)

# Init state vectors
velocity = np.zeros((len(df), 3))
position = np.zeros((len(df), 3))

# Quaternion init (w, x, y, z)
quaternions = np.zeros((len(df), 4))
quaternions[0] = np.array([1, 0, 0, 0])  # Initial quaternion (1+i0+j0+k0)

# Loop to integrate IMU data
for i in range(1, len(df)):
    dt = df['dt'].iloc[i]

    # RK4 on acceleration
    a_prev = df.loc[i-1, [ax_column, ay_column, az_column]].values
    a_curr = df.loc[i, [ax_column, ay_column, az_column]].values
    k1_v = a_prev
    k2_v = (a_prev + a_curr)*0.5
    k3_v = (a_prev + a_curr)*0.5
    k4_v = a_curr
    # Velocity calculation
    velocity[i] = velocity[i-1] + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    # RK4 on position
    v_prev = velocity[i-1]
    v_curr = velocity[i]
    k1_p = v_prev
    k2_p = (v_prev + v_curr)/2
    k3_p = (v_prev + v_curr)/2
    k4_p = v_curr
    # Position calculation
    position[i] = position[i-1] + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

    # Angular velocity for orientation update
    omega = df.loc[i, [gx_column, gy_column, gz_column]].values

    # Magnitude of angular velocity
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 1e-8:
        # Rotation quaternion over dt
        theta = omega_mag * dt
        axis = omega/omega_mag # Normalizing axis
        delta_q = R.from_rotvec(axis * theta).as_quat()  # x, y, z, w format
        delta_q = np.roll(delta_q, 1)  # Convert to w, x, y, z

        # Quaternion multiplication: q_new = delta_q*q_prev
        q_prev = quaternions[i-1]
        w1, x1, y1, z1 = delta_q
        w2, x2, y2, z2 = q_prev
        q_new = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        quaternions[i] = q_new/np.linalg.norm(q_new)  # Normalize
    else:
        quaternions[i] = quaternions[i-1] # Use previous if magnitude <= 1e-8

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
#%%
# ------------------------------------------------------------
# 1) Rotations prêtes à l'emploi
rot_list = R.from_quat(quaternions_xyzw)   # xyzw déjà calculé plus haut

# ------------------------------------------------------------
# 2) Préparation figure / axes
fig = plt.figure(figsize=(11, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title('Trajectoire 3D + repère orienté')

# Trajectoire ligne
traj_line, = ax.plot([], [], [], lw=2, color='steelblue')

# Conserver les flèches actuelles pour pouvoir les supprimer
arrow_elems = []

# Aspect 1:1:1 pour éviter l’écrasement
def set_equal(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ranges = np.array([xlim, ylim, zlim])
    centers = ranges.mean(axis=1)
    max_range = (ranges[:,1] - ranges[:,0]).max() / 2
    for ctr, axis in zip(centers, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        axis(ctr - max_range, ctr + max_range)

# ------------------------------------------------------------
# 3) Animation
def update(frame):
    global arrow_elems

    # --- mettre à jour la ligne ---
    traj_line.set_data(df['x'][:frame+1], df['y'][:frame+1])
    traj_line.set_3d_properties(df['z'][:frame+1])

    # --- retirer les flèches précédentes ---
    for artist in arrow_elems:
        artist.remove()
    arrow_elems = []

    # --- repère local depuis le quaternion ---
    pos = df[['x', 'y', 'z']].iloc[frame].values
    rot = rot_list[frame]                      # rotation directement issue du quaternion

    x_vec = rot.apply([1, 0, 0])
    y_vec = rot.apply([0, 1, 0])
    z_vec = rot.apply([0, 0, 1])

    L = 0.5   # longueur flèches
    qx = ax.quiver(*pos, *(L*x_vec), color='r')
    qy = ax.quiver(*pos, *(L*y_vec), color='g')
    qz = ax.quiver(*pos, *(L*z_vec), color='b')
    arrow_elems = [qx, qy, qz]

    return [traj_line] + arrow_elems

ani = FuncAnimation(fig, update,
                    frames=len(df), interval=120,
                    blit=False, repeat=False)

# Ajuster les bornes avant de dessiner (et aspect = 1:1)
ax.set_xlim(df['x'].min(), df['x'].max())
ax.set_ylim(df['y'].min(), df['y'].max())
ax.set_zlim(df['z'].min(), df['z'].max())
set_equal(ax)

plt.show()
#%%
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
#%%
# Plot vx, vy, vz over time
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['vx'], label='Vx (m/s)', color='tab:blue')
plt.plot(df['time'], df['vy'], label='Vy (m/s)', color='tab:green')
plt.plot(df['time'], df['vz'], label='Vz (m/s)', color='tab:red')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Components over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
