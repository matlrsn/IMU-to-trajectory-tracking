# =============================================================
# IMU2Track GUI
# =============================================================
# Requirements:
#   pip install pandas numpy matplotlib customtkinter scipy
#
# This GUI lets you:
#   1. Choose an Excel file containing IMU logs (Ax,Ay,Az,Gx,Gy,Gz).
#   2. Run RK4 integration for position and velocity.
#   3. Integrate angular velocity with quaternions.
#   4. Visualise results:
#        - 3D trajectory
#        - Roll/Pitch/Yaw vs time
#        - Optional 3D orientation animation (opens a matplotlib window)
#
# Gravity compensation & sensor fusion are assumed done in firmware.
# =============================================================

import os
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

ctk.set_default_color_theme("dark-blue")
ctk.set_appearance_mode("System")  # "Light", "Dark", or "System"

TIME_INTERVAL = 0.1  # seconds per sample (edit if needed)

# -------------------------------------------------------------
# Core processing functions
# -------------------------------------------------------------

def rk4_process(df: pd.DataFrame) -> pd.DataFrame:
    """Run RK4 + quaternion orientation on an IMU dataframe.
    Assumes columns: Ax,Ay,Az,Gx,Gy,Gz (deg/s)"""

    # Ensure numeric
    df = df.astype(float)

    # Time vector
    df["time"] = np.arange(len(df)) * TIME_INTERVAL
    df["dt"] = df["time"].diff().fillna(0)

    # Allocate
    velocity = np.zeros((len(df), 3))
    position = np.zeros((len(df), 3))
    quats = np.zeros((len(df), 4))
    quats[0] = np.array([1, 0, 0, 0])

    # Integration loop
    for i in range(1, len(df)):
        dt = df["dt"].iloc[i]

        # ---- RK4 on linear motion ----
        a_prev = df.loc[i - 1, ["Ax", "Ay", "Az"]].values
        a_curr = df.loc[i, ["Ax", "Ay", "Az"]].values
        k1_v = a_prev
        k2_v = (a_prev + a_curr) / 2
        k3_v = k2_v
        k4_v = a_curr
        velocity[i] = velocity[i - 1] + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        v_prev, v_curr = velocity[i - 1], velocity[i]
        k1_p = v_prev
        k2_p = (v_prev + v_curr) / 2
        k3_p = k2_p
        k4_p = v_curr
        position[i] = position[i - 1] + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)

        # ---- Quaternion orientation ----
        omega = np.deg2rad(df.loc[i, ["Gx", "Gy", "Gz"]].values)
        omega_mag = np.linalg.norm(omega)
        if omega_mag > 1e-8:
            theta = omega_mag * dt
            axis = omega / omega_mag
            d_quat_xyzw = R.from_rotvec(axis * theta).as_quat()
            d_quat_wxyz = np.roll(d_quat_xyzw, 1)

            # Quaternion multiply (delta * prev)
            w1, x1, y1, z1 = d_quat_wxyz
            w2, x2, y2, z2 = quats[i - 1]
            q_new = np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])
            quats[i] = q_new / np.linalg.norm(q_new)
        else:
            quats[i] = quats[i - 1]

    # Euler extraction
    quats_xyzw = np.roll(quats, -1, axis=1)
    euler = R.from_quat(quats_xyzw).as_euler("xyz", degrees=True)

    # Assemble into DataFrame
    df_out = df.copy()
    df_out[["vx", "vy", "vz"]] = velocity
    df_out[["x", "y", "z"]] = position
    df_out[["roll", "pitch", "yaw"]] = euler
    return df_out

# -------------------------------------------------------------
# GUI class
# -------------------------------------------------------------

class IMUGui(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IMU2Track Visualiser")
        self.geometry("1000x700")

        # File selection frame
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", padx=10, pady=10)
        self.file_entry = ctk.CTkEntry(file_frame, width=600)
        self.file_entry.pack(side="left", padx=5)
        browse_btn = ctk.CTkButton(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side="left", padx=5)
        load_btn = ctk.CTkButton(file_frame, text="Load & Process", command=self.load_process)
        load_btn.pack(side="left", padx=5)

        # Notebook for plots
        self.notebook = ctk.CTkTabview(self, segmented_button_selected_color="#3a7ebf")
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.traj_tab = self.notebook.add("Trajectoire 3D")
        self.rpy_tab = self.notebook.add("Angles R/P/Y")

        # Placeholders for canvas
        self.traj_canvas = None
        self.rpy_canvas = None

        self.df = None  # processed data

    # ---------- GUI callbacks ----------

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, file_path)

    def load_process(self):
        path = self.file_entry.get()
        if not os.path.isfile(path):
            messagebox.showerror("Erreur", "Fichier introuvable")
            return
        try:
            raw_df = pd.read_excel(path).dropna(subset=["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]).reset_index(drop=True)
            self.df = rk4_process(raw_df)
        except Exception as e:
            messagebox.showerror("Erreur au traitement", str(e))
            return
        messagebox.showinfo("Succès", "Traitement terminé !")
        self.draw_plots()

    # ---------- Plotting ----------

    def draw_plots(self):
        if self.df is None:
            return
        # --- 3D trajectory ---
        if self.traj_canvas:
            self.traj_canvas.get_tk_widget().destroy()
        fig_traj = plt.figure(figsize=(5, 4))
        ax = fig_traj.add_subplot(111, projection="3d")
        ax.plot(self.df["x"], self.df["y"], self.df["z"], lw=2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Trajectoire 3D")
        self.traj_canvas = FigureCanvasTkAgg(fig_traj, master=self.traj_tab)
        self.traj_canvas.draw()
        self.traj_canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Roll/Pitch/Yaw vs time ---
        if self.rpy_canvas:
            self.rpy_canvas.get_tk_widget().destroy()
        fig_rpy = plt.figure(figsize=(5, 4))
        plt.plot(self.df["time"], self.df["roll"], label="Roll")
        plt.plot(self.df["time"], self.df["pitch"], label="Pitch")
        plt.plot(self.df["time"], self.df["yaw"], label="Yaw")
        plt.xlabel("Temps (s)")
        plt.ylabel("Angle (deg)")
        plt.title("Évolution des angles")
        plt.legend()
        plt.grid(True)
        self.rpy_canvas = FigureCanvasTkAgg(fig_rpy, master=self.rpy_tab)
        self.rpy_canvas.draw()
        self.rpy_canvas.get_tk_widget().pack(fill="both", expand=True)

# -------------------------------------------------------------
# Main entry
# -------------------------------------------------------------

if __name__ == "__main__":
    app = IMUGui()
    app.mainloop()
