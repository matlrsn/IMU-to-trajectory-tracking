#======CHANGELOG======
# -- Toujours utiliser np.cos/np.sin plutôt que math.cos/math.sin
# -- Rz @ Ry @ Rx <=> np.dot(Rz, np.dot(Ry, Rx))
# -- Nouvelle fonction kalman_filter qui implémente un filtre de kalman simple
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation

# Chemin vers le fichier Excel
current_dir = 'C:/Users/mathi/OneDrive/Documents/Toulouse/Stages_Travail/eNova/G_SAT/IMU2Track'
file_name = 'IMU2Track-RotXComplet.xlsx'
file_path = os.path.join(current_dir, file_name)

# Lire le fichier Excel
df = pd.read_excel(file_path)

# Fonction pour lire les colonnes Arx, Ary, Arz, X, Y, Z du fichier Excel
def read_excel_columns(file_path):
   df = pd.read_excel(file_path, skiprows=0)
   TetaX = df['Arx'].values
   TetaY = df['Ary'].values
   TetaZ = df['Arz'].values
   X = df['X'].values
   Y = df['Y'].values
   Z = df['Z'].values
   TetaX = TetaX[1:]
   TetaY = TetaY[1:]
   TetaZ = TetaZ[1:]
   X = X[1:]
   Y = Y[1:]
   Z = Z[1:]
   return TetaX, TetaY, TetaZ, X, Y, Z

# Fonction pour créer les matrices de rotation Rx, Ry, Rz
def create_rotation_matrices(TetaX, TetaY, TetaZ):
   Rx = np.array([
       [1, 0, 0],
       [0, np.cos(TetaX), -np.sin(TetaX)],
       [0, np.sin(TetaX), np.cos(TetaX)]
   ])
   Ry = np.array([
       [np.cos(TetaY), 0, np.sin(TetaY)],
       [0, 1, 0],
       [-np.sin(TetaY), 0, np.cos(TetaY)]
   ])
   Rz = np.array([
       [np.cos(TetaZ), -np.sin(TetaZ), 0],
       [np.sin(TetaZ), np.cos(TetaZ), 0],
       [0, 0, 1]
   ])
   return Rx, Ry, Rz

# Fonction pour multiplier les matrices de rotation Rz, Ry, Rx
def multiply_three_matrices(Rx, Ry, Rz):
   return  Rz @ Ry @ Rx # <=> np.dot(Rz, np.dot(Ry, Rx))

def kalman_filter(positions, dt=1.0, process_variance=1e-4, measurement_variance=0.1):
    n = len(positions)
    filtered_positions = []

    # Initialisation
    x = np.array([[positions[0]], [0]])  # [position, vitesse]
    P = np.eye(2)

    F = np.array([[1, dt], [0, 1]])  # Modèle de transition
    H = np.array([[1, 0]])           # Observation de la position
    Q = process_variance * np.eye(2)
    R = np.array([[measurement_variance]])

    for z in positions:
        # Prédiction
        x = F@x
        P = F@P@F.T + Q

        # Mise à jour
        y = np.array([[z]]) - H@x
        S = H@P@H.T + R
        K = P@H.T@np.linalg.inv(S)
        x = x + K@y
        P = (np.eye(2) - K@H)@P

        filtered_positions.append(x[0, 0])

    return filtered_positions

''' old version
# Fonction principale pour traiter le fichier Excel et calculer les matrices de rotation et les positions transformées
def process_excel_file(file_path):
   TetaX_values, TetaY_values, TetaZ_values, X_values, Y_values, Z_values = read_excel_columns(file_path)
   results = []
   for TetaX, TetaY, TetaZ, X, Y, Z in zip(TetaX_values, TetaY_values, TetaZ_values, X_values, Y_values, Z_values):
       Rx, Ry, Rz = create_rotation_matrices(TetaX, TetaY, TetaZ)
       R_final = multiply_three_matrices(Rx, Ry, Rz)
       position_vector = np.array([X, Y, Z])
       transformed_position = np.dot(R_final, position_vector)
       results.append({
           'R_final': R_final,
           'transformed_position': transformed_position
       })
   return results
'''
def process_excel_file(file_path):
    TetaX_values, TetaY_values, TetaZ_values, X_values, Y_values, Z_values = read_excel_columns(file_path)
    raw_positions = []

    for TetaX, TetaY, TetaZ, X, Y, Z in zip(TetaX_values, TetaY_values, TetaZ_values, X_values, Y_values, Z_values):
        Rx, Ry, Rz = create_rotation_matrices(TetaX, TetaY, TetaZ)
        R_final = multiply_three_matrices(Rx, Ry, Rz)
        position_vector = np.array([X, Y, Z])
        transformed_position = np.dot(R_final, position_vector)
        raw_positions.append(transformed_position) #new

    raw_positions = np.array(raw_positions) #new

    # Application du filtre de Kalman sur chaque axe (new)
    X_filtered = kalman_filter(raw_positions[:, 0])
    Y_filtered = kalman_filter(raw_positions[:, 1])
    Z_filtered = kalman_filter(raw_positions[:, 2])

    results = []
    for x, y, z in zip(X_filtered, Y_filtered, Z_filtered):
        results.append({'transformed_position': np.array([x, y, z])})

    return results

# Fonction pour créer un graphique fixe des positions transformées
def plot_transformed_positions(results):
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   X_transformed = [res['transformed_position'][0] for res in results]
   Y_transformed = [res['transformed_position'][1] for res in results]
   Z_transformed = [res['transformed_position'][2] for res in results]
   ax.plot(X_transformed, Y_transformed, Z_transformed, marker='o')
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   plt.title('Positions transformées')
   plt.show()


# Fonction pour créer une animation GIF des positions transformées
def create_animation(results):
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   X_transformed = [res['transformed_position'][0] for res in results]
   Y_transformed = [res['transformed_position'][1] for res in results]
   Z_transformed = [res['transformed_position'][2] for res in results]
   line, = ax.plot([], [], [], marker='o')
   ax.set_xlim(min(X_transformed), max(X_transformed))
   ax.set_ylim(min(Y_transformed), max(Y_transformed))
   ax.set_zlim(min(Z_transformed), max(Z_transformed))
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   def update(frame):
       line.set_data(X_transformed[:frame], Y_transformed[:frame])
       line.set_3d_properties(Z_transformed[:frame])
       return line,
   ani = FuncAnimation(fig, update, frames=len(X_transformed), blit=True)
   ani.save('transformed_positions.gif', writer='imagemagick', fps=5)
   plt.show()

result_matrices = process_excel_file(file_path)
plot_transformed_positions(result_matrices)
create_animation(result_matrices)

