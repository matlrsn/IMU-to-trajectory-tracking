import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Chemin vers le fichier Excel
current_dir = 'C:/Users/mathi/OneDrive/Documents/Toulouse/Stages_Travail/eNova/G_SAT/IMU2Track'
file_name = 'test_marche.xlsx'
file_path = os.path.join(current_dir, file_name)

# Lire le fichier Excel
df = pd.read_excel(file_path)

# Utiliser les noms de colonnes exacts après inspection
x_column = 'Ax'
y_column = 'Ay'
z_column = 'Az'
wx_column = 'Gx'  
wy_column = 'Gy'  
wz_column = 'Gz'  

# Vérifiez les colonnes et ajustez si nécessaire
required_columns = [x_column, y_column, z_column, wx_column, wy_column, wz_column]
if not all(col in df.columns for col in required_columns):
    raise ValueError("Les noms des colonnes ne correspondent pas. Veuillez vérifier les noms des colonnes dans le fichier Excel.")

# Convertir les colonnes en numérique
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convertir les accélérations de g à m/s^2


# Convertir les vitesses angulaires de degrés à radians
#df[wx_column] = np.deg2rad(df[wx_column])
#df[wy_column] = np.deg2rad(df[wy_column])
#df[wz_column] = np.deg2rad(df[wz_column])

# Définir l'intervalle de temps en secondes
time_interval = 0.1
df['time'] = np.arange(0, len(df) * time_interval, time_interval)

# Utiliser 'time' pour les différences de temps
df['dt'] = df['time'].diff().fillna(0)

# Initialiser les vitesses et positions
velocity = np.zeros((len(df), 3))
position = np.zeros((len(df), 3))
orientation = np.zeros((len(df), 3))  # Pour stocker l'orientation angulaire

# Calculer les vitesses, positions et orientation
for i in range(1, len(df)):
    dt = df['dt'].iloc[i]
    # Intégrer pour obtenir la vitesse
    velocity[i] = velocity[i-1] + np.array([df[x_column].iloc[i], df[y_column].iloc[i], df[z_column].iloc[i]]) * dt
    # Intégrer pour obtenir la position
    position[i] = position[i-1] + velocity[i] * dt
    # Intégrer pour obtenir l'orientation angulaire
    orientation[i] = orientation[i-1] + np.array([df[wx_column].iloc[i], df[wy_column].iloc[i], df[wz_column].iloc[i]]) * dt

# Ajouter les résultats au DataFrame
df['vx'], df['vy'], df['vz'] = velocity.T
df['x'], df['y'], df['z'] = position.T
df['ox'], df['oy'], df['oz'] = orientation.T

# Vérifier les données
print(df[['time' , 'x', 'y', 'z','vx', 'vy', 'vz', 'ox', 'oy', 'oz']].head(51))
print(df['time'].head(51))

# Tracer un graphique statique pour vérifier les données
fig_static = plt.figure()
ax_static = fig_static.add_subplot(111, projection='3d')
ax_static.plot(df['x'], df['y'], df['z'])
ax_static.set_title('Trajectoire statique')
ax_static.set_xlabel('Position X (m)')
ax_static.set_ylabel('Position Y (m)')
ax_static.set_zlabel('Position Z (m)')
plt.show()

# Configuration de l'affichage
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.min(df['x']), np.max(df['x']))
ax.set_ylim(np.min(df['y']), np.max(df['y']))
ax.set_zlim(np.min(df['z']), np.max(df['z']))
ax.set_title('Animation de la trajectoire')
ax.set_xlabel('Position X (m)')
ax.set_ylabel('Position Y (m)')
ax.set_zlabel('Position Z (m)')

# Initialisation des lignes
line, = ax.plot([], [], [], label='Trajectoire (X-Y-Z)')
ax.legend()

# Fonction d'initialisation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Fonction de mise à jour
def update(num):
    line.set_data(df['x'][:num+1], df['y'][:num+1])
    line.set_3d_properties(df['z'][:num+1])
    return line,

# Animation avec intervalle et saut de frames ajustés
skip = 10  # Sauter des frames pour accélérer
ani = FuncAnimation(fig, update, frames=range(0, len(df), skip), init_func=init, blit=False, repeat=False, interval=1)

# Afficher l'animation
plt.show()