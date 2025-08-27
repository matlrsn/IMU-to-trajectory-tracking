import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy import signal

# Simulation de la collecte de données d'un accéléromètre et d'un gyroscope
def collect_sensor_data():
# Simuler des valeurs aléatoires pour l'accéléromètre et le gyroscope
    ax, ay, az = [random.uniform(-10, 10) for _ in range(3)]
    gx, gy, gz = [random.uniform(-10, 10) for _ in range(3)]
    timestamp = time.time()
    return timestamp, ax, ay, az, gx, gy, gz

# Initialisation d'un DataFrame pour stocker les donnees
columns = ['timestamp', 'ax' , 'ay' , 'az' , 'gx' , 'gy' , 'gz' ]
data = pd. DataFrame(columns=columns)

# Collecter des données pendant un certain temps (par exemple 10 secondes)
start_time = time. time()
duration = 100

while time.time() - start_time < duration:
    sensor_data = collect_sensor_data()
    new_row = pd.DataFrame([sensor_data], columns=columns)
    data = pd.concat([data, new_row], ignore_index=True)
    time.sleep(0.1)  # Pause de 100 ms entre les lecture
    
# Affichage des données sous forme de tableau
print(data)

# Fonction pour appliquer un filtre passe-bas
def low_pass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# Paramètres du filtre
cutoff_frequency = 2.0 # Fréquence de coupure en Hz
sample_rate = 10.0 # Frequence d'echantillonnage en Hz (10 lectures par seconde)
'''
# Application du filtre passe-bas aux donnees de l'acceleromètre
data['ax_filtered' ] = low_pass_filter(data['ax'], cutoff_frequency, sample_rate)
data['ay_filtered'] = low_pass_filter(data['ay'], cutoff_frequency, sample_rate)
data['az_filtered' ] = low_pass_filter(data['az'], cutoff_frequency, sample_rate)
'''
# Calcul de la trajectoire
# Intégration des accélérations pour obtenir les vitesses
data['vx'] = np.cumsum(data['ax']) * (1/sample_rate)
data['vy'] = np.cumsum(data['ay']) * (1/sample_rate)
data['vz' ] = np.cumsum(data['az']) * (1/sample_rate)

# Intégration des vitesses pour obtenir les positions
data['x' ] = np.cumsum(data['vx']) * (1/sample_rate)
data['y'] = np.cumsum(data['vy']) * (1/sample_rate)
data['z' ] = np.cumsum(data['vz']) * (1/sample_rate)

# Tracé des positions intégrées
plt.subplot(4, 1,4)
plt.plot(data['timestamp'], data['x'], label='x')
plt.plot(data['timestamp' ], data['y'], label='y' )
plt.plot(data['timestamp'], data['z'], label='z')
plt.title('Trajectoire Intégrée')
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.legend()

# Tracé 3D de la trajectoire finale
fig = plt.figure(figsize=(10, 8))
ax = fig. add_subplot(111, projection='3d' )
ax.plot(data['x'], data['y'], data['z'], label='Trajectoire' )
ax. set_title('Trajectoire 3D de 1\'Objet')
ax.set_xlabel('Position X (m)')
ax.set_ylabel('Position Y (m)')
ax.set_zlabel('Position Z (m)')
ax. legend ()
plt. show()