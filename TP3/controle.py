import numpy as np
from math import floor
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt

###########################################################
# CRAMPETTE
# Olivier
# TP 3
###########################################################

def lecture(fichier, nb_bits):
    fs, signal = wav.read(fichier)
    signal = signal / (2 ** (nb_bits - 1))  # Normalisation
    duree = len(signal) / fs  # Durée en secondes
    return signal, fs, duree

def parametrisation(signal, taille_fenetre, nbe_coef):
    recouvrement = floor(taille_fenetre / 2)
    nb_fen = floor((np.size(signal) - taille_fenetre) / recouvrement) + 1
    mfcc = np.zeros((nb_fen, nbe_coef))
    
    for fen in range(nb_fen):
        p = fen * recouvrement
        spectre = abs(np.fft.fft(np.multiply(signal[p:p + taille_fenetre], np.hamming(taille_fenetre))))
        cepstre = np.fft.fft(np.log(spectre))
        mfcc[fen, :] = cepstre[1:nbe_coef+1].real
    
    return mfcc

def parametrisation_total(nb_bits, taille_fenetre, nbe_coef, obs_rep):
    mfcc_dict = {}
    for fichier in os.listdir(obs_rep):
        if fichier.endswith('.wav'):
            chemin_fichier = os.path.join(obs_rep, fichier)
            signal, fs, duree = lecture(chemin_fichier, nb_bits)
            mfcc = parametrisation(signal, taille_fenetre, nbe_coef)
            mfcc_dict[fichier] = mfcc
    return mfcc_dict

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def dtw(mfcc_ref, mfcc_obs):
    n_ref, _ = mfcc_ref.shape
    n_obs, _ = mfcc_obs.shape
    dtw_matrix = np.zeros((n_ref, n_obs))
    
    dtw_matrix[0, 0] = euclidean_distance(mfcc_ref[0], mfcc_obs[0])
    
    for i in range(1, n_ref):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + euclidean_distance(mfcc_ref[i], mfcc_obs[0])
    for j in range(1, n_obs):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + euclidean_distance(mfcc_ref[0], mfcc_obs[j])
    
    for i in range(1, n_ref):
        for j in range(1, n_obs):
            cost = euclidean_distance(mfcc_ref[i], mfcc_obs[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n_ref-1, n_obs-1] / (n_ref + n_obs)

def plot_dtw_path(mfcc_ref, mfcc_obs):
    n_ref, _ = mfcc_ref.shape
    n_obs, _ = mfcc_obs.shape
    dtw_matrix = np.zeros((n_ref, n_obs))
    
    # Calcul de la matrice DTW
    dtw_matrix[0, 0] = euclidean_distance(mfcc_ref[0], mfcc_obs[0])
    for i in range(1, n_ref):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + euclidean_distance(mfcc_ref[i], mfcc_obs[0])
    for j in range(1, n_obs):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + euclidean_distance(mfcc_ref[0], mfcc_obs[j])
    for i in range(1, n_ref):
        for j in range(1, n_obs):
            cost = euclidean_distance(mfcc_ref[i], mfcc_obs[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    
    # Rétropropagation pour trouver le meilleur chemin
    i, j = n_ref - 1, n_obs - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            if min_val == dtw_matrix[i-1, j]:
                i -= 1
            elif min_val == dtw_matrix[i, j-1]:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()
    
    # Affichage de la matrice DTW et du meilleur chemin
    plt.figure(figsize=(10, 6))
    plt.imshow(dtw_matrix, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Coût DTW')
    plt.plot([p[1] for p in path], [p[0] for p in path], color='red', linewidth=2, label='Meilleur chemin')
    plt.xlabel('Observation (test2.wav)')
    plt.ylabel('Référence')
    plt.title('Matrice DTW et meilleur chemin')
    plt.legend()
    plt.show()

def DTW_total(nb_bits, taille_fenetre, nbe_coef, rep_mfcc_ref, rep_wave_obs):
    mfcc_dict_ref = parametrisation_total(nb_bits, taille_fenetre, nbe_coef, rep_mfcc_ref)
    mfcc_dict_obs = parametrisation_total(nb_bits, taille_fenetre, nbe_coef, rep_wave_obs)
    
    cost_matrix = np.zeros((len(mfcc_dict_obs), len(mfcc_dict_ref)))
    
    for i, (obs_name, mfcc_obs) in enumerate(mfcc_dict_obs.items()):
        for j, (ref_name, mfcc_ref) in enumerate(mfcc_dict_ref.items()):
            cost_matrix[i, j] = dtw(mfcc_ref, mfcc_obs)
    
    return cost_matrix, mfcc_dict_obs, mfcc_dict_ref

# Paramètres
nb_bits = 16
taille_fenetre = 1024
nbe_coef = 16
rep_ref = '/home/python/PW_speech_processing/TP3/SIGNAL/REF'
rep_obs = '/home/python/PW_speech_processing/TP3/SIGNAL/OBS'

# Calcul de la matrice des coûts
cost_matrix, mfcc_dict_obs, mfcc_dict_ref = DTW_total(nb_bits, taille_fenetre, nbe_coef, rep_ref, rep_obs)

# Affichage des scores DTW
print("Scores DTW entre les observations et les références :")
for i, obs_name in enumerate(mfcc_dict_obs.keys()):
    for j, ref_name in enumerate(mfcc_dict_ref.keys()):
        print(f"{obs_name} vs {ref_name} : {cost_matrix[i, j]:.4f}")

# Affichage des coûts sous forme d'histogramme en matrice 5x4
n_obs = len(mfcc_dict_obs)
n_rows, n_cols = 5, 4
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 10))
axes = axes.flatten()

for i, (obs_name, row_cost) in enumerate(zip(mfcc_dict_obs.keys(), cost_matrix)):
    ax = axes[i]
    ax.bar(range(len(row_cost)), row_cost, color='blue')
    ax.set_xlabel('Référence')
    ax.set_ylabel('Coût DTW')
    ax.set_title(f'Obs {obs_name}')
    ax.set_xticks(range(len(row_cost)))
    ax.set_xticklabels(mfcc_dict_ref.keys(), rotation=45)

for i in range(n_obs, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Détermination des mots les plus probables
correct_predictions = 0
for i, (obs_name, row_cost) in enumerate(zip(mfcc_dict_obs.keys(), cost_matrix)):
    min_cost_index = np.argmin(row_cost)
    predicted_word = list(mfcc_dict_ref.keys())[min_cost_index]
    print(f"Observation {obs_name} : Mot prédit = {predicted_word} (Coût = {row_cost[min_cost_index]:.4f})")
    
    if obs_name.split('.')[0] == predicted_word.split('.')[0]:
        correct_predictions += 1

# Calcul et affichage du score de reconnaissance
score = (correct_predictions / len(mfcc_dict_obs)) * 100
print(f"Score de reconnaissance : {score:.2f}%")

# Affichage du meilleur chemin pour le fichier reconnu
for obs_name, mfcc_obs in mfcc_dict_obs.items():
    min_cost_index = np.argmin(cost_matrix[i])
    ref_name = list(mfcc_dict_ref.keys())[min_cost_index]
    mfcc_ref = mfcc_dict_ref[ref_name]
    print(f"Affichage du meilleur chemin pour {obs_name} vs {ref_name}")
    plot_dtw_path(mfcc_ref, mfcc_obs)