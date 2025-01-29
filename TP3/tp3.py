# TP3 : Reconnaissance de mots isolés par programmation dynamique (DTW)

# NOM - PRÉNOM : Déposer le notebook sur moodle à la fin de la séance.

import numpy as np
from math import floor, inf, dist
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from canaux24 import canaux


# LECTURE d'un fichier Wave
import numpy as np
import scipy.io.wavfile as wav

def lecture(fichier, nb_bits):
    fs, signal = wav.read(fichier)
    
    # Normalisation du signal en fonction du nombre de bits
    signal = signal / (2 ** (nb_bits - 1))  # Normalisation du signal
    
    # Calcul de la durée du fichier
    duree = len(signal) / fs  # Durée en secondes
    
    return signal, fs, duree

#Test lecture
fichier = '/home/python/PW_speech_processing/TP3/SIGNAL/OBS/alpha.wav'  # Remplacez par le chemin vers votre fichier
nb_bits = 16  
signal, fs, duree = lecture(fichier, nb_bits)

print(f"Durée du fichier : {duree} secondes")
print(f"Fréquence d'échantillonnage : {fs} Hz")
print(f"Signal (normalisé) : {signal[:10]}...")  # Affichage des 10 premiers échantillons


# Calcul de "nbe_coef" coefficients cepstraux
def parametrisation(signal, taille_fenetre, nbe_coef):
    # Initialisation de la matrice résultat
    recouvrement = floor(taille_fenetre / 2)
    nb_fen = floor((np.size(signal) - taille_fenetre) / recouvrement) + 1
    mfcc = np.zeros((nb_fen, nbe_coef))

    # Calcul des MFCC
    for fen in range(nb_fen):
        p = fen * recouvrement
        spectre = abs(np.fft.fft(np.multiply(signal[p:p + taille_fenetre], np.hamming(taille_fenetre))))
        cepstre = np.fft.fft(np.log(spectre))
        cc = cepstre[1:nbe_coef+1].real
        mfcc[fen, :] = cc

    return mfcc
    # Code de paramétrisation à compléter

taille_fenetre = 1024  # Taille de la fenêtre pour le calcul des MFCC
nbe_coef = 16  # Nombre de coefficients cepstraux à calculer
mfcc = parametrisation(signal, taille_fenetre, nbe_coef)

# Affichage des résultats de la paramétrisation
print("Exemple des coefficients cepstraux (MFCC) calculés :")
print(mfcc[:5])  # Afficher les 5 premiers vecteurs de MFCC

# Affichage graphique des MFCC
plt.figure(figsize=(10, 5))
plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='jet')
plt.title("Coefficients Cepstraux (MFCC)")
plt.colorbar()
plt.ylabel("MFCC Coefficients")
plt.xlabel("Frames")
plt.show()


# Fonction de paramétrisation totale pour tous les fichiers du répertoire
def parametrisation_total(nb_bits, taille_fenetre, nbe_coef, obs_rep):
    # Dictionnaire pour stocker les MFCC de chaque fichier
    mfcc_dict = {}
    
    # Parcours de tous les fichiers dans le répertoire obs_rep
    for fichier in os.listdir(obs_rep):
        if fichier.endswith('.wav'):
            # Chemin complet du fichier
            chemin_fichier = os.path.join(obs_rep, fichier)
            
            # Lecture du fichier audio
            signal, fs, duree = lecture(chemin_fichier, nb_bits)
            
            # Paramétrisation du fichier audio
            mfcc = parametrisation(signal, taille_fenetre, nbe_coef)
            
            # Stocker les résultats dans le dictionnaire
            mfcc_dict[fichier] = mfcc
    
    return mfcc_dict


obs_rep = '/home/python/PW_speech_processing/TP3/SIGNAL/OBS'  # Répertoire contenant les fichiers audio
ref_rep = '/home/python/PW_speech_processing/TP3/SIGNAL/REF' 
# Appel à la fonction de paramétrisation totale
mfcc_dict_obs = parametrisation_total(nb_bits, taille_fenetre, nbe_coef, obs_rep)
mfcc_dict_ref = parametrisation_total(nb_bits, taille_fenetre, nbe_coef, ref_rep)


# Programmation dynamique

# Ecrire une fonction dtw qui prend deux arguments en entrée : 
# la matrice de coefficients cepstraux du signal à reconnaître (observation) 
# et la matrice de coefficients cepstraux d'un signal de référence. 
# Cette fonction renvoie le coût normalisé.

# Calcul de la DTW entre deux vecteurs
# Fonction pour calculer la distance euclidienne entre deux vecteurs
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

# Fonction DTW pour calculer le coût entre deux matrices MFCC
def dtw(mfcc_ref, mfcc_obs):
    # Dimensions des matrices de référence et d'observation

    n_ref, n_coef = mfcc_ref.shape#nombre de fenetre du signal de ref

    n_obs, _ = mfcc_obs.shape#pareil pour obs
    
    dtw_matrix = np.zeros((n_ref, n_obs))
    



    # Initialisation du premier élément de la matrice
    dtw_matrix[0, 0] = euclidean_distance(mfcc_ref[0], mfcc_obs[0])
    
    # Remplissage de la première colonne (cumulative cost)
    for i in range(1, n_ref):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + euclidean_distance(mfcc_ref[i], mfcc_obs[0])
    
    # Remplissage de la première ligne (cumulative cost)
    for j in range(1, n_obs):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + euclidean_distance(mfcc_ref[0], mfcc_obs[j])
    
    # Remplissage du reste de la matrice de coût
    for i in range(1, n_ref):
        for j in range(1, n_obs):
            cost = euclidean_distance(mfcc_ref[i], mfcc_obs[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],   # Haut
                                          dtw_matrix[i, j-1],   # Gauche
                                          dtw_matrix[i-1, j-1]) # Diagonal
    
    # Coût normalisé
    normalized_cost = dtw_matrix[n_ref-1, n_obs-1] / (n_ref + n_obs)
    
    return normalized_cost


# Ecrire une fonction dtw_total (utilisant la fonction dtw) qui calcule la DTW 
# sur toutes les observations (fichiers Wave) du répertoire rep_obs 
# par rapport à toutes les références (fichiers MFCC) du répertoire rep_ref.
# Cette fonction renvoie une matrice de coûts de taille nb_fichier_obs x nb_fichier_ref.

# DTW sur toutes les observations par rapport à toutes les références
# Fonction DTW sur toutes les observations par rapport à toutes les références
def DTW_total(nb_bits, taille_fenetre, nbe_coef, rep_mfcc_ref, rep_wave_obs):
    # Initialisation de la matrice de coûts
    cost_matrix = []
    
    # Paramétrisation des fichiers de référence
    mfcc_dict_ref = parametrisation_total(nb_bits, taille_fenetre, nbe_coef, rep_mfcc_ref)
    
    # Paramétrisation des fichiers d'observation
    mfcc_dict_obs = parametrisation_total(nb_bits, taille_fenetre, nbe_coef, rep_wave_obs)
    
    # Comparaison de chaque observation avec chaque référence
    for obs_name, mfcc_obs in mfcc_dict_obs.items():
        row_cost = []
        for ref_name, mfcc_ref in mfcc_dict_ref.items():
            # Calcul de la distance DTW pour chaque paire observation / référence
            cost = dtw(mfcc_ref, mfcc_obs)
            row_cost.append(cost)
        cost_matrix.append(row_cost)
    
    # Conversion de la matrice de coûts en numpy array pour un traitement facile
    cost_matrix = np.array(cost_matrix)
    
    return cost_matrix



# Affichages et tests

# Ecrire un programme principal qui lance les fonctions précédentes 
# et affiche pour chaque observation (mot inconnu), le mot le plus probable.

# Initialisation
# Définir les répertoires des observations et des références
rep_ref = '/home/python/PW_speech_processing/TP3/SIGNAL/REF'  # Répertoire contenant les fichiers de référence
rep_obs = '/home/python/PW_speech_processing/TP3/SIGNAL/OBS'  # Répertoire contenant les fichiers d'observation

# Appeler la fonction DTW_total pour calculer la matrice des coûts
cost_matrix = DTW_total(nb_bits, taille_fenetre, nbe_coef, rep_ref, rep_obs)

# Affichage de la matrice des coûts
print("Matrice des coûts (DTW) entre les observations et les références :")
print(cost_matrix)

# Affichage des coûts pour chaque observation sous forme d'histogramme
for i, row_cost in enumerate(cost_matrix):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(row_cost)), row_cost, color='blue')
    plt.xlabel('Référence')
    plt.ylabel('Coût DTW')
    plt.title(f'Coûts DTW pour l\'observation {list(mfcc_dict_obs.keys())[i]}')
    plt.xticks(range(len(row_cost)), list(mfcc_dict_ref.keys()), rotation=45)
    plt.show()


# Affichage des résultats avec le mot le plus probable
for i, row_cost in enumerate(cost_matrix):
    min_cost_index = np.argmin(row_cost)
    predicted_word = list(mfcc_dict_ref.keys())[min_cost_index]
    print(f"Observation {list(mfcc_dict_obs.keys())[i]} : Mot prédit = {predicted_word} (Coût = {row_cost[min_cost_index]:.4f})")


# Paramétrisation des fichiers références
# Code de paramétrisation des fichiers de référence à compléter

# Test de la DTW
# Code du test de DTW à compléter

# DTW sur toutes les observations par rapport à chaque référence
# Code à compléter

# Affichage des couts
# Code d'affichage des coûts à compléter


# Affichage amélioré

# Ajouter à votre programme principal, un affichage des coûts entre une observation et chaque référence 
# sous forme d'histogramme via la commande bar.
# Calculer le score de reconnaissance.

# Pour chaque observation, affichage des coûts (par rapport aux références) sous forme d'histogramme
# Code d'affichage à compléter

# Affichage score final
# Code d'affichage du score final à compléter

# Autres tests
# Bien évidemment, afin d’améliorer les résultats, vous pourrez modifier :
# - le nombre de paramètres (coefficients cepstraux) : nbe_coef,
# - la taille de la fenêtre d’analyse : taille_fenetre,
# ET TESTER AVEC VOS PROPRES ENREGISTREMENTS !
