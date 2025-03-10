import numpy as np
from math import floor
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as em
import os




########################################################
#TP2 SYSTEME DE VERIFICATION DU LOCUTEUR
# Crampette Olivier 05/02/2025
########################################################



########################################################
# #1 - DONNEES a traiter
########################################################

# Écrire une fonction « lecture » permettant de lire un fichier son (en normalisant les échantillons 
# entre -1 et 1, grâce au nombre de bits de quantification), de connaître sa fréquence d’échantillonnage 
# et sa durée. Le nombre de bits de quantification de nos fichiers est 16.

########################################################

def lecture(fichier, nb_bits=16):
    """
    Lit un fichier audio WAV, normalise les échantillons et renvoie la fréquence d'échantillonnage et la durée.
    
    :param fichier: Chemin du fichier audio WAV
    :param nb_bits: Nombre de bits de quantification (par défaut 16)
    :return: tuple (fréquence d'échantillonnage, durée, signal normalisé)
    """
    # Lire le fichier audio
    freq_echantillonnage, signal = wav.read(fichier)
    
    # Normalisation du signal entre -1 et 1
    amplitude_max = 2**(nb_bits - 1)
    signal_normalise = signal.astype(np.float32) / amplitude_max
    
    # Calcul de la durée du fichier audio
    duree = len(signal) / freq_echantillonnage
    
    return freq_echantillonnage, duree, signal_normalise


########################################################
    # 2 - Decomposition Parole/non parole #
########################################################


def energie(signal, taille_fenetre):
    """
    Calcule l'énergie d'un signal sur des fenêtres glissantes avec recouvrement.

    :param signal: Signal d'entrée à analyser (array-like)
    :param taille_fenetre: Taille de chaque fenêtre (int)
    :return: Tableau contenant l'énergie de chaque fenêtre (array-like)
    """

    # Initialisation du vecteur résultat
    recouvrement = floor(taille_fenetre / 2)
    nb_fen = floor((np.size(signal) - taille_fenetre) / recouvrement) + 1
    nrj_res = np.zeros(nb_fen)

    # Calcul de l’énergie
    for fen in range(nb_fen):
        p = fen * recouvrement
        nrj_res[fen] = np.sum((signal[p:p + taille_fenetre]) ** 2) / taille_fenetre

    return nrj_res


def etiquetage(signal, taille_fenetre, seuil):
    """
    Étiquette un signal en parole (1) ou non-parole (0) en fonction de l'énergie dans chaque fenêtre.
    
    :param signal: Signal d'entrée à analyser (array-like)
    :param taille_fenetre: Taille de chaque fenêtre (int)
    :param seuil: Seuil d'énergie pour distinguer parole de non-parole (float)
    :return: Tableau contenant des étiquettes (1 pour parole, 0 pour non-parole)
    """
    # Calcul de l'énergie du signal
    energie_signal = energie(signal, taille_fenetre)
    
    # Étiquetage : parole si l'énergie est au-dessus du seuil, non-parole sinon
    etiquettes = (energie_signal > seuil).astype(int)
    
    return etiquettes

##########################################
    # Test sur 1 seul fichier #
#########################################
# Constantes
print("TEST SUR 1 FICHIER :\n")
taille_fenetre = 1024
nb_locuteur = 10
nb_fic_app = 8
seuil = 0.0001
q = 16
nb_MFCC = 8
locuteur_cible = 3
nbe_gauss = 1

# Lecture d'un fichier
fe,duree,signal = lecture('/home/python/PW_speech_processing/TP2/APP/L1_fic1.wav', q)

# Affichage du signal audio et des valeurs
# plt.figure(1)
# plt.plot(np.arange(len(signal)) / fe, signal)
# plt.show()
print('Fe =', fe, 'Hz et durée =', duree, 's')


# Calcul de l'énergie d'un fichier
nrj_res = energie(signal, taille_fenetre)

# Affichage du signal et de l'énergie
# plt.figure(2)
# plt.subplot(211)
# plt.plot(np.arange(len(signal)) / fe, signal)
# plt.subplot(212)
# plt.plot(nrj_res)
# plt.show()

# Etiquetage d'un fichier
etiq = etiquetage(signal, taille_fenetre, seuil)

# Affichage du signal et de l'énergie
plt.figure(3)
plt.subplot(311)
plt.plot(np.arange(len(signal)) / fe, signal)
plt.subplot(312)
plt.plot(nrj_res)
plt.subplot(313)
plt.plot(etiq, '.')
plt.show()


def etiquetage_total(chemin_app, chemin_labels, nb_bits, taille_fenetre, seuil):
    """
    Étiquette tous les fichiers audio dans le répertoire 'chemin_app' en parole/non-parole 
    et enregistre les résultats dans des fichiers '.lab' dans le répertoire 'chemin_labels'.
    
    :param chemin_app: Répertoire contenant les fichiers audio WAV
    :param chemin_labels: Répertoire où enregistrer les fichiers d'étiquettes '.lab'
    :param nb_bits: Nombre de bits de quantification pour la normalisation
    :param taille_fenetre: Taille des fenêtres pour le calcul de l'énergie
    :param seuil: Seuil d'énergie pour distinguer parole de non-parole
    """
    # Vérifier si le répertoire LABELS existe, sinon exception
    if not os.path.exists(chemin_labels):
        raise FileNotFoundError(f"Le répertoire '{chemin_labels}' n'existe pas. Veuillez créer le répertoire avant d'exécuter le script.")


    fichiers_audio = [f for f in os.listdir(chemin_app) if f.endswith('.wav')]

    for fichier in fichiers_audio:
        chemin_fichier = os.path.join(chemin_app, fichier)
        
        freq, duree, signal_normalise = lecture(chemin_fichier, nb_bits)
        
        etiquettes = etiquetage(signal_normalise, taille_fenetre, seuil)
        
        # Construire le nom du fichier .lab correspondant
        nom_fichier_lab = os.path.splitext(fichier)[0] + '.lab'
        chemin_fichier_lab = os.path.join(chemin_labels, nom_fichier_lab)
        
        # Sauvegarder les étiquettes dans un fichier txt
        np.savetxt(chemin_fichier_lab, etiquettes, fmt='%d')

        print(f"Étiquetage effectué pour {fichier} -> {nom_fichier_lab}")



##########################################
    # Test sur n  fichiers #
#########################################

# Définir les chemins des répertoires
chemin_app = "/home/python/PW_speech_processing/TP2/APP"         # Exemple : "/home/user/APP"
chemin_labels = "/home/python/PW_speech_processing/TP2/LABELS"   # Exemple : "/home/user/LABELS"

# Appliquer l'étiquetage sur tous les fichiers du répertoire 'APP'
# etiquetage_total(chemin_app=chemin_app, chemin_labels=chemin_labels, nb_bits=q, taille_fenetre=1024, seuil=seuil)



##########################################
    # 3 - PARAMETRISATION #
#########################################

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


def parametrisation_total(nb_bits, taille_fenetre, nbe_coef, nbe_loc, nbe_fic, chemin_app, chemin_mfcc):
    """
    Calcule les MFCC pour tous les fichiers audio dans le répertoire 'chemin_app'
    et enregistre les résultats dans des fichiers '.mfcc' dans le répertoire 'chemin_mfcc'.
    
    :param nb_bits: Nombre de bits de quantification pour la normalisation
    :param taille_fenetre: Taille des fenêtres pour le calcul de l'énergie
    :param nbe_coef: Nombre de coefficients MFCC à calculer
    :param nbe_loc: Nombre de locuteurs (non utilisé ici mais pourrait être pertinent pour l'organisation des fichiers)
    :param nbe_fic: Nombre de fichiers par locuteur (non utilisé ici mais pourrait être pertinent pour l'organisation des fichiers)
    :param chemin_app: Répertoire contenant les fichiers audio WAV
    :param chemin_mfcc: Répertoire où enregistrer les fichiers MFCC
    """
    # Vérifier si le répertoire MFCC existe, sinon exception
    if not os.path.exists(chemin_mfcc):
        raise FileNotFoundError(f"Le répertoire '{chemin_mfcc}' n'existe pas. Veuillez créer le répertoire avant d'exécuter le script.")
    
    fichiers_audio = [f for f in os.listdir(chemin_app) if f.endswith('.wav')]

    for fichier in fichiers_audio:
        chemin_fichier = os.path.join(chemin_app, fichier)
        
        # Lecture du fichier audio
        freq, duree, signal_normalise = lecture(chemin_fichier, nb_bits)
        
        # Calcul des MFCC
        mfcc = parametrisation(signal_normalise, taille_fenetre, nbe_coef)
        
        # Construire le nom du fichier .mfcc 
        nom_fichier_mfcc = os.path.splitext(fichier)[0] + '.mfcc'
        chemin_fichier_mfcc = os.path.join(chemin_mfcc, nom_fichier_mfcc)
        
        # Sauvegarder
        np.savetxt(chemin_fichier_mfcc, mfcc)
        
        print(f"MFCC calculés et enregistrés pour {fichier} -> {nom_fichier_mfcc}")

################################
# TEST PARAMETRISATION TOTALE #
################################


chemin_app = "/home/python/PW_speech_processing/TP2/APP"  
chemin_mfcc = "/home/python/PW_speech_processing/TP2/MFCC"
# parametrisation_total(q, taille_fenetre, nb_MFCC, nb_locuteur, nb_fic_app,chemin_app=chemin_app,chemin_mfcc=chemin_mfcc)



################################
    # 4 - APPRENTISSAGE #
################################


# Affectation des coefficients cepstraux du répertoire "MFCC" suivant les labels
def affectation(REP_LAB, REP_MFCC, nbe_coef, nbe_loc, nbe_fic, locuteur):
    # Initialisation de "param" (MFCC du "monde")
    param = np.empty((0, nbe_coef))

    # Boucle sur tous les fichiers
    for x in range(1, nbe_loc+1):
        # Initialisation de "param_indices" (MFCC du "locuteur")
        param_loc = np.empty((0, nbe_coef))

        for y in range(1, nbe_fic+1):
            # Nom des fichiers LAB et MFCC
            fichier_lab = REP_LAB + '/L' + str(x) + '_fic' + str(y) + '.lab'
            fichier_mfcc = REP_MFCC + '/L' + str(x) + '_fic' + str(y) + '.mfcc'

            # Lecture des 2 fichiers
            lab = np.loadtxt(fichier_lab)
            mfcc = np.loadtxt(fichier_mfcc)

            # Vérification des fichiers
            if (np.shape(lab)[0] != np.shape(mfcc)[0]):
                print('Les fichiers ont des tailles différentes', np.shape(lab)[0], np.shape(mfcc)[0])

            # Récupération des labels à 1
            indices = lab == 1

            # Concaténation des MFCC correspondant aux labels à 1
            mfcc_val = mfcc[indices, :]
            param_loc = np.concatenate((param_loc, mfcc_val))

        # Test sur le numéro du locuteur
        if (x == locuteur):
            # Nom du fichier MFCC locuteur
            fichier_loc = REP_MFCC + '/L' + str(x) + '.mfcc'

            # Enregistrement dans un fichier texte des MFCC pour le modèle "locuteur"
            np.savetxt(fichier_loc, param_loc, fmt='%f')

        # Concaténation des MFCC du locuteur "x" aux autres
        param = np.concatenate((param, param_loc))

    # Enregistrement dans un fichier texte des MFCC pour le modèle "monde"
    np.savetxt('/home/python/PW_speech_processing/TP2/MFCC/monde.mfcc', param, fmt='%f')





# Affectation
affectation('/home/python/PW_speech_processing/TP2/LABELS', '/home/python/PW_speech_processing/TP2/MFCC', nb_MFCC, nb_locuteur, nb_fic_app, locuteur_cible)

# EM (appelant VQ) pour le modèle du "monde"
d_monde = np.loadtxt('/home/python/PW_speech_processing/TP2/MFCC/monde.mfcc')
#------------------------------------------------------------> A compléter...

# Entraînement du modèle GM pour le "monde"
gmm_monde = em(n_components=nbe_gauss, covariance_type='full', max_iter=200, random_state=0)
gmm_monde.fit(d_monde)

# Affichage des paramètres du modèle GM pour le "monde"
print("Paramètres du modèle pour le monde :")
print("Poids :", gmm_monde.weights_)
print("Moyennes :", gmm_monde.means_)
print("Covariances :", gmm_monde.covariances_)

# EM (MAP) pour le modèle du "locuteur"
#-----------------------------------------------------------> A compléter...

d_locuteur = np.loadtxt('/home/python/PW_speech_processing/TP2/MFCC/L' + str(locuteur_cible) + '.mfcc')

# Entraînement du modèle GMM pour le "locuteur"
gmm_locuteur = em(n_components=nbe_gauss, covariance_type='full', max_iter=200, random_state=0)
gmm_locuteur.fit(d_locuteur)

# Affichage des paramètres du modèle GMM pour le "locuteur"
print("\nParamètres du modèle GM pour le locuteur :")
print("Poids :", gmm_locuteur.weights_)
print("Moyennes :", gmm_locuteur.means_)
print("Covariances :", gmm_locuteur.covariances_)


################################
    # 5 - RECONNAISSANCE #
################################


def tests_total(nbe_loc, nbe_fic, nb_bits, taille_fenetre, nbe_coef, monde, loc, seuil):
    """
    Effectue la reconnaissance des fichiers audio inconnus dans le répertoire 'RECO'.
    
    :param nbe_loc: Nombre de locuteurs
    :param nbe_fic: Nombre de fichiers par locuteur
    :param nb_bits: Nombre de bits de quantification
    :param taille_fenetre: Taille de la fenêtre d'analyse
    :param nbe_coef: Nombre de coefficients cepstraux (MFCC)
    :param monde: Modèle GMM pour le "monde"
    :param loc: Modèle GMM pour le "locuteur cible"
    :param seuil: Seuil de décision pour la reconnaissance
    :return: Résultats de reconnaissance pour chaque fichier
    """
    # Chemin du répertoire contenant les fichiers audio inconnus
    chemin_reco = "/home/python/PW_speech_processing/TP2/RECO"
    
    # Vérifier si le répertoire existe
    if not os.path.exists(chemin_reco):
        raise FileNotFoundError(f"Le répertoire '{chemin_reco}' n'existe pas.")
    
    # Liste des fichiers audio dans le répertoire
    fichiers_audio = [f for f in os.listdir(chemin_reco) if f.endswith('.wav')]
    
    # Initialisation des résultats
    resultats = []
    
    # Boucle sur tous les fichiers audio inconnus
    for fichier in fichiers_audio:
        chemin_fichier = os.path.join(chemin_reco, fichier)
        
        # Lecture du fichier audio
        freq, duree, signal_normalise = lecture(chemin_fichier, nb_bits)
        
        # Étiquetage pour identifier les zones de parole
        etiquettes = etiquetage(signal_normalise, taille_fenetre, 0.0001)  # Utiliser le même seuil que pour l'apprentissage
        
        # Calcul des MFCC
        mfcc = parametrisation(signal_normalise, taille_fenetre, nbe_coef)
        
        # Sélection des trames correspondant à la parole
        indices_parole = np.where(etiquettes == 1)[0]
        
        # Vérification qu'il y a des trames de parole détectées
        if len(indices_parole) == 0:
            print(f"Aucune trame de parole détectée dans {fichier}, impossible de faire la reconnaissance")
            resultats.append({'fichier': fichier, 'reconnaissance': 'Non reconnu', 'score': 0.0})
            continue
        
        # Sélection des MFCC correspondant aux trames de parole
        mfcc_parole = mfcc[indices_parole]
        
        # Calcul des scores de vraisemblance pour chaque trame de parole
        # Utilisation de score_samples qui retourne la log-vraisemblance pour chaque échantillon
        scores_monde = monde.score_samples(mfcc_parole)
        scores_locuteur = loc.score_samples(mfcc_parole)
        
        # Calcul du rapport de vraisemblance pour chaque trame
        rapports = scores_locuteur - scores_monde
        
        # Décision par vote majoritaire pour chaque trame
        decisions_trames = (rapports > seuil).astype(int)
        votes_positifs = np.sum(decisions_trames)
        
        # Score de confiance: proportion de votes positifs
        score_confiance = votes_positifs / len(decisions_trames) if len(decisions_trames) > 0 else 0
        
        # Décision finale par vote majoritaire
        reconnaissance = "Reconnu" if score_confiance > 0.5 else "Non reconnu"
        
        # Ajout du résultat à la liste
        resultats.append({
            'fichier': fichier,
            'reconnaissance': reconnaissance,
            'score': score_confiance
        })
        
        # Affichage des résultats pour chaque fichier
        print(f"Fichier : {fichier}")
        print(f"Nombre de trames de parole : {len(indices_parole)}")
        print(f"Votes positifs : {votes_positifs} / {len(decisions_trames)}")
        print(f"Score de confiance : {score_confiance:.4f}")
        print(f"Reconnaissance : {reconnaissance}")
        print("-" * 40)
    
    # Calcul des statistiques globales
    nb_acceptes = sum(1 for r in resultats if r['reconnaissance'] == "Reconnu")
    print(f"Nombre de fichiers acceptés : {nb_acceptes} / {len(resultats)}")
    
    # Vérification des résultats (si on connaît déjà qui est le locuteur cible)
    fichiers_cible = [f for f in fichiers_audio if f'L{locuteur_cible}_' in f]
    reconnaissance_correcte = 0
    
    for r in resultats:
        # Vérifier si c'est un fichier du locuteur cible
        est_cible = any(cible in r['fichier'] for cible in fichiers_cible)
        est_reconnu = r['reconnaissance'] == "Reconnu"
        
        # Vérifier si la reconnaissance est correcte
        if (est_cible and est_reconnu) or (not est_cible and not est_reconnu):
            reconnaissance_correcte += 1
    
    taux_correcte = (reconnaissance_correcte / len(resultats)) * 100
    print(f"Taux de reconnaissance correcte : {taux_correcte:.2f}%")
    
    return resultats

# Exemple d'utilisation de la fonction tests_total
resultats = tests_total(1, nb_fic_app, q, taille_fenetre, nb_MFCC, gmm_monde, gmm_locuteur, 0.0)