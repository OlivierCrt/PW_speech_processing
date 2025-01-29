import numpy as np# type: ignore
import scipy.io.wavfile as wav # type: ignore
from canaux24 import canaux
from scipy import linalg# type: ignore
import matplotlib.pyplot as plt# type: ignore
import pickle


#Crampette
#Olivier
#TP1


##################################################
### PARTIE 1 : Réduction de la dimensionnalité ###
##################################################


# Définition de la fonction lecture 
def lecture(classe, NbFichiers):
    # Initialisation
    obs = []

    for NoFichier in range(NbFichiers):
        # Création du nom du fichier
           dizaine = str((NoFichier + 1) // 10)
           unite = str((NoFichier + 1) % 10)
           nom = 'Signal/' + classe + dizaine + unite + '.wav'
           print(nom)
           
           # Lecture du fichier son
           fs, signal = wav.read(nom)
           signal=signal/(2**15)
           
           # Calcul des 24 canaux d'énergie
           can = canaux(signal, fs)
           
           # Ajout des données à la matrice des observations
           obs.append(can)
    return np.array(obs)


# Lecture des 20 fichiers pour chacune des classes
obsaa = lecture('aa', 20)
obsuu = lecture('uu', 20)
obsii = lecture('ii', 20)



# Créer une matrice contenant l'ensemble des observations à traiter
R = np.vstack((obsaa, obsuu, obsii))

print(R)
print(np.shape(R))

# En utilisant les fonctions cov et eig, constituer une matrice de passage qui permette de passer 
# de l'espace de dimension 24 généré par la fonction canaux dans un espace à deux dimensions 
# correspondant aux deux composantes principales.

# COVARIANCE
print("Matrice des cov:\n")
covariance = np.cov(np.transpose(R))

print(covariance)
print("\n\n\n")


# Valeurs et vecteurs propres
Lambda, V = linalg.eig(covariance)
print(Lambda)
print("\n\n\n")
print(V)
print(np.shape(V))
print(np.shape(Lambda))


# Pour cela, 
# 1 - Rechercher les deux valeurs propres maximales 

# à compléter



# 2-  Prendre les vercteurs propres correspondants
# Rechercher les deux valeurs propres maximales
indices_max = np.argsort(Lambda)[-2:]

# Prendre les vecteurs propres correspondants
W = V[:, indices_max]
print(np.shape(W))
print("\n\n\n\n")
print(W,np.shape(W))

# 3-  Projeter R dans ce nouvel espace 
Rproj = np.dot(R, W)

aaproj = Rproj[:20,:]
print(aaproj)
print(np.shape(aaproj))
uuproj = Rproj[20:40,:]
print(uuproj)
print(np.shape(uuproj))
iiproj = Rproj[40:,:]
print(iiproj)
print(np.shape(iiproj))

# 4- Affichage
# Nuages de points de chacune des classes de sons sur une même figure en les différenciant 
# par des couleurs ou une forme de tracé de points différentes + légende
"""
plt.figure(1)
plt.plot(aaproj[:, 0], aaproj[:, 1], 'r*', label="A")
plt.plot(uuproj[:, 0], uuproj[:, 1], 'go', label="U")
plt.plot(iiproj[:, 0], iiproj[:, 1], 'bx', label="I")
plt.legend()
plt.show()"""

# Utilisez les deux vecteurs propres les moins représentatifs pour la matrice de passage, 
# Que devient la représentation des nuages ? Est-ce qu'une discrimination linéaire est possible ?
# Pour cela, 
# 5- Rechercher les deux valeurs propres minimales
# Rechercher les deux valeurs propres minimales
indices_min = np.argsort(Lambda)[:2]

# Prendre les vecteurs propres correspondants
W_min = V[:, indices_min]

# Projeter R dans ce nouvel espace
Rproj_min = np.dot(R, W_min)

aaproj_min = Rproj_min[:20,:]
uuproj_min = Rproj_min[20:40,:]
iiproj_min = Rproj_min[40:,:]

# Affichage
"""
plt.figure(2)
plt.plot(aaproj_min[:, 0], aaproj_min[:, 1], 'r*', label="A")
plt.plot(uuproj_min[:, 0], uuproj_min[:, 1], 'go', label="U")
plt.plot(iiproj_min[:, 0], iiproj_min[:, 1], 'bx', label="I")
plt.legend()
plt.show()"""



# à compléter


        ######################################################
        ### PARTIE 2 : Classification par lois gaussiennes ###
        ######################################################

# 1- Charger les données d'apprentissage avec le code suivant :
print("\n\n")
print("############### PART 2 ####################### \n\n")
f = open('APP.pkl', 'rb')
dicoAPP = pickle.load(f)
app_aa = dicoAPP["aa"]
app_ii = dicoAPP["ii"]
app_uu = dicoAPP["uu"]
f.close()

# Il s'agit de matrices 80x2 résultant d'une ACP sur une paramétrisation cepstrale (cf. PARTIE 1)
print("app_aa: \n",app_aa)
print("app_aa ligne: \n",app_aa[:,0])
print(np.shape(app_aa))
print(np.shape(app_ii))
print(np.shape(app_uu))

# 2- Afficher sur une seule figure, en utilisant subplot(nb_lignes, nb_colonnes, numero_trace), 
# l'histogramme de chacune des classes : aa, uu et ii.
# à compléter
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(app_aa[:, 0], bins=20, alpha=0.7, label="aa - dim 1", color='r')
plt.hist(app_aa[:, 1], bins=20, alpha=0.7, label="aa - dim 2", color='b')
plt.legend()
plt.title("Histogramme - Classe AA")

plt.subplot(1, 3, 2)
plt.hist(app_uu[:, 0], bins=20, alpha=0.7, label="uu - dim 1", color='r')
plt.hist(app_uu[:, 1], bins=20, alpha=0.7, label="uu - dim 2", color='b')
plt.legend()
plt.title("Histogramme - Classe UU")

plt.subplot(1, 3, 3)
plt.hist(app_ii[:, 0], bins=20, alpha=0.7, label="ii - dim 1", color='r')
plt.hist(app_ii[:, 1], bins=20, alpha=0.7, label="ii - dim 2", color='b')
plt.legend()
plt.title("Histogramme - Classe II")

plt.tight_layout()
plt.show()

# Que pouvez-vous conclure ? 
# àCela suit bien une loi gaussienne a vu d'oeil



# 3- APPRENTISSAGE : estimer les paramètres (moyenne et matrice de covariance) de chaque classe

# Moyennes
m_aa = np.mean(app_aa, axis=0)
m_uu = np.mean(app_uu, axis=0)
m_ii = np.mean(app_ii, axis=0)

# Matrices de covariance
c_aa = np.cov(app_aa, rowvar=False)
c_uu = np.cov(app_uu, rowvar=False)
c_ii = np.cov(app_ii, rowvar=False)
# Affichage des résultats
print("Moyenne et covariance de la classe 'aa':")
print("Moyenne:", m_aa)
print("Covariance:\n", c_aa, "\n")

print("Moyenne et covariance de la classe 'uu':")
print("Moyenne:", m_uu)
print("Covariance:\n", c_uu, "\n")

print("Moyenne et covariance de la classe 'ii':")
print("Moyenne:", m_ii)
print("Covariance:\n", c_ii, "\n")



# Quelles sont les dimensions des variables crées ; que contiennent-elles ? 
print("Dimensions':")
print(m_aa)
print(c_aa)
print(np.shape(m_aa))
print(np.shape(c_aa))


# 4- RECONNAISSANCE : écrire une fonction classer qui renvoie la classe obtenue 
# par maximum de vraisemblance sur les lois gaussiennes estimées (modèle de chaque classe) 
# sur l'observation fournie en entrée :

def log_densite_gaussienne(x, moyenne, covariance):
    """ Calcule log(p(x)) pour une loi normale 2D """
    det_cov = np.linalg.det(covariance)  # Déterminant de la covariance
    inv_cov = np.linalg.inv(covariance)  # Inverse de la covariance

    diff = x - moyenne
    zerp = np.dot(diff.T, np.dot(inv_cov, diff))

    # Calcul de log(p(x)) (simplifié pour d=2)
    log_proba = -np.log(2 * np.pi) - 0.5 * np.log(det_cov) - 0.5 * zerp
    
    return log_proba

def classer(observation, moyenne_aa, covariance_aa, moyenne_uu, covariance_uu, moyenne_ii, covariance_ii):
    # Calcul des log-probabilités pour chaque classe
    log_proba_aa = log_densite_gaussienne(observation, moyenne_aa, covariance_aa)
    log_proba_uu = log_densite_gaussienne(observation, moyenne_uu, covariance_uu)
    log_proba_ii = log_densite_gaussienne(observation, moyenne_ii, covariance_ii)

    log_probas = [log_proba_aa, log_proba_uu, log_proba_ii]
    classes = ["aa", "uu", "ii"]
    no_classe = classes[np.argmax(log_probas)]  

    return no_classe



# 5- Des observations à tester sont présentes dans le fichier Pickle suivant : TST.pkl
# Charger ces données de tests comme  pour les données d'apprentissage)
ft = open('TST.pkl', 'rb')
dicoTST = pickle.load(ft)
tst_aa = dicoTST["aa"]
tst_ii = dicoTST["ii"]
tst_uu = dicoTST["uu"]
ft.close()

# Vérifier leurs dimensions
# à compléter



# Exemple de résultat : 
classe = classer(tst_aa[0,:], m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
print(classe)
# => no_classe = 1

# A tester pour vérifier votre fonction "classer" !


# 6- Définir la fonction tout_tester qui effectue tous les tests de classification 

def tout_tester(tst_aa, tst_uu, tst_ii, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii):
    """ Teste tous les échantillons et construit la matrice de confusion """
    
    matrice_confusion = np.zeros((3, 3))  # Matrice 3x3 pour les 3 classes
    classes = {"aa": 0, "uu": 1, "ii": 2}  # Correspondance des labels

    for x in tst_aa:
        classe_predite = classer(x, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
        matrice_confusion[0, classes[classe_predite]] += 1  # Classe réelle = 0 (aa)

    for x in tst_uu:
        classe_predite = classer(x, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
        matrice_confusion[1, classes[classe_predite]] += 1  # Classe réelle = 1 (uu)

    for x in tst_ii:
        classe_predite = classer(x, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
        matrice_confusion[2, classes[classe_predite]] += 1  # Classe réelle = 2 (ii)

    # Calcul du taux de bonne reconnaissance
    total_correct = np.trace(matrice_confusion)  # Somme des éléments sur la diagonale
    total_samples = np.sum(matrice_confusion)  # Nombre total d'échantillons testés
    taux_reco = total_correct / total_samples  # Taux de reconnaissance
    
    return matrice_confusion, taux_reco
    

# 7- Réaliser les tests et afficher la matrice de confusion et le taux de bonne reconnaisance correspondant 
mat_conf, score = tout_tester(tst_aa, tst_uu, tst_ii, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
print("mat confiance:\n",mat_conf)
print("score: ",score)

# Commenter les résultats obtenus
# c est parfait