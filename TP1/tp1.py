import numpy as np
import scipy.io.wavfile as wav
from canaux24 import canaux
from scipy import linalg
import matplotlib.pyplot as plt
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


"""
# 2-  Prendre les vercteurs propres correspondants
W = # à compléter

print(np.shape(W))
print(W)

# 3-  Projeter R dans ce nouvel espace 
Rproj = # à compléter

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
plt.figure(1)
plt.plot(aaproj[:, 0], aaproj[:, 1], 'r*', label="A")
plt.plot(uuproj[:, 0], uuproj[:, 1], 'go', label="U")
plt.plot(iiproj[:, 0], iiproj[:, 1], 'bx', label="I")
plt.legend()
plt.show()

# Utilisez les deux vecteurs propres les moins représentatifs pour la matrice de passage, 
# Que devient la représentation des nuages ? Est-ce qu'une discrimination linéaire est possible ?
# Pour cela, 
# 5- Rechercher les deux valeurs propres minimales

# à compléter


        ######################################################
        ### PARTIE 2 : Classification par lois gaussiennes ###
        ######################################################

# 1- Charger les données d'apprentissage avec le code suivant :
f = open('APP.pkl', 'rb')
dicoAPP = pickle.load(f)
app_aa = dicoAPP["aa"]
app_ii = dicoAPP["ii"]
app_uu = dicoAPP["uu"]
f.close()

# Il s'agit de matrices 80x2 résultant d'une ACP sur une paramétrisation cepstrale (cf. PARTIE 1)
print(np.shape(app_aa))
print(np.shape(app_ii))
print(np.shape(app_uu))

# 2- Afficher sur une seule figure, en utilisant subplot(nb_lignes, nb_colonnes, numero_trace), 
# l'histogramme de chacune des classes : aa, uu et ii.
# à compléter

# Que pouvez-vous conclure ? 
# à compléter



# 3- APPRENTISSAGE : estimer les paramètres (moyenne et matrice de covariance) de chaque classe

m_aa = # à compléter
c_aa = # à compléter
m_uu = # à compléter
c_uu = # à compléter
m_ii = # à compléter
c_ii = # à compléter

# Quelles sont les dimensions des variables crées ; que contiennent-elles ? 
print(m_aa)
print(c_aa)
print(np.shape(m_aa))
print(np.shape(c_aa))


# 4- RECONNAISSANCE : écrire une fonction classer qui renvoie la classe obtenue 
# par maximum de vraisemblance sur les lois gaussiennes estimées (modèle de chaque classe) 
# sur l'observation fournie en entrée :

def classer(observation, moyenne_aa, covariance_aa, moyenne_uu, covariance_uu, moyenne_ii, covariance_ii):
    
    # à compléter

    return(no_classe)



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
# classe = classer(tst_aa[0,:], m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
# => no_classe = 1

# A tester pour vérifier votre fonction "classer" !


# 6- Définir la fonction tout_tester qui effectue tous les tests de classification 

def tout_tester(tst_aa, tst_uu, tst_ii, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii):
    matrice_confusion = np.zeros((3,3))
    
    # à compléter
    
    taux_reco = # à compléter
    
    return(matrice_confusion, taux_reco)
    

# 7- Réaliser les tests et afficher la matrice de confusion et le taux de bonne reconnaisance correspondant 
mat_conf, score = tout_tester(tst_aa, tst_uu, tst_ii, m_aa, c_aa, m_uu, c_uu, m_ii, c_ii)
print(mat_conf)
print(score)

# Commenter les résultats obtenus
# à compléter """