import numpy as np
import matplotlib.pyplot as plt
from math import floor

def canaux(extrait, Fs=16000, nfilt=24):
    # extrait : fenêtre d'analyse du signal
    # Fs : fréquence d'échantillonnage (en Hertz)
    # nfilt : nombre de canaux

    # Calcul du spectre avec fenêtrage de Hamming
    ham1024 = np.hamming(len(extrait))
    extrait_ham1024 = np.multiply(extrait, ham1024)
    spectre = abs(np.fft.fft(extrait_ham1024))
    demi_spectre = spectre[:floor(len(spectre) / 2)]

    # Init taille du spectre
    lgsig1 = len(demi_spectre)

    # Init freq coupures (par rapport à l'échelle MEL)
    fcoup = [0,100,200,300,400,500,600,700,800,900,1000,1150,1300,1500,1700,2000,2350,2700,3100,3550,4000,4500,5050,5600,6200,6850,7500,8000]

    # Init filtres triangulaires, x1 (montant), x2 (descendant)
    x2 = [0]*lgsig1
    x1 = [0]*lgsig1

    temp = [j for j in range(1,lgsig1+1)]

    # Fréquences possibles en fonction de la taille de la fenêtre et de Fs
    f = [a/lgsig1*Fs/2 for a in temp]

    # Calcul des poids (entre 0 et 1) des branches du triangle : x1 et x2 avec symétrie par rapport à 0.5
    # + Stockage des indices de fréquences de coupure par rapport aux fréquences dans f
    nm = [0]*lgsig1
    for i in range(0,lgsig1):
        for n in range(0,nfilt+1):
            if (f[i] >= fcoup[n]) and (f[i] < fcoup[n+1]):
                #print (fcoup[n])
                nm[i] = n
                x1[i] = (f[i]-fcoup[n])/(fcoup[n+1]-fcoup[n])
                x2[i] = 1-x1[i]

    # Pondération des valeurs spectrales par rapport aux filtres triangulaires
    x1sig = np.array(x1) * np.array(demi_spectre)
    x2sig = np.array(x2) * np.array(demi_spectre)

    # Somme des valeurs spectrales pondérées pour chaque triangle
    en = [0]*nfilt
    for i in range(0, nfilt):
        for j in range(0,len(nm)):
            if nm[j] == i :
                en[i] += x1sig[j]+x2sig[j]

    # Affichage
    #plt.figure(6)
    #plt.subplot(311)
    #abscisses_extrait = np.arange(len(extrait)) / Fs
    #plt.plot(abscisses_extrait, extrait)
    #plt.xlabel('Temps (s)')
    #plt.ylabel('Amplitude')
    #plt.subplot(312)
    #abscisses_demi_spectre = np.arange(len(demi_spectre)) / len(demi_spectre) * (Fs / 2)
    #plt.plot(abscisses_demi_spectre, demi_spectre)
    #plt.xlabel('Fréquence (Hz)')
    #plt.ylabel('Intensité')
    #plt.subplot(313)
    #plt.step(fcoup[0:nfilt], en)
    #plt.xlabel('Fréquence (Hz)')
    #plt.ylabel('Intensité')
    #plt.show()

    return(en)