import nltk
from nltk import parse, data
import os

# Make sure to set the correct path to your grammar file
nom_fichier_grammaire_V3 = '/home/python/PW_speech_processing/TP5/GRAMMAIRE_ASSISTANT_V3.fcfg'

# Load the grammar and create a parser
data.show_cfg(nom_fichier_grammaire_V3)
analyseur_V3 = parse.load_parser(nom_fichier_grammaire_V3)

def tester_par_lot(Dico_test, analyseur): 
    resultat_analyse_lot = {}
    
    for k in Dico_test.keys():
        print(k)
        tokens = Dico_test[k].split()
        print(tokens)
        resultat_analyse_lot[k] = analyseur.parse(tokens)
        print("=========================================\n")
    
    return resultat_analyse_lot

def extraire_interpretation(arbre):
    tmp = str(arbre[0])
    racine = tmp.split('\n')
    interpretation = racine[0]+')'
    print('interpretation = ', interpretation)
    
    indice = interpretation.find('=')
    chaine_a_interpreter = interpretation[indice+2:len(interpretation) -3]
    print('chaine_a_interpreter = ', chaine_a_interpreter)
    
    return chaine_a_interpreter

def calculer(operation):
    # Convert string operation to an actual calculation
    # Handle different operators properly
    parts = operation.split(", ")
    if len(parts) >= 3:
        num1 = parts[0].strip()
        operator = parts[1].strip()
        num2 = parts[2].strip()
        
        expression = f"{num1} {operator} {num2}"
        # Handle additional operands if present (like in "5 + 3 + 10 - 3")
        for i in range(3, len(parts), 2):
            if i+1 < len(parts):
                expression += f" {parts[i].strip()} {parts[i+1].strip()}"
                
        print(f"Evaluating: {expression}")
        try:
            if operator == "^":  # Handle power operation
                result = float(num1) ** float(num2)
            else:
                result = eval(expression.replace("^", "**"))
            return result
        except Exception as e:
            print(f"Error calculating: {e}")
            return None
    return None

# Define test examples
exemple_test_G3 = { 
    'exemple1' : '5 + 3', 
    'exemple2' : 'six moins deux', 
    'exemple3' : '0 1 2 3 4', 
    'exemple4': '1 2 3 4 5', 
    'exemple5' : 'envoie_un_sms_au 0 1 2 3 4', 
    'exemple6': 'appelle_le 1 2 3 4 5', 
    'exemple7': 'vingt divisé_par quatre', 
    'exemple8' : 'appelle_le cinq fois trois', 
    'exemple9': 'cinq plus trois plus dix moins trois',
    'exemple10': 'appelle Polo', 
    'exemple11': 'vingt puissance deux'
}

# Process the test examples
resultat_test_G3 = tester_par_lot(exemple_test_G3, analyseur_V3)

# Process the results
for k in resultat_test_G3.keys():
    for arbre in resultat_test_G3[k]: 
        print("-------------------")
        chaine_a_interpreter = extraire_interpretation(arbre)
        print(chaine_a_interpreter)
        
        # Identify the command type based on the first part and tree structure
        if "CALCUL_A_FAIRE" in str(arbre[0]):
            # This is a calculation
            resultat = calculer(chaine_a_interpreter)
            if resultat is not None:
                print("resultat = ", resultat)
            else:
                print("Je n'ai pas pu effectuer ce calcul")
        elif "COMMUNICATION" in str(arbre[0]):
            # Identify communication type
            parts = chaine_a_interpreter.split(", ")
            commande = parts[0].strip()
            destinataire = parts[1].strip() if len(parts) > 1 else ""
            
            if commande == "appelle":
                print(f"Nous appelons {destinataire}")
            elif commande == "sms":
                print(f"Vous voulez envoyer un texto à {destinataire}")
            else:
                print("Je n'ai pas compris votre commande de communication")
        else:
            print("Je n'ai pas compris votre requête")
        print("..................................\n\n\n")
        print("=================================================================\n")