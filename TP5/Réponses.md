# Réponses TP5
## 1.1
La grammaire décrit un langage pour interpréter des expressions arithmétiques simples en français, comme "un plus deux" ou "cinq moins trois". Elle associe des mots français (ex: "un", "plus") à des nombres et opérateurs (ex: 1, +). Le langage permet de comprendre et de calculer des opérations de base (addition et soustraction) en utilisant des mots plutôt que des symboles mathématiques.

FCFG (Feature-Based Context-Free Grammar) est un format utilisé pour définir des grammaires à traits dans le cadre du traitement automatique du langage naturel (TALN). Il s'agit d'une variante des grammaires hors contexte (CFG) enrichie avec des traits sémantiques et syntaxiques.

Caractéristiques principales d'une grammaire FCFG :
- Basée sur des règles contextuelles (comme les CFG classiques).
- Ajoute des traits sémantiques pour donner du sens aux phrases.
- Permet d'extraire des structures interprétables
## 1.2
L'analyseur créé a pour rôle de comprendre et interpréter des expressions arithmétiques simples en français (comme "un plus deux") en utilisant la grammaire définie dans GRAMMAIRE_ASSISTANT_V1.fcfg. Il décompose la phrase en éléments (nombres, opérateurs), associe chaque mot à sa signification (ex: "un" → 1, "plus" → +), et prépare l'expression pour un calcul (ex: "un plus deux" → 1 + 2).
## 1.3
### Entrées :
**dico_test** :
Un dictionnaire de phrases à analyser (ex: {"phrase1": "un plus deux"}).

**analyseur** :
Un analyseur syntaxique pour interpréter les phrases.

**Sortie** :
resultat_analyse_lot :
Un dictionnaire contenant les résultats de l'analyse pour chaque phrase.

**Fonctionnement** :
La fonction analyse chaque phrase du dictionnaire avec l'analyseur et retourne les résultats.
## 1.4
rien
## 2.1 Exécuter les trois cellules suivantes pour tester cette version de la grammaire et le fonctionnement de l'analyseur correspondant.

## 2.2
Avec le mode trace : On voit chaque étape de l'analyse syntaxique et comment la phrase est transformée en une structure interprétable.
Sans le mode trace : On obtient uniquement les phrases d’entrée validées sans explication du traitement.
## 3.3
**Entrées :**
operation : une chaîne contenant une opération mathématique sous forme textuelle (ex: "5, +, 3").
**Sorties :**
Un nombre (résultat du calcul) ou None en cas d’erreur.
```python
calculer("5, +, 3")  # → 8  
calculer("2, ^, 3")  # → 8  
calculer("10, /, 0")  # → None (erreur)  
```
## 3.4
Les résultats montrent que le système d'analyse et d'interprétation de langage naturel fonctionne correctement avec les différents types de commandes. Le système réussit à analyser et traiter les opérations mathématiques ainsi que les commandes de communication en utilisant la grammaire définie.

Pour les opérations mathématiques, le système reconnaît les nombres sous forme numérique et textuelle, ainsi que les différents opérateurs (+, -, /, *, ^). La fonction d'extraction d'interprétation isole correctement les éléments pertinents du résultat d'analyse, et la fonction de calcul modifiée traite les opérandes et les opérateurs pour produire le résultat attendu. Par exemple, "5 + 3" est correctement évalué à 8, "6 - 2" à 4, "20 / 4" à 5.0, et "20 ^ 2" à 400.0.

Pour les commandes de communication, le système identifie correctement le type de commande (appel) et le destinataire (Polo). Il génère des réponses appropriées comme "Nous appelons Polo", démontrant que la grammaire a été correctement étendue pour gérer non seulement les calculs mathématiques mais aussi les interactions communicatives.