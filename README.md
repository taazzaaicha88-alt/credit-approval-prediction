# Prédiction de l'approbation de crédit bancaire

Projet de classification visant à évaluer le risque d'accorder un prêt à un client, à partir de son profil (revenus, historique de crédit, zone du bien, etc.).

## Données
Dataset de 614 demandes de prêt (`Dataset.csv`), variable cible `Loan_Status` (approuvé / refusé).

## Méthode
1. Exploration et vérification des données
2. Traitement des valeurs manquantes (mode pour le catégoriel, médiane pour le numérique)
3. Feature engineering : logarithme des revenus et des montants, encodage des variables
4. Comparaison de six modèles, optimisés par GridSearchCV (validation croisée 5 plis)
5. Évaluation sur le jeu de test : accuracy, F1-score, matrice de confusion

## Résultats (jeu de test)

| Modèle | Accuracy | F1 (approuvé) | Faux positifs |
|---|---|---|---|
| Régression logistique | 0,789 | 0,859 | 25 |
| SVM | 0,789 | 0,859 | 25 |
| Random Forest | 0,772 | 0,846 | 25 |
| Gradient Boosting | 0,76 | 0,832| **24** |
| Arbre de décision | 0,740 | 0,820 | 25 |
| KNN | 0,707 | 0,795 | 26 |

Le modèle retenu est le **Gradient Boosting**, qui minimise les faux positifs (prêts accordés à tort), l'erreur la plus coûteuse pour une banque.

## Limites
Le modèle repose presque entièrement sur l'historique de crédit, qui explique la majorité des décisions. Cette dépendance limite la finesse de l'évaluation des autres critères.

## Reproduire le projet
```bash
pip install -r requirements.txt
jupyter notebook credit_approval.ipynb
```

## Auteur
Aicha Taazza : Master Data Science et Actuariat, Faculté d'Économie et de Gestion, Settat
