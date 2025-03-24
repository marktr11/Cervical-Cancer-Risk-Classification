# Prédiction du Risque de Cancer du Col de l'Utérus

Ce projet vise à prédire le risque de cancer du col de l'utérus en fonction des comportements des patients. Plusieurs modèles d'apprentissage automatique ont été utilisés, tels que SVM, Naïve Bayes, régression logistique, et arbres de décision pour effectuer cette prédiction.

## 1. Préparation des Données

La préparation des données est une étape essentielle dans ce projet. Toutes les informations nécessaires à cette étape sont disponibles dans le fichier de préparation. Ce fichier effectue le nettoyage, le traitement et la prétraitement des données, qui seront ensuite utilisées par tous les modèles d'apprentissage automatique.

Vous pouvez consulter ce fichier en suivant ce lien :  
**[Préparation des Données](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Code/Preparation.ipynb)**

## 2. Modèles Utilisés

Pour prédire le risque de cancer du col de l'utérus, plusieurs modèles d'apprentissage automatique ont été testés :

- **SVM (Support Vector Machine)**  
  [Lien vers le modèle SVM](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Code/SVM.ipynb)

- **Naïve Bayes**  
  [Lien vers le modèle Naïve Bayes](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Code/NaiveBayes.ipynb)

- **Régression Logistique**  
  [Lien vers le modèle Logistique](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Code/logistique_regression.ipynb)

- **Arbres de Décision**  
  [Lien vers le modèle Arbres de Décision](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Code/ArbreDeDecision.ipynb)

Les détails de ces modèles ainsi que leur performance sont expliqués dans le rapport final.

👉 Pour une évaluation détaillée des modèles, consultez le fichier suivant :  
**[Évaluation des Modèles](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Code/Comparer%20les%20mod%C3%A8les.ipynb)**

## 3. Téléchargement du Dataset

Le jeu de données utilisé dans ce projet est disponible ici :  
**[Télécharger le dataset](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Data/sobar-72.csv)**

## 4. Rapport Final

Le rapport détaillant l'ensemble du processus du projet, y compris les étapes de préparation des données, l'expérimentation des modèles et les résultats obtenus, est disponible ici :  
**[Lien vers le rapport](https://github.com/marktr11/Cervical-Cancer-Risk-Classification/blob/main/Rapport/Cervical_Rapport.pdf)**

## 5. Structure du Projet
```txt
Cervical-Cancer-Risk-Classification/
|-- .gitattributes
|-- .gitignore
|-- Code/
|   |-- Comparer les modèles.ipynb
|   |-- NaiveBayes.ipynb
|   |-- Preparation.ipynb
|   |-- SVM.ipynb
|   |-- TreeDecision.ipynb
|   |-- Variables/
|   |   |-- roc_data_NB.pkl
|   |   |-- roc_data_log.pkl
|   |   |-- roc_data_svm.pkl
|   |   |-- roc_data_tree.pkl
|   |   |-- variables.pkl
|   |-- __pycache__/
|   |   |-- fonctionUtile.cpython-311.pyc
|   |-- fonctionUtile.py
|   |-- logistique_regression.ipynb
|-- Data/
|   |-- sobar-72.csv
|-- Image/
|   |-- ROC_4modeles.png
|   |-- ROC_NB.png
|   |-- ROC_NB_smote.png
|   |-- ROC_SVM.png
|   |-- ROC_SVM_SMOTE.png
|   |-- ROC_dt1.png
|   |-- ROC_dt2.png
|   |-- ROC_dt3.png
|   |-- ROC_log1.png
|   |-- ROC_log2.png
|   |-- ROC_log3.png
|   |-- ROC_opt.png
|   |-- boxplot_ca_cervix.png
|   |-- cropped-But-SD.png
|   |-- matrice_correlation.png
|   |-- perform_4modeles.png
|-- Rapport/
|   |-- Cervical_Rapport.pdf
|-- README.md
```


