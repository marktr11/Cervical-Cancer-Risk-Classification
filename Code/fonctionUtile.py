import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score

def pca_et_visualisation(X, y):
    """
    Cette fonction prend en entrée un ensemble d'entraînement (X) et ses étiquettes (y), 
    effectue une normalisation, applique PCA (2 composantes), entraîne un SVM linéaire, 
    puis visualise les résultats dans un espace 2D avec l'hyperplan et les marges douces.

    Paramètres :
    - X : Données d'entrée (ensemble d'entraînement).
    - y : Étiquettes des données (classes).
    """
    # Normalisation des données
    scaler = StandardScaler()
    X_normalise = scaler.fit_transform(X)
    
    # PCA avec 2 composantes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_normalise)
    
    # Entraînement du SVM linéaire sur l'ensemble d'entraînement avec soft margin
    svm_model = SVC(kernel='linear', C=1.0)  # C kiểm soát mức độ soft margin
    svm_model.fit(X_pca, y)
    
    # Prédiction et calcul de la précision sur l'ensemble d'entraînement
    y_pred = svm_model.predict(X_pca)
    train_accuracy = accuracy_score(y, y_pred)
    print(f"Précision sur l'ensemble d'entraînement : {train_accuracy:.2f}")
    
    # Création de la figure
    plt.figure(figsize=(10, 8))

    # Tracé des points de l'ensemble d'entraînement
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], 
                c='orange', edgecolor='k', s=50, label='cervix (train)')
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], 
                c='blue', edgecolor='k', s=50, label='non cervix (train)')
    
    # Marquage des points mal classés sur l'ensemble d'entraînement
    mal_classifies = y != y_pred
    if np.any(mal_classifies):
        plt.scatter(X_pca[mal_classifies, 0], X_pca[mal_classifies, 1], 
                    c='red', marker='x', s=100, label='Mal classés (train)')

    # Récupération des paramètres de l'hyperplan
    w = svm_model.coef_[0]  # [w1, w2]
    b = svm_model.intercept_[0]
    
    # Création de la frontière de décision (droite : w1*x1 + w2*x2 + b = 0)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 100)
    yy = (-w[0] * xx - b) / w[1]  # Hyperplan central
    
    # Calcul des marges douces (soft margins)
    margin = 1 / np.linalg.norm(w)  # Largeur de la marge = 1 / ||w||
    yy_pos = yy + margin * np.sqrt(1 + (w[0] / w[1])**2)  # Marge positive
    yy_neg = yy - margin * np.sqrt(1 + (w[0] / w[1])**2)  # Marge négative
    
    # Tracé de l'hyperplan et des marges
    plt.plot(xx, yy, 'k-', linewidth=2, label='Hyperplan')  # Hyperplan central
    plt.plot(xx, yy_pos, 'k--', linewidth=1, label='Marge positive')  # Marge +1
    plt.plot(xx, yy_neg, 'k--', linewidth=1, label='Marge négative')  # Marge -1

    # Étiquettes et titre
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.title(f'Visualisation 2D avec PCA, Hyperplan et Marges\nPrécision Train: {train_accuracy:.2f}')
    
    # Ajout de la légende
    plt.legend(loc='best')
    
    # Affichage du graphique
    plt.show()

