import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def pca_et_visualisation(X, y):
    """
    Cette fonction prend en entrée un ensemble d'entraînement (X) et ses étiquettes (y), 
    effectue une normalisation des données, applique la réduction de dimension avec PCA, 
    puis visualise les résultats dans un espace 2D et vérifie la possibilité de séparation linéaire.

    Paramètres :
    - X : Données d'entrée (ensemble d'entraînement).
    - y : Étiquettes des données (classes).
    
    Retour :
    - Aucune valeur retournée : la fonction effectue seulement la réduction de dimension, la visualisation et l'évaluation de la séparation linéaire.
    """
    #Normalisation des données
    scaler = StandardScaler()
    X_normalise = scaler.fit_transform(X)  # Normalisation des données
    
    #PCA
    pca = PCA(n_components=2)  
    X_pca = pca.fit_transform(X_normalise)  
    
    plt.figure(figsize=(8, 6))


    color_map = {0: 'red', 1: 'blue'} 

    labels = ['cervix' if label == 1 else 'non cervix' for label in y]



    scatter_cervix = plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='orange', edgecolor='k', s=50, label='cervix')
    scatter_non = plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='blue', edgecolor='k', s=50, label='non cervix')

    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.title('Visualisation 2D après réduction de dimension avec PCA')


    plt.legend(loc='best') 
    
    # Entraîner un modèle SVM linéaire
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_pca, y)
    
    # Tracer la frontière de décision (hyperplan)
    # Récupérer les coefficients du modèle SVM linéaire
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]
    
    # Calculer les coordonnées de la frontière de décision
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracer la frontière de décision
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='--')
    
    # Tracer les marges (soft margin)
    # Calculer les marges : y = wx + b, les marges sont données par (1 - w * x + b) et (-1 - w * x + b)
    margin = 1 / np.linalg.norm(w)
    
    # Calculer les marges positives et négatives
    Z_margin_pos = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]) - margin
    Z_margin_neg = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]) + margin
    
    Z_margin_pos = Z_margin_pos.reshape(xx.shape)
    Z_margin_neg = Z_margin_neg.reshape(xx.shape)
    
    # Tracer les marges
    plt.contour(xx, yy, Z_margin_pos, levels=[0], linewidths=1, colors='black', linestyles='--')
    plt.contour(xx, yy, Z_margin_neg, levels=[0], linewidths=1, colors='black', linestyles='--')
    
    plt.show()

