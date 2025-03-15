import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_variance_cumul_explique(X, n_components=None):
    """
    Paramètres:
    - X : ndarray (shape : n_samples, n_features), données d'entrée
    - n_components : int (facultatif), nombre de composantes principales à conserver. 
                     Si None, utilise toutes les composantes.

    Retour:
    - Affiche un graphique de la variance cumulée expliquée.
    """
    # Appliquer PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    #la variance expliquée cumulée
    variance_cumulee = np.cumsum(pca.explained_variance_ratio_)

    # Tracer le graphique
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(variance_cumulee) + 1), variance_cumulee, marker='o', linestyle='--', color='b')

    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance cumulée expliquée")
    plt.title("Variance Cumulée Expliquée en fonction du Nombre de Composantes")
    plt.grid(True)

    seuils = [0.9]
    for seuil in seuils:
        n_composantes = np.argmax(variance_cumulee >= seuil) + 1
        plt.axhline(y=seuil, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=n_composantes, color='g', linestyle='--', alpha=0.7)
        plt.scatter(n_composantes, variance_cumulee[n_composantes - 1], color='red', zorder=3)
        plt.text(n_composantes, variance_cumulee[n_composantes - 1] - 0.05, f"{n_composantes} PCs", fontsize=10, color="red")

    plt.show()