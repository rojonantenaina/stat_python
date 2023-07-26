import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import f, pearsonr
from scipy.stats import f_oneway

# Saisie manuelle des données
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
Y = np.array([3, 5, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 23, 25, 26, 28, 30, 31, 33, 35, 36, 38, 40, 41, 43, 45, 46, 48,49, 60])

# Création d'un DataFrame
data = pd.DataFrame({'X': X, 'Y': Y})

# Régression linéaire
model = smf.ols('Y ~ X', data=data).fit()

# Test de Fisher
f_value, p_value = f_oneway(X, Y)

# Affichage des résultats de régression
print(model.summary())

#Affichage de test de Fisher
print("\nTest de Fisher")
print("Valeur F :", f_value)
print("Valeur p :", p_value)

# Calcul de l'ANOVA
anova_table = sm.stats.anova_lm(model)

# Affichage de l'ANOVA
print("\nTableau ANOVA :")
print(anova_table)

# Graphique de l'ANOVA
fig, ax = plt.subplots()
ax.plot(X, Y, 'bo', label='Données')
ax.plot(X, model.fittedvalues, 'r-', label='Régression linéaire')
ax.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression linéaire')
plt.show()

# Prévision et corrélation
predicted_values = model.predict(data)
correlation, p_value = pearsonr(Y, predicted_values)

print(f"\nCorrélation entre les valeurs observées et prévues : {correlation}")
print(f"P-value de corrélation : {p_value}")
