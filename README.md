# 💉 DiabèteScope

**Dépistage du risque de diabète par machine learning, avec une application Streamlit pour tester le modèle.**

Projet réalisé dans le cadre de l'**UMEF AI Bootcamp** · *GalsenAI Ambassador Program*.

> ⚠️ **Avertissement :** ce projet est pédagogique. Le modèle estime un *risque* ; il ne pose **pas** de diagnostic médical et ne remplace pas l'avis d'un professionnel de santé.

---

## 🎯 Objectif

À partir de 8 mesures médicales simples (glycémie, IMC, âge, tension, etc.), prédire la probabilité qu'une patiente soit diabétique. Le projet couvre toute la chaîne :

1. **Analyse** : nettoyage, exploration, comparaison de modèles (notebook).
2. **Application** : une interface web pour tester le modèle sur une patiente ou un fichier entier.

## 📊 Données

Jeu de données **Pima Indians Diabetes** : 768 femmes d'origine Pima (21 ans et plus), 8 variables + la cible `Outcome` (1 = diabétique).

| Variable | Description |
|---|---|
| `Pregnancies` | Nombre de grossesses |
| `Glucose` | Glycémie (test de tolérance au glucose, 2 h) |
| `BloodPressure` | Tension diastolique (mm Hg) |
| `SkinThickness` | Pli cutané du triceps (mm) |
| `Insulin` | Insuline sérique à 2 h (µU/ml) |
| `BMI` | Indice de masse corporelle |
| `DiabetesPedigreeFunction` | Score d'antécédents familiaux |
| `Age` | Âge (années) |

**Point clé du nettoyage :** dans les données d'origine, un `0` pour `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` ou `BMI` signifie « non mesuré ». Ces zéros sont remplacés par des valeurs manquantes, puis imputés par la médiane dans le pipeline.

## 🧠 Méthode

- **Pipeline scikit-learn** : imputation (médiane) → standardisation → modèle.
- **Trois modèles comparés** : régression logistique, forêt aléatoire, gradient boosting (avec pondération des classes pour compenser le déséquilibre).
- **Sélection** sur le ROC-AUC en validation croisée stratifiée à 5 plis, comparé à une référence naïve (toujours la classe majoritaire).
- **Seuil de décision ajustable** : un faux négatif (diabète non détecté) est plus grave qu'une fausse alerte, on privilégie donc un bon **rappel**.
- **Interprétabilité** : importance des variables par permutation.

### Résultats (jeu de test, 154 patientes)

Meilleur modèle : **régression logistique**.

| Métrique | Valeur |
|---|---|
| Accuracy | 0,73 |
| Rappel (classe « diabète ») | 0,70 |
| Précision (classe « diabète ») | 0,60 |
| ROC-AUC | ≈ 0,82 |

Les performances sont correctes sans être parfaites : 8 mesures ne suffisent pas à tout prédire.

## 🖥️ L'application

L'application Streamlit propose 5 onglets :

| Onglet | Contenu |
|---|---|
| 🧑‍⚕️ **Une patiente** | Saisie des mesures (avec option « Inconnue »), probabilité de diabète et explication des facteurs qui pèsent dans la prédiction |
| 📄 **Fichier CSV** | Prédiction sur un lot de patientes, métriques si la colonne `Outcome` est présente, export des résultats |
| 📊 **Données** | Exploration du jeu de données : distributions, qualité, comparaison des groupes, corrélations |
| 📈 **Performances** | Métriques en validation croisée, matrice de confusion, courbe ROC, importance des variables, effet du seuil |
| ℹ️ **À propos** | Fonctionnement du modèle et limites |

Le seuil de décision se règle dans la barre latérale et met à jour tous les résultats.

## 🚀 Installation et lancement

```bash
git clone https://github.com/<votre-utilisateur>/diabete-scope.git
cd diabete-scope

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

Le modèle `modele_diabete.joblib` est réutilisé s'il est compatible avec votre version de scikit-learn ; sinon, il est **réentraîné automatiquement** à partir de `diabetes.csv` (quelques secondes).

Pour refaire l'analyse complète, ouvrez le notebook :

```bash
jupyter notebook "05_Diabete complete .ipynb"
```

## 📁 Structure

```
.
├── app.py                        # Application Streamlit
├── 05_Diabete complete .ipynb    # Notebook : nettoyage, EDA, modélisation
├── diabetes.csv                  # Données Pima Indians Diabetes
├── modele_diabete.joblib         # Pipeline entraîné (imputation + scaling + modèle)
└── requirements.txt
```

## ⚖️ Limites

- **Population restreinte** : 768 femmes Pima. Les résultats ne se généralisent pas automatiquement à d'autres populations (par exemple la population sénégalaise).
- **Petit jeu de données** et nombreuses valeurs manquantes (notamment `Insulin` et `SkinThickness`).
- L'explication par variable est une approximation : elle montre des associations, **pas des causes**.

## 🙏 Remerciements

- **UMEF** et la communauté **GalsenAI** pour le bootcamp.
- Jeu de données : *National Institute of Diabetes and Digestive and Kidney Diseases*, via [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
