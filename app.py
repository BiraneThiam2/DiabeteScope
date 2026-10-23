"""
💉 Dépistage du diabète : interface de test du modèle
UMEF AI Bootcamp · GalsenAI Ambassador Program

Lancer :  streamlit run app.py
Prérequis : diabetes.csv dans le même dossier. Le modèle est créé automatiquement
            (ou modele_diabete.joblib est réutilisé s'il est compatible).
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dépistage du diabète", page_icon="💉", layout="wide")

MODEL_PATH = Path(__file__).parent / "modele_diabete.joblib"
CSV_PATH = Path(__file__).parent / "diabetes.csv"
RANDOM_STATE = 42
NOMS_CLASSES = ["Non diabétique", "Diabétique"]

# Les 5 colonnes où un 0 signifie « mesure non faite » (comme dans le notebook)
COLS_ZEROS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Description de chaque variable : l'ordre est celui des colonnes du notebook.
# "ref" = valeur typique (médiane du jeu Pima), utilisée pour expliquer la prédiction.
VARIABLES = {
    "Pregnancies": dict(label="Nombre de grossesses", unit="", mini=0, maxi=20, defaut=2, pas=1,
                        ref=3, manquant=False, aide="Nombre total de grossesses."),
    "Glucose": dict(label="Glycémie", unit="mg/dL", mini=40, maxi=300, defaut=110, pas=1,
                    ref=117, manquant=True,
                    aide="Glycémie 2 h après un test de charge en glucose. C'est la variable la plus importante."),
    "BloodPressure": dict(label="Tension artérielle (diastolique)", unit="mmHg", mini=20, maxi=140, defaut=72, pas=1,
                          ref=72, manquant=True, aide="Pression diastolique (le chiffre du bas)."),
    "SkinThickness": dict(label="Épaisseur du pli cutané (triceps)", unit="mm", mini=5, maxi=100, defaut=29, pas=1,
                          ref=29, manquant=True, aide="Mesure souvent non faite : cochez « inconnue » dans ce cas."),
    "Insulin": dict(label="Insuline (2 h)", unit="µU/mL", mini=10, maxi=900, defaut=125, pas=1,
                    ref=125, manquant=True, aide="Mesure souvent non faite : cochez « inconnue » dans ce cas."),
    "BMI": dict(label="IMC (indice de masse corporelle)", unit="kg/m²", mini=10.0, maxi=70.0, defaut=28.0, pas=0.1,
                ref=32.3, manquant=True, aide="Poids (kg) divisé par la taille (m) au carré."),
    "DiabetesPedigreeFunction": dict(label="Antécédents familiaux (score)", unit="", mini=0.05, maxi=2.50, defaut=0.40,
                                     pas=0.01, ref=0.37, manquant=False,
                                     aide="Score estimant l'influence de l'hérédité (0,1 = faible ; 1,5 = forte)."),
    "Age": dict(label="Âge", unit="ans", mini=18, maxi=100, defaut=30, pas=1,
                ref=29, manquant=False, aide="Âge en années."),
}
FEATURES = list(VARIABLES)

EXEMPLES = {
    "Profil à faible risque": dict(Pregnancies=1, Glucose=92, BloodPressure=68, SkinThickness=22, Insulin=85,
                                   BMI=23.0, DiabetesPedigreeFunction=0.25, Age=24),
    "Profil à risque élevé": dict(Pregnancies=6, Glucose=168, BloodPressure=88, SkinThickness=35, Insulin=240,
                                  BMI=36.5, DiabetesPedigreeFunction=0.95, Age=52),
    "Cas du notebook (glycémie 150, IMC 35, 45 ans)": dict(Pregnancies=3, Glucose=150, BloodPressure=72,
                                                           SkinThickness=29, Insulin=125, BMI=35.0,
                                                           DiabetesPedigreeFunction=0.37, Age=45),
    "Mesures manquantes (insuline et pli cutané)": dict(Pregnancies=2, Glucose=135, BloodPressure=80,
                                                        SkinThickness=None, Insulin=None, BMI=31.0,
                                                        DiabetesPedigreeFunction=0.5, Age=38),
}

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .carte {border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: .8rem; border: 1px solid;}
    .carte h2 {margin: 0 0 .2rem 0; font-size: 1.5rem;}
    .carte p {margin: .2rem 0 0 0;}
    .vert  {background: #E8F7EE; border-color: #2F9E5B; color: #14532D;}
    .orange{background: #FFF4E0; border-color: #E08A00; color: #7A4A00;}
    .rouge {background: #FDEBEF; border-color: #D6336C; color: #7F1D3F;}
    .barre {position: relative; height: 22px; border-radius: 11px; margin: .6rem 0 1.4rem 0;
            background: linear-gradient(90deg, #2F9E5B 0%, #E0B400 50%, #D6336C 100%);}
    .curseur {position: absolute; top: -5px; width: 5px; height: 32px; background: #0B1F4B; border-radius: 3px;}
    .seuil {position: absolute; top: -5px; width: 0; height: 32px; border-left: 3px dashed #0B1F4B;}
    .legende {display: flex; justify-content: space-between; font-size: .8rem; opacity: .7; margin-top: -1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Fonctions utiles
# ----------------------------------------------------------------------------
def modele_valide(modele) -> bool:
    """Vérifie que le modèle fonctionne avec la version de scikit-learn installée."""
    try:
        test = pd.DataFrame([{f: float(v["defaut"]) for f, v in VARIABLES.items()}])
        modele.predict_proba(test[FEATURES])
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner="Entraînement du modèle (quelques secondes)…")
def entrainer(donnees: pd.DataFrame):
    """Même méthode que le notebook : nettoyage, 3 modèles, on garde le meilleur (ROC-AUC en validation croisée)."""
    df = donnees.drop_duplicates().copy()
    df[COLS_ZEROS] = df[COLS_ZEROS].replace(0, np.nan)
    X, y = df[FEATURES], df["Outcome"]
    candidats = {
        "Régression logistique": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                  random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    meilleur, meilleur_score = None, -1
    for est in candidats.values():
        pipe = Pipeline([
            ("prep", ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                          ("scaler", StandardScaler())]), FEATURES)])),
            ("model", est),
        ])
        score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        if score > meilleur_score:
            meilleur, meilleur_score = pipe, score
    meilleur.fit(X, y)
    return meilleur


def obtenir_modele(joblib_envoye, csv_envoye):
    """1) .joblib envoyé  2) modele_diabete.joblib valide  3) réentraînement depuis diabetes.csv."""
    if joblib_envoye is not None:
        try:
            m = joblib.load(joblib_envoye)
            if modele_valide(m):
                return m, "Modèle envoyé"
        except Exception:
            pass
        st.sidebar.warning("Le fichier .joblib envoyé est incompatible avec ta version de scikit-learn.")
    if MODEL_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            if modele_valide(m):
                return m, "modele_diabete.joblib"
        except Exception:
            pass
    donnees = None
    if csv_envoye is not None:
        donnees = pd.read_csv(csv_envoye)
    elif CSV_PATH.exists():
        donnees = pd.read_csv(CSV_PATH)
    if donnees is not None:
        m = entrainer(donnees)
        try:
            joblib.dump(m, MODEL_PATH)
        except Exception:
            pass
        return m, "Modèle réentraîné depuis diabetes.csv"
    return None, None


ECHELLE = alt.Scale(domain=NOMS_CLASSES, range=["#1E55B3", "#D6336C"])


def charger_donnees(csv_envoye):
    """Jeu de données de référence (diabetes.csv) pour les onglets Données et Performances."""
    try:
        if csv_envoye is not None:
            csv_envoye.seek(0)
            d = pd.read_csv(csv_envoye)
        elif CSV_PATH.exists():
            d = pd.read_csv(CSV_PATH)
        else:
            return None
    except Exception:
        return None
    return d if set(FEATURES + ["Outcome"]).issubset(d.columns) else None


def nettoyer(donnees: pd.DataFrame) -> pd.DataFrame:
    """Comme dans le notebook : sans doublons, et 0 impossible → valeur manquante."""
    d = donnees.drop_duplicates().copy()
    d[COLS_ZEROS] = d[COLS_ZEROS].replace(0, np.nan)
    return d


@st.cache_resource(show_spinner="Calcul des performances (validation croisée)…")
def evaluer(_modele, donnees, cle):
    """Prédictions « honnêtes » (validation croisée 5 plis) + importance des variables."""
    d = nettoyer(donnees)
    X, y = d[FEATURES], d["Outcome"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(clone(_modele), X, y, cv=cv, method="predict_proba")[:, 1]
    imp = permutation_importance(_modele, X, y, n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc")
    importances = pd.Series(imp.importances_mean, index=[VARIABLES[f]["label"] for f in FEATURES])
    return y.values, proba, importances.sort_values(ascending=False)


def probabilite(modele, donnees: pd.DataFrame) -> np.ndarray:
    return modele.predict_proba(donnees[FEATURES])[:, 1]


def contributions(modele, patient: pd.DataFrame) -> pd.Series:
    """Explication simple : de combien la probabilité change si on remplace
    une variable par une valeur typique ? (positif = pousse vers « diabète »)"""
    p_patient = probabilite(modele, patient)[0]
    lignes = []
    for f in FEATURES:
        test = patient.copy()
        test[f] = float(VARIABLES[f]["ref"])
        lignes.append(p_patient - probabilite(modele, test)[0])
    return pd.Series(lignes, index=[VARIABLES[f]["label"] for f in FEATURES]).sort_values()


def niveau(p: float, seuil: float):
    if p >= seuil:
        return "rouge", "Risque élevé", "Le modèle annonce « diabète ». Un examen de confirmation (glycémie à jeun, HbA1c) est recommandé."
    if p >= max(seuil - 0.15, 0.05):
        return "orange", "Risque modéré", "Sous le seuil de décision, mais proche : à surveiller, surtout si des mesures sont inconnues."
    return "vert", "Risque faible", "Le modèle n'annonce pas « diabète » pour ce profil."


def afficher_resultat(p: float, seuil: float):
    classe, titre, texte = niveau(p, seuil)
    st.markdown(
        f"""
        <div class="carte {classe}">
          <h2>{titre} : {p:.0%}</h2>
          <p>{texte}</p>
        </div>
        <div class="barre">
          <div class="seuil" style="left:{seuil*100:.1f}%"></div>
          <div class="curseur" style="left:calc({p*100:.1f}% - 2px)"></div>
        </div>
        <div class="legende"><span>0 %</span><span>trait pointillé = seuil ({seuil:.0%})</span><span>100 %</span></div>
        """,
        unsafe_allow_html=True,
    )


def initialiser_etat():
    for f, v in VARIABLES.items():
        st.session_state.setdefault(f"v_{f}", v["defaut"])
        st.session_state.setdefault(f"na_{f}", False)


def appliquer_exemple():
    exemple = EXEMPLES.get(st.session_state["choix_exemple"])
    if exemple is None:
        return
    for f, val in exemple.items():
        if val is None:
            st.session_state[f"na_{f}"] = True
        else:
            st.session_state[f"na_{f}"] = False
            st.session_state[f"v_{f}"] = type(VARIABLES[f]["defaut"])(val)


# ----------------------------------------------------------------------------
# Barre latérale
# ----------------------------------------------------------------------------
initialiser_etat()

with st.sidebar:
    st.title("💉 Dépistage du diabète")
    st.caption("UMEF AI Bootcamp · GalsenAI")

    st.subheader("Modèle")
    with st.expander("Fichiers (si besoin)"):
        csv_envoye = st.file_uploader("diabetes.csv", type="csv", key="csv_modele")
        joblib_envoye = st.file_uploader("modele_diabete.joblib", type="joblib")
    modele, origine = obtenir_modele(joblib_envoye, csv_envoye)
    if modele is None:
        st.error("Aucun modèle utilisable. Place `diabetes.csv` dans le même dossier que `app.py` "
                 "(ou envoie-le dans « Fichiers » ci-dessus) : le modèle sera créé automatiquement.")
        st.stop()
    nom_modele = type(modele.steps[-1][1]).__name__ if hasattr(modele, "steps") else type(modele).__name__
    st.success(f"{nom_modele}\n\nSource : {origine}")
    donnees_ref = charger_donnees(csv_envoye)

    st.subheader("Seuil de décision")
    seuil = st.slider("Annoncer « diabète » à partir de", 0.05, 0.95, 0.40, 0.05,
                      help="Plus le seuil est bas, plus on détecte de cas (rappel ↑) mais plus on alerte "
                           "des personnes saines (précision ↓). Le notebook recommande un bon rappel.")
    st.caption("Un faux négatif (cas raté) est plus grave qu'un faux positif (examen de confirmation).")

    st.divider()
    st.warning("⚠️ Exercice pédagogique. Ne remplace pas un avis médical. "
               "Modèle entraîné sur 768 femmes du jeu *Pima Indians*.")

# ----------------------------------------------------------------------------
# Onglets
# ----------------------------------------------------------------------------
st.title("Tester le modèle de dépistage")
onglet_patient, onglet_lot, onglet_donnees, onglet_perf, onglet_info = st.tabs(
    ["🧑‍⚕️ Une patiente", "📄 Fichier CSV", "📊 Données", "📈 Performances", "ℹ️ À propos"])

# ------------------------------- Une patiente -------------------------------
with onglet_patient:
    st.selectbox("Charger un profil d'exemple", ["—"] + list(EXEMPLES), key="choix_exemple",
                 on_change=appliquer_exemple)

    col_saisie, col_resultat = st.columns([3, 2], gap="large")

    valeurs = {}
    with col_saisie:
        st.subheader("Mesures")
        for f, v in VARIABLES.items():
            c1, c2 = st.columns([3, 1])
            unite = f" ({v['unit']})" if v["unit"] else ""
            inconnue = False
            if v["manquant"]:
                with c2:
                    st.write("")  # aligne la case avec le champ
                    inconnue = st.checkbox("Inconnue", key=f"na_{f}", help="Mesure non faite : le modèle utilisera une valeur typique.")
            with c1:
                saisie = st.number_input(
                    v["label"] + unite, min_value=v["mini"], max_value=v["maxi"], step=v["pas"],
                    key=f"v_{f}", disabled=inconnue, help=v["aide"],
                    format="%.2f" if isinstance(v["pas"], float) and v["pas"] < 0.1 else ("%.1f" if isinstance(v["pas"], float) else "%d"),
                )
            valeurs[f] = np.nan if inconnue else float(saisie)

    patient = pd.DataFrame([valeurs])[FEATURES]
    p = float(probabilite(modele, patient)[0])

    with col_resultat:
        st.subheader("Résultat")
        afficher_resultat(p, seuil)

        manquantes = [VARIABLES[f]["label"] for f in FEATURES if np.isnan(valeurs[f])]
        if manquantes:
            st.info("Mesures inconnues, remplacées par une valeur typique : " + ", ".join(manquantes) + ".")
        if np.isnan(valeurs["Glucose"]):
            st.warning("Sans la glycémie, la prédiction est beaucoup moins fiable.")

        st.markdown("**Ce qui pèse dans la prédiction**")
        contrib = contributions(modele, patient)
        st.bar_chart(contrib, horizontal=True, height=300)
        st.caption("Effet de chaque variable par rapport à une valeur typique. "
                   "Positif : pousse vers « diabète » ; négatif : éloigne du diabète.")

# --------------------------------- Fichier CSV ---------------------------------
with onglet_lot:
    st.write("Envoie un fichier CSV (par exemple `diabetes.csv`) pour tester le modèle sur beaucoup de cas à la fois. "
             "Colonnes attendues : " + ", ".join(f"`{f}`" for f in FEATURES) + ". "
             "Si la colonne `Outcome` est présente, les performances sont calculées.")
    csv = st.file_uploader("Fichier CSV", type="csv", key="csv_lot")

    if csv is not None:
        donnees = pd.read_csv(csv)
        absentes = [f for f in FEATURES if f not in donnees.columns]
        if absentes:
            st.error("Colonnes manquantes : " + ", ".join(absentes))
        else:
            # Même nettoyage que dans le notebook : 0 impossible → valeur manquante
            propre = donnees.copy()
            propre[COLS_ZEROS] = propre[COLS_ZEROS].replace(0, np.nan)
            probas = probabilite(modele, propre)

            resultat = donnees.copy()
            resultat["Probabilité diabète"] = probas.round(3)
            resultat["Prédiction"] = np.where(probas >= seuil, NOMS_CLASSES[1], NOMS_CLASSES[0])

            c1, c2, c3 = st.columns(3)
            c1.metric("Lignes analysées", len(resultat))
            c2.metric("Annoncées « diabète »", int((probas >= seuil).sum()))
            c3.metric("Part annoncée", f"{(probas >= seuil).mean():.0%}")

            if "Outcome" in donnees.columns:
                vrai = donnees["Outcome"].astype(int)
                pred = (probas >= seuil).astype(int)
                st.subheader(f"Performances au seuil {seuil:.2f}")
                m = st.columns(5)
                m[0].metric("Accuracy", f"{accuracy_score(vrai, pred):.2f}")
                m[1].metric("Précision", f"{precision_score(vrai, pred, zero_division=0):.2f}")
                m[2].metric("Rappel", f"{recall_score(vrai, pred, zero_division=0):.2f}")
                m[3].metric("F1", f"{f1_score(vrai, pred, zero_division=0):.2f}")
                m[4].metric("ROC-AUC", f"{roc_auc_score(vrai, probas):.2f}")
                matrice = pd.crosstab(vrai.map(lambda i: "Réel : " + NOMS_CLASSES[i]),
                                      pd.Series(pred, index=vrai.index).map(lambda i: "Prédit : " + NOMS_CLASSES[i]))
                st.dataframe(matrice, use_container_width=True)
                st.caption("Attention : si ce fichier contient les données d'entraînement du modèle, "
                           "les scores sont trop optimistes.")

            st.dataframe(resultat, use_container_width=True, height=380)
            st.download_button("⬇️ Télécharger les résultats (CSV)",
                               resultat.to_csv(index=False).encode("utf-8"),
                               file_name="predictions_diabete.csv", mime="text/csv")

# ----------------------------------- Données -----------------------------------
with onglet_donnees:
    if donnees_ref is None:
        st.info("Place `diabetes.csv` à côté de `app.py` (ou envoie-le dans « Fichiers » de la barre latérale) "
                "pour voir les informations sur les données.")
    else:
        brut = donnees_ref
        net = nettoyer(brut)
        net["Statut"] = net["Outcome"].astype(int).map(lambda i: NOMS_CLASSES[i])

        st.subheader("Vue d'ensemble")
        c = st.columns(5)
        c[0].metric("Lignes", f"{len(brut):,}".replace(",", " "))
        c[1].metric("Colonnes", brut.shape[1], help="8 mesures + la cible « Outcome »")
        c[2].metric("Doublons", int(brut.duplicated().sum()))
        c[3].metric("Zéros impossibles", int((brut[COLS_ZEROS] == 0).sum().sum()),
                    help="Mesures non faites codées par 0 (glycémie, tension, pli cutané, insuline, IMC).")
        c[4].metric("Diabétiques", f"{net['Outcome'].mean():.0%}")
        c = st.columns(4)
        c[0].metric("Âge moyen", f"{net['Age'].mean():.0f} ans", help=f"De {net['Age'].min():.0f} à {net['Age'].max():.0f} ans")
        c[1].metric("Glycémie médiane", f"{net['Glucose'].median():.0f} mg/dL")
        c[2].metric("IMC médian", f"{net['BMI'].median():.1f}")
        c[3].metric("Valeurs textuelles", 0 if brut.select_dtypes(exclude="number").shape[1] == 0
                    else brut.select_dtypes(exclude="number").shape[1],
                    help="Colonnes non numériques : aucune, toutes les variables sont des nombres.")

        with st.expander("Aperçu du tableau et statistiques", expanded=False):
            st.dataframe(brut.head(10), use_container_width=True)
            resume = pd.DataFrame({
                "Manquantes (nb)": net[FEATURES].isna().sum(),
                "Manquantes (%)": (net[FEATURES].isna().mean() * 100).round(1),
                "Moyenne": net[FEATURES].mean().round(2),
                "Médiane": net[FEATURES].median().round(2),
                "Min": net[FEATURES].min().round(2),
                "Max": net[FEATURES].max().round(2),
            })
            st.dataframe(resume, use_container_width=True)
            st.caption("Statistiques après nettoyage (doublons retirés, zéros impossibles remplacés par des valeurs manquantes).")

        st.subheader("Répartition et qualité des données")
        g1, g2 = st.columns(2)
        with g1:
            eff = (net["Statut"].value_counts().reindex(NOMS_CLASSES).rename_axis("Statut")
                   .reset_index(name="Nombre"))
            eff["Part"] = eff["Nombre"] / eff["Nombre"].sum()
            st.markdown("**Répartition de la cible** (classes déséquilibrées)")
            st.altair_chart(
                alt.Chart(eff).mark_bar().encode(
                    x=alt.X("Statut:N", sort=NOMS_CLASSES, title=None),
                    y=alt.Y("Nombre:Q"),
                    color=alt.Color("Statut:N", scale=ECHELLE, legend=None),
                    tooltip=["Statut", "Nombre", alt.Tooltip("Part:Q", format=".0%")]),
                use_container_width=True)
        with g2:
            manq = (brut[COLS_ZEROS] == 0).mean().mul(100).round(1).rename_axis("Variable").reset_index(name="% de zéros")
            manq["Variable"] = manq["Variable"].map(lambda f: VARIABLES[f]["label"])
            st.markdown("**Valeurs manquantes cachées** (zéros impossibles, en %)")
            st.altair_chart(
                alt.Chart(manq).mark_bar(color="#E08A00").encode(
                    y=alt.Y("Variable:N", sort="-x", title=None),
                    x=alt.X("% de zéros:Q"),
                    tooltip=["Variable", "% de zéros"]),
                use_container_width=True)

        st.subheader("Comparer les deux groupes")
        var = st.selectbox("Variable à explorer", FEATURES, key="var_graph",
                           format_func=lambda f: VARIABLES[f]["label"])
        etiquette = VARIABLES[var]["label"]
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("**Distribution**")
            st.altair_chart(
                alt.Chart(net).mark_bar(opacity=0.65).encode(
                    x=alt.X(f"{var}:Q", bin=alt.Bin(maxbins=30), title=etiquette),
                    y=alt.Y("count()", stack=None, title="Nombre"),
                    color=alt.Color("Statut:N", scale=ECHELLE, title=None)),
                use_container_width=True)
        with h2:
            st.markdown("**Boîte à moustaches**")
            st.altair_chart(
                alt.Chart(net).mark_boxplot(extent="min-max", size=60).encode(
                    x=alt.X("Statut:N", sort=NOMS_CLASSES, title=None),
                    y=alt.Y(f"{var}:Q", title=etiquette, scale=alt.Scale(zero=False)),
                    color=alt.Color("Statut:N", scale=ECHELLE, legend=None)),
                use_container_width=True)

        st.subheader("Liens entre les variables")
        corr = net[FEATURES + ["Outcome"]].corr()
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Corrélation avec le diabète**")
            cc = corr["Outcome"].drop("Outcome").rename_axis("Variable").reset_index(name="Corrélation")
            cc["Variable"] = cc["Variable"].map(lambda f: VARIABLES[f]["label"])
            st.altair_chart(
                alt.Chart(cc).mark_bar().encode(
                    y=alt.Y("Variable:N", sort="-x", title=None),
                    x=alt.X("Corrélation:Q", scale=alt.Scale(domain=[0, 1])),
                    color=alt.value("#1E55B3"),
                    tooltip=["Variable", alt.Tooltip("Corrélation:Q", format=".2f")]),
                use_container_width=True)
        with k2:
            st.markdown("**Matrice de corrélation**")
            cm = corr.loc[FEATURES, FEATURES].stack().reset_index()
            cm.columns = ["a", "b", "corr"]
            base = alt.Chart(cm).encode(x=alt.X("a:N", title=None), y=alt.Y("b:N", title=None))
            heat = base.mark_rect().encode(
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), legend=None))
            txt = base.mark_text(size=10).encode(
                text=alt.Text("corr:Q", format=".1f"),
                color=alt.condition("abs(datum.corr) > 0.5", alt.value("white"), alt.value("black")))
            st.altair_chart((heat + txt).properties(height=300), use_container_width=True)
        st.caption("La glycémie est la variable la plus liée au diabète ; l'âge et le nombre de grossesses sont liés entre eux.")

# --------------------------------- Performances ---------------------------------
with onglet_perf:
    if donnees_ref is None:
        st.info("Place `diabetes.csv` à côté de `app.py` (ou envoie-le dans « Fichiers ») pour voir les performances.")
    else:
        try:
            y_vrai, proba_cv, importances = evaluer(modele, donnees_ref, origine)
            ok = True
        except Exception as erreur:
            ok = False
            st.warning(f"Impossible de calculer les performances : {erreur}")

        if ok:
            pred = (proba_cv >= seuil).astype(int)
            st.caption("Évaluation honnête : chaque patiente est prédite par un modèle qui ne l'a jamais vue "
                       f"(validation croisée à 5 plis). Les résultats dépendent du seuil choisi ({seuil:.2f}, barre latérale).")
            m = st.columns(5)
            m[0].metric("Accuracy", f"{accuracy_score(y_vrai, pred):.2f}")
            m[1].metric("Précision", f"{precision_score(y_vrai, pred, zero_division=0):.2f}")
            m[2].metric("Rappel", f"{recall_score(y_vrai, pred, zero_division=0):.2f}")
            m[3].metric("F1", f"{f1_score(y_vrai, pred, zero_division=0):.2f}")
            m[4].metric("ROC-AUC", f"{roc_auc_score(y_vrai, proba_cv):.2f}")

            tn, fp, fn, tp = confusion_matrix(y_vrai, pred, labels=[0, 1]).ravel()
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Matrice de confusion**")
                mc = pd.DataFrame({
                    "Réel": ["Réel : " + NOMS_CLASSES[0]] * 2 + ["Réel : " + NOMS_CLASSES[1]] * 2,
                    "Prédit": ["Prédit : " + NOMS_CLASSES[0], "Prédit : " + NOMS_CLASSES[1]] * 2,
                    "Nombre": [tn, fp, fn, tp]})
                base = alt.Chart(mc).encode(x=alt.X("Prédit:N", title=None, axis=alt.Axis(labelAngle=0)),
                                            y=alt.Y("Réel:N", title=None))
                st.altair_chart(
                    (base.mark_rect().encode(color=alt.Color("Nombre:Q", scale=alt.Scale(scheme="blues"), legend=None))
                     + base.mark_text(size=22).encode(text="Nombre:Q",
                                                      color=alt.condition("datum.Nombre > 150", alt.value("white"), alt.value("black")))
                     ).properties(height=280), use_container_width=True)
                st.caption(f"{fn} cas manqués (faux négatifs) et {fp} fausses alertes (faux positifs).")
            with p2:
                st.markdown("**Courbe ROC**")
                fpr, tpr, _ = roc_curve(y_vrai, proba_cv)
                roc = pd.DataFrame({"Faux positifs (fausses alertes)": fpr, "Vrais positifs (rappel)": tpr})
                courbe = alt.Chart(roc).mark_line(color="#1E55B3", strokeWidth=3).encode(
                    x=alt.X("Faux positifs (fausses alertes):Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Vrais positifs (rappel):Q", scale=alt.Scale(domain=[0, 1])))
                hasard = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                    color="gray", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
                point = alt.Chart(pd.DataFrame({
                    "Faux positifs (fausses alertes)": [fp / max(fp + tn, 1)],
                    "Vrais positifs (rappel)": [tp / max(tp + fn, 1)]})).mark_point(
                    color="#D6336C", size=140, filled=True).encode(
                    x="Faux positifs (fausses alertes):Q", y="Vrais positifs (rappel):Q")
                st.altair_chart((hasard + courbe + point).properties(height=280), use_container_width=True)
                st.caption("Le point rose indique le seuil actuel ; la ligne pointillée représente le hasard.")

            q1, q2 = st.columns(2)
            with q1:
                st.markdown("**Importance des variables**")
                imp_df = importances.rename_axis("Variable").reset_index(name="Importance")
                st.altair_chart(
                    alt.Chart(imp_df).mark_bar(color="#1E55B3").encode(
                        y=alt.Y("Variable:N", sort="-x", title=None),
                        x=alt.X("Importance:Q", title="Baisse du ROC-AUC si on mélange la variable"),
                        tooltip=["Variable", alt.Tooltip("Importance:Q", format=".3f")]),
                    use_container_width=True)
            with q2:
                st.markdown("**Effet du seuil de décision**")
                lignes = []
                for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    pt = (proba_cv >= t).astype(int)
                    lignes.append({"Seuil": t, "Annoncés « diabète »": int(pt.sum()),
                                   "Précision": round(precision_score(y_vrai, pt, zero_division=0), 2),
                                   "Rappel": round(recall_score(y_vrai, pt, zero_division=0), 2),
                                   "F1": round(f1_score(y_vrai, pt, zero_division=0), 2)})
                tableau = pd.DataFrame(lignes).set_index("Seuil")
                st.dataframe(tableau, use_container_width=True)
                st.line_chart(tableau[["Précision", "Rappel", "F1"]], height=200)

# ---------------------------------- À propos ----------------------------------
with onglet_info:
    st.markdown(
        """
**Comment ça marche ?**
Le fichier `modele_diabete.joblib` contient tout le pipeline du notebook : remplacement des valeurs manquantes
par la médiane, mise à l'échelle, puis le meilleur modèle (choisi sur le ROC-AUC en validation croisée).
L'application lui envoie un tableau avec les 8 mesures et récupère une **probabilité de diabète**.

**Valeurs manquantes.** Dans les données d'origine, un 0 pour la glycémie, la tension, le pli cutané, l'insuline
ou l'IMC veut dire « non mesuré ». Ici, cocher **Inconnue** envoie une valeur manquante, que le pipeline remplace par la médiane.

**Seuil de décision.** Le modèle donne une probabilité ; c'est le seuil (barre latérale) qui la transforme en
« diabétique / non diabétique ». Un seuil plus bas détecte plus de cas, au prix de plus de fausses alertes.

**Limites.**
- Données : 768 femmes d'une population précise (Pima). Elles ne se généralisent pas automatiquement à la population sénégalaise.
- Le modèle repère un risque ; il ne pose pas de diagnostic.
- Le graphique « Ce qui pèse dans la prédiction » est une explication simplifiée, pas une relation de cause à effet.
        """
    )
