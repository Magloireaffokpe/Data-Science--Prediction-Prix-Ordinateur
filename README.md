# 📄 Prédiction du prix des ordinateurs portables

## 🔧 Contexte et objectifs
Dans un contexte d’essor du e-commerce et de l’informatique décisionnelle, ce projet propose une **application intelligente** capable d’estimer automatiquement le prix d’un ordinateur portable à partir de ses caractéristiques techniques.  
Objectif : fournir une solution prédictive rapide et fiable pour les vendeurs, distributeurs et analystes du marché.

---

## 🧪 Étapes principales du projet
- **Analyse exploratoire (EDA)** : nettoyage des données, création de variables dérivées (ppi, tactile, IPS…).  
- **Prétraitement & encodage** : LabelEncoder + StandardScaler, sauvegarde des objets (`encoders.pkl`, `scaler.pkl`).  
- **Modélisation** : Random Forest Regressor (scikit-learn) entraîné sur un split 80/20.  
- **Interface utilisateur** : développée avec **Streamlit**, proposant des champs d’entrée simples (marque, RAM, CPU, poids, écran, etc.) et retournant une estimation du prix en **FCFA**.

---

## 💾 Fichiers générés
- `app.py` : application Streamlit.  
- `laptop_price_model.pkl` : modèle Random Forest entraîné.  
- `encoders.pkl` : dictionnaire des encodeurs.  
- `scaler.pkl` : scaler utilisé pour la normalisation.  

---

## ✅ Résultats et avantages
- Interface intuitive et ergonomique.  
- Temps de réponse rapide.  
- Prédictions fiables dans une marge d’erreur acceptable.  
- Possibilité d’amélioration avec de nouvelles données.  

---

## 📈 Pistes d’amélioration
- Ajouter une visualisation interactive des prix par marque/configuration.  
- Intégrer un système de feedback utilisateur.  
- Déployer l’application en ligne (Streamlit Cloud ou serveur dédié).  

---

## 📄 Rapport complet
Le rapport technique détaillé est disponible dans [`docs/Rapport_Prévision_Prix_Ordinateurs.pdf`](docs/Rapport_Prévision_Prix_Ordinateurs.pdf).

---

## 👤 Auteur
**Magloire AFFOKPE**  
Étudiant en informatique de gestion | Intéressé par l’IA, la Data Science et le Développement
