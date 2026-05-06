# Package IA - Sélection intelligente des fournisseurs

Ce package contient trois livrables :

1. `notebooks/supplier_selection_notebook.ipynb`
   - notebook de préparation des données, fusion, nettoyage, création des features, entraînement, évaluation et export du meilleur modèle.
2. `docs/datasets_links.md`
   - liens et notes sur les datasets recommandés.
3. `app/`
   - une application web simple avec backend FastAPI et frontend HTML/CSS/JS.

## Architecture recommandée

- **Données publiques massives** : TED / GPPD / FPDS-figshare
- **Données internes à ajouter plus tard** : OTIF, PPM, audits qualité, KPI ESG, SAV, flexibilité réelle
- **Sortie du modèle** : probabilité de pertinence fournisseur + score métier multicritère

## Lancer l'application

```bash
cd app/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Ensuite ouvrir `http://127.0.0.1:8000`.

## Remarque importante

Le modèle fourni dans `app/backend/artifacts/best_model.joblib` est un **modèle** entraîné sur les données des datasets réalistes pour rendre l'application directement utilisable.

Pour votre projet , il faut **remplacer ce modèle** par celui exporté par le notebook après exécution sur vos datasets ou des datasets publics  + vos KPI fournisseurs internes.
