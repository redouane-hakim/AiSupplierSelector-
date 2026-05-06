# Datasets pour le projet fournisseur IA

à cause de stockage veuillez télécharger les datasets  à fin faire une prediction pour votre data.
si non utiliser directement le model créé par notre training

## 1) TED CSV subset (source officielle UE)
- Description : avis de marchés publics et avis d'attribution, avec acheteur, fournisseur, montant, procédure, critères d'attribution, CPV, pays.
- Période : 2006-01-01 à 2021-12-31.
- Lien officiel : https://data.europa.eu/data/datasets/ted-1?locale=en
- Documentation CSV : https://data.europa.eu/euodp/en/data/storage/f/2022-02-14T122429/TED%28csv%29_data_information_v3.4.pdf
- API miroir : https://api.store/eu-institutions-api/directorate-general-for-internal-market-industry-entrepreneurship-and-smes-api/tenders-electronic-daily-ted-csv-subset-public-procurement-notices-api

## 2) Global Contract-level Public Procurement Dataset (GPPD)
- Description : dataset harmonisé couvrant 42 pays, 72+ millions de contrats, informations acheteurs, fournisseurs, produits, prix, dates et indicateurs de risque.
- Article : https://www.sciencedirect.com/science/article/pii/S2352340924003810
- Données : https://data.mendeley.com/datasets/fwzpywbhgw/3

## 3) A Comprehensive Data Set of US Federal Procurement (1979-2023)
- Description : quasi 100 millions d’actions contractuelles, plus de 200 variables, format Parquet.
- Article : https://pmc.ncbi.nlm.nih.gov/articles/PMC12328818/
- Bundle Figshare : https://springernature.figshare.com/articles/dataset/A_Comprehensive_Data_Set_of_US_Federal_Procurement_1979-2023/28057043

## Recommandation pratique pour votre soutenance

### Option la plus simple et crédible
Utiliser **TED comme dataset principal** pour construire le moteur ML, puis ajouter ensuite un fichier interne `supplier_kpis.csv` contenant :
- OTIF
- défauts PPM
- score qualité audit
- score flexibilité
- score innovation
- score ESG

### Pourquoi ?
Les gros datasets publics sont excellents pour apprendre :
- qui gagne quel type d'appel d'offres
- à quel prix
- dans quel secteur
- dans quel pays
- avec quel historique

Mais ils sont souvent insuffisants pour les KPI opérationnels fins (OTIF réel, SAV, flexibilité réelle). Il faut donc un **modèle hybride** : public + données internes.
