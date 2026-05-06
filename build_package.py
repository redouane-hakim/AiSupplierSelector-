from __future__ import annotations

import json
from pathlib import Path
import textwrap

import joblib
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/mnt/data/supplier_ai_package')
BACKEND = ROOT / 'app' / 'backend'
FRONTEND = ROOT / 'app' / 'frontend'
ART = BACKEND / 'artifacts'
DOCS = ROOT / 'docs'
NOTEBOOKS = ROOT / 'notebooks'

SECTOR_WEIGHTS = {
    'global': {'quality': 0.28, 'cost': 0.20, 'delivery': 0.17, 'innovation': 0.13, 'flexibility': 0.12, 'esg': 0.10},
    'automobile': {'quality': 0.30, 'cost': 0.22, 'delivery': 0.20, 'innovation': 0.08, 'flexibility': 0.12, 'esg': 0.08},
    'electronique': {'quality': 0.22, 'cost': 0.18, 'delivery': 0.15, 'innovation': 0.25, 'flexibility': 0.10, 'esg': 0.10},
}

FEATURES = [
    'unit_cost', 'logistics_cost', 'lead_time_days', 'quality_score', 'delivery_score',
    'flexibility_score', 'innovation_score', 'esg_score', 'historical_win_rate',
    'historical_volume', 'defect_rate_ppm', 'risk_score', 'budget_fit', 'margin_rate',
    'cost_advantage', 'quality_proxy', 'delivery_proxy', 'innovation_proxy', 'esg_proxy'
]


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def make_demo_training_data(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unit_cost = rng.uniform(20, 300, n)
    logistics_cost = rng.uniform(0.5, 30, n)
    lead_time_days = rng.integers(3, 60, n)
    quality_score = rng.uniform(45, 98, n)
    delivery_score = rng.uniform(40, 99, n)
    flexibility_score = rng.uniform(35, 98, n)
    innovation_score = rng.uniform(20, 99, n)
    esg_score = rng.uniform(15, 98, n)
    historical_win_rate = rng.uniform(0.01, 0.85, n)
    historical_volume = rng.integers(1, 5000, n)
    defect_rate_ppm = rng.uniform(5, 4000, n)
    risk_score = rng.uniform(2, 95, n)
    budget_per_unit = rng.uniform(35, 330, n)
    selling_price_per_unit = budget_per_unit * rng.uniform(1.05, 1.8, n)

    total_cost_unit = unit_cost + logistics_cost
    budget_fit = clip01(1 - np.maximum(total_cost_unit - budget_per_unit, 0) / np.maximum(budget_per_unit, 1))
    margin_rate = clip01((selling_price_per_unit - total_cost_unit) / np.maximum(selling_price_per_unit, 1))
    cost_advantage = clip01(1 - total_cost_unit / np.maximum(budget_per_unit * 1.25, 1))
    quality_proxy = clip01(0.7 * (quality_score / 100) + 0.3 * (1 - defect_rate_ppm / 4000))
    delivery_proxy = clip01(0.7 * (delivery_score / 100) + 0.3 * (1 - lead_time_days / 60))
    innovation_proxy = clip01(innovation_score / 100)
    esg_proxy = clip01(esg_score / 100)

    latent = (
        1.7 * quality_proxy +
        1.2 * delivery_proxy +
        1.0 * margin_rate +
        0.8 * innovation_proxy +
        0.6 * esg_proxy +
        0.55 * (flexibility_score / 100) +
        0.65 * historical_win_rate +
        0.18 * np.log1p(historical_volume) / np.log(5001) -
        1.2 * (risk_score / 100) -
        0.9 * (defect_rate_ppm / 4000) +
        0.7 * budget_fit +
        0.35 * cost_advantage
    )
    latent += rng.normal(0, 0.22, n)
    prob = 1 / (1 + np.exp(-(latent - 2.1)))
    y = rng.binomial(1, prob)

    df = pd.DataFrame({
        'unit_cost': unit_cost,
        'logistics_cost': logistics_cost,
        'lead_time_days': lead_time_days,
        'quality_score': quality_score,
        'delivery_score': delivery_score,
        'flexibility_score': flexibility_score,
        'innovation_score': innovation_score,
        'esg_score': esg_score,
        'historical_win_rate': historical_win_rate,
        'historical_volume': historical_volume,
        'defect_rate_ppm': defect_rate_ppm,
        'risk_score': risk_score,
        'budget_fit': budget_fit,
        'margin_rate': margin_rate,
        'cost_advantage': cost_advantage,
        'quality_proxy': quality_proxy,
        'delivery_proxy': delivery_proxy,
        'innovation_proxy': innovation_proxy,
        'esg_proxy': esg_proxy,
        'label': y,
    })
    return df


def top_k_hit_rate(y_true: np.ndarray, group_ids: np.ndarray, scores: np.ndarray, k: int = 1) -> float:
    temp = pd.DataFrame({'y': y_true, 'group': group_ids, 'score': scores})
    hits = []
    for _, grp in temp.groupby('group'):
        grp = grp.sort_values('score', ascending=False).head(k)
        hits.append(int((grp['y'] == 1).any()))
    return float(np.mean(hits)) if hits else 0.0


def train_demo_model() -> dict:
    df = make_demo_training_data()
    X = df[FEATURES]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=800, n_jobs=None))
        ]),
        'rf': RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=8, random_state=42, n_jobs=1),
        'hgb': HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=240, random_state=42),
    }

    results = {}
    best_name = None
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        blended = 0.6 * auc + 0.4 * ap
        results[name] = {'roc_auc': float(auc), 'avg_precision': float(ap), 'blended': float(blended)}
        if blended > best_score:
            best_name = name
            best_score = blended
            best_model = model

    assert best_model is not None
    joblib.dump(best_model, ART / 'demo_supplier_model.joblib')
    with open(ART / 'feature_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'features': FEATURES, 'sector_weights': SECTOR_WEIGHTS, 'selected_model': best_name, 'metrics': results}, f, indent=2, ensure_ascii=False)
    return {'selected_model': best_name, 'metrics': results}


def write_backend_files() -> None:
    (BACKEND / '__init__.py').write_text('', encoding='utf-8')
    (FRONTEND / 'index.html').write_text(textwrap.dedent('''\
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>AI Supplier Recommender</title>
      <link rel="stylesheet" href="/static/styles.css" />
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
      <div class="page">
        <header>
          <div>
            <p class="eyebrow">Innovation & Futur</p>
            <h1>Recommandation intelligente des fournisseurs</h1>
            <p class="subtitle">Modèle hybride : score multicritère inspiré de votre présentation + probabilité ML.</p>
          </div>
        </header>

        <section class="card grid two">
          <div>
            <h2>Besoin d'achat</h2>
            <div class="form-grid">
              <label>Secteur
                <select id="sector">
                  <option value="global">Global</option>
                  <option value="automobile">Automobile</option>
                  <option value="electronique">Électronique / Tech</option>
                </select>
              </label>
              <label>Catégorie produit
                <input id="product_category" value="Composants industriels" />
              </label>
              <label>Modèle demandé
                <input id="required_model" value="Moteur brushless 24V" />
              </label>
              <label>Coût produit cible (unité)
                <input id="product_cost_target" type="number" value="120" />
              </label>
              <label>Budget total
                <input id="budget" type="number" value="150000" />
              </label>
              <label>Prix de vente prévu
                <input id="selling_price" type="number" value="195000" />
              </label>
              <label>Quantité
                <input id="quantity" type="number" value="1000" />
              </label>
            </div>
          </div>
          <div>
            <h2>Poids métier</h2>
            <p class="hint">Les poids changent automatiquement selon le secteur, d'après votre PDF.</p>
            <div id="weightsBox" class="weights-box"></div>
          </div>
        </section>

        <section class="card">
          <div class="section-head">
            <h2>Fournisseurs candidats</h2>
            <button id="addSupplierBtn" class="ghost">+ Ajouter un fournisseur</button>
          </div>
          <div id="suppliersContainer"></div>
          <div class="actions">
            <button id="predictBtn">Lancer la recommandation</button>
          </div>
        </section>

        <section class="grid two">
          <div class="card">
            <h2>Classement final</h2>
            <div id="resultsTable"></div>
          </div>
          <div class="card">
            <h2>Fournisseur recommandé</h2>
            <div id="recommendationBox" class="recommendation-box">Aucun résultat pour l'instant.</div>
          </div>
        </section>

        <section class="grid two charts">
          <div class="card">
            <h2>Comparaison des scores finaux</h2>
            <canvas id="scoreChart"></canvas>
          </div>
          <div class="card">
            <h2>Profil multicritère du meilleur fournisseur</h2>
            <canvas id="radarChart"></canvas>
          </div>
        </section>
      </div>
      <script src="/static/app.js"></script>
    </body>
    </html>
    '''), encoding='utf-8')

    (FRONTEND / 'styles.css').write_text(textwrap.dedent('''\
    :root {
      --bg: #0b1220;
      --card: #121a2b;
      --muted: #96a3b8;
      --text: #f3f7ff;
      --line: #23314d;
      --accent: #5bb4ff;
      --accent-2: #7be2c4;
      --warn: #ffd166;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #09101c 0%, #0d1525 100%);
      color: var(--text);
    }
    .page {
      max-width: 1300px;
      margin: 0 auto;
      padding: 28px;
    }
    header {
      margin-bottom: 20px;
      padding: 24px 0 10px;
    }
    .eyebrow {
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      margin: 0 0 8px;
      font-weight: 700;
    }
    h1, h2, h3 { margin: 0 0 12px; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.15rem; }
    .subtitle, .hint { color: var(--muted); }
    .grid { display: grid; gap: 20px; }
    .grid.two { grid-template-columns: 1.2fr 0.8fr; }
    .charts { align-items: start; }
    .card {
      background: rgba(18, 26, 43, 0.95);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 12px 35px rgba(0, 0, 0, 0.28);
      margin-bottom: 20px;
    }
    .form-grid, .supplier-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 0.93rem;
      color: var(--muted);
    }
    input, select {
      border: 1px solid var(--line);
      background: #0d1424;
      color: var(--text);
      border-radius: 14px;
      padding: 12px 14px;
      outline: none;
    }
    .weights-box {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .weight-chip {
      background: #0d1424;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }
    .weight-chip span { color: var(--muted); display: block; font-size: 0.85rem; }
    .supplier-card {
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #0d1424;
      margin-bottom: 14px;
    }
    .section-head, .supplier-head, .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }
    button {
      border: 0;
      background: linear-gradient(135deg, var(--accent), #7c9cff);
      color: #08111f;
      font-weight: 800;
      padding: 12px 16px;
      border-radius: 14px;
      cursor: pointer;
    }
    button.ghost {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--line);
    }
    .results-table table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }
    .results-table th, .results-table td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }
    .badge {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      background: rgba(123, 226, 196, 0.12);
      color: var(--accent-2);
      border: 1px solid rgba(123, 226, 196, 0.2);
    }
    .recommendation-box {
      min-height: 220px;
      padding: 16px;
      border-radius: 18px;
      background: #0d1424;
      border: 1px solid var(--line);
      color: var(--text);
    }
    .small {
      font-size: 0.88rem;
      color: var(--muted);
    }
    @media (max-width: 980px) {
      .grid.two, .form-grid, .supplier-grid { grid-template-columns: 1fr; }
      .page { padding: 16px; }
    }
    '''), encoding='utf-8')

    (FRONTEND / 'app.js').write_text(textwrap.dedent('''\
    const sectorWeights = {
      global: { quality: 28, cost: 20, delivery: 17, innovation: 13, flexibility: 12, esg: 10 },
      automobile: { quality: 30, cost: 22, delivery: 20, innovation: 8, flexibility: 12, esg: 8 },
      electronique: { quality: 22, cost: 18, delivery: 15, innovation: 25, flexibility: 10, esg: 10 },
    };

    const supplierDefaults = [
      { name: 'Atlas Components', unit_cost: 110, logistics_cost: 5, lead_time_days: 14, quality_score: 90, delivery_score: 88, flexibility_score: 80, innovation_score: 72, esg_score: 68, historical_win_rate: 0.37, historical_volume: 420, defect_rate_ppm: 35, risk_score: 22 },
      { name: 'Nova Industrial', unit_cost: 104, logistics_cost: 8, lead_time_days: 19, quality_score: 83, delivery_score: 79, flexibility_score: 85, innovation_score: 77, esg_score: 74, historical_win_rate: 0.29, historical_volume: 310, defect_rate_ppm: 70, risk_score: 35 },
      { name: 'Sigma Tech Supply', unit_cost: 118, logistics_cost: 3, lead_time_days: 10, quality_score: 94, delivery_score: 92, flexibility_score: 78, innovation_score: 88, esg_score: 80, historical_win_rate: 0.44, historical_volume: 500, defect_rate_ppm: 18, risk_score: 18 },
    ];

    const container = document.getElementById('suppliersContainer');
    const sectorEl = document.getElementById('sector');
    let scoreChart;
    let radarChart;

    function renderWeights() {
      const weights = sectorWeights[sectorEl.value];
      document.getElementById('weightsBox').innerHTML = Object.entries(weights)
        .map(([k, v]) => `<div class="weight-chip"><strong>${v}%</strong><span>${k}</span></div>`)
        .join('');
    }

    function supplierCardTemplate(index, supplier) {
      return `
        <div class="supplier-card" data-index="${index}">
          <div class="supplier-head">
            <h3>Fournisseur ${index + 1}</h3>
            <button class="ghost" onclick="removeSupplier(${index})">Supprimer</button>
          </div>
          <div class="supplier-grid">
            ${inputField('Nom', 'name', supplier.name, 'text')}
            ${inputField('Coût unitaire', 'unit_cost', supplier.unit_cost)}
            ${inputField('Coût logistique', 'logistics_cost', supplier.logistics_cost)}
            ${inputField('Lead time (jours)', 'lead_time_days', supplier.lead_time_days)}
            ${inputField('Score qualité /100', 'quality_score', supplier.quality_score)}
            ${inputField('Score livraison /100', 'delivery_score', supplier.delivery_score)}
            ${inputField('Score flexibilité /100', 'flexibility_score', supplier.flexibility_score)}
            ${inputField('Score innovation /100', 'innovation_score', supplier.innovation_score)}
            ${inputField('Score ESG /100', 'esg_score', supplier.esg_score)}
            ${inputField('Win rate historique', 'historical_win_rate', supplier.historical_win_rate)}
            ${inputField('Volume historique', 'historical_volume', supplier.historical_volume)}
            ${inputField('Défauts PPM', 'defect_rate_ppm', supplier.defect_rate_ppm)}
            ${inputField('Score risque /100', 'risk_score', supplier.risk_score)}
          </div>
        </div>
      `;
    }

    function inputField(label, key, value, type = 'number') {
      return `<label>${label}<input data-key="${key}" type="${type}" value="${value}" /></label>`;
    }

    function renderSuppliers(list) {
      container.innerHTML = list.map((supplier, index) => supplierCardTemplate(index, supplier)).join('');
    }

    window.removeSupplier = (index) => {
      supplierDefaults.splice(index, 1);
      renderSuppliers(supplierDefaults);
    };

    document.getElementById('addSupplierBtn').addEventListener('click', () => {
      supplierDefaults.push({ name: `Nouveau fournisseur ${supplierDefaults.length + 1}`, unit_cost: 100, logistics_cost: 6, lead_time_days: 15, quality_score: 80, delivery_score: 80, flexibility_score: 80, innovation_score: 80, esg_score: 80, historical_win_rate: 0.20, historical_volume: 100, defect_rate_ppm: 100, risk_score: 30 });
      renderSuppliers(supplierDefaults);
    });

    function gatherRequest() {
      const cards = [...document.querySelectorAll('.supplier-card')];
      const suppliers = cards.map(card => {
        const values = {};
        card.querySelectorAll('input').forEach(input => {
          const key = input.dataset.key;
          values[key] = key === 'name' ? input.value : Number(input.value);
        });
        return values;
      });

      return {
        sector: document.getElementById('sector').value,
        product_category: document.getElementById('product_category').value,
        required_model: document.getElementById('required_model').value,
        product_cost_target: Number(document.getElementById('product_cost_target').value),
        budget: Number(document.getElementById('budget').value),
        selling_price: Number(document.getElementById('selling_price').value),
        quantity: Number(document.getElementById('quantity').value),
        suppliers,
      };
    }

    function renderResults(data) {
      const rows = data.ranking.map((row, idx) => `
        <tr>
          <td>${idx === 0 ? '<span class="badge">Choisi</span>' : ''}</td>
          <td>${row.name}</td>
          <td>${(row.final_score * 100).toFixed(1)}%</td>
          <td>${(row.ml_probability * 100).toFixed(1)}%</td>
          <td>${(row.business_score * 100).toFixed(1)}%</td>
          <td>${row.estimated_tco.toFixed(2)}</td>
        </tr>
      `).join('');
      document.getElementById('resultsTable').innerHTML = `<div class="results-table"><table><thead><tr><th></th><th>Fournisseur</th><th>Score final</th><th>ML</th><th>Métier</th><th>TCO estimé</th></tr></thead><tbody>${rows}</tbody></table></div>`;

      const best = data.ranking[0];
      document.getElementById('recommendationBox').innerHTML = `
        <h3>${best.name}</h3>
        <p><strong>Pourquoi ce choix :</strong> ${best.explanation.summary}</p>
        <p class="small"><strong>Forces :</strong> ${best.explanation.strengths.join(' · ')}</p>
        <p class="small"><strong>Points à surveiller :</strong> ${best.explanation.watchouts.join(' · ')}</p>
        <p class="small"><strong>Poids secteur :</strong> ${Object.entries(data.weights_used).map(([k,v]) => `${k} ${Math.round(v*100)}%`).join(' | ')}</p>
      `;

      if (scoreChart) scoreChart.destroy();
      scoreChart = new Chart(document.getElementById('scoreChart'), {
        type: 'bar',
        data: {
          labels: data.ranking.map(r => r.name),
          datasets: [{ label: 'Score final', data: data.ranking.map(r => Number((r.final_score * 100).toFixed(2))) }]
        },
        options: { responsive: true, plugins: { legend: { labels: { color: '#f3f7ff' } } }, scales: { x: { ticks: { color: '#cdd7e6' } }, y: { ticks: { color: '#cdd7e6' } } } }
      });

      if (radarChart) radarChart.destroy();
      radarChart = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
          labels: ['Qualité', 'Coût', 'Livraison', 'Flexibilité', 'Innovation', 'ESG'],
          datasets: [{ label: best.name, data: [best.criteria_breakdown.quality, best.criteria_breakdown.cost, best.criteria_breakdown.delivery, best.criteria_breakdown.flexibility, best.criteria_breakdown.innovation, best.criteria_breakdown.esg].map(v => Number((v * 100).toFixed(1))) }]
        },
        options: { responsive: true, plugins: { legend: { labels: { color: '#f3f7ff' } } }, scales: { r: { angleLines: { color: '#3b4b68' }, grid: { color: '#3b4b68' }, pointLabels: { color: '#dbe6f3' }, ticks: { color: '#dbe6f3', backdropColor: 'transparent' } } } }
      });
    }

    document.getElementById('predictBtn').addEventListener('click', async () => {
      const payload = gatherRequest();
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      renderResults(data);
    });

    sectorEl.addEventListener('change', renderWeights);
    renderWeights();
    renderSuppliers(supplierDefaults);
    '''), encoding='utf-8')

    (BACKEND / 'requirements.txt').write_text(textwrap.dedent('''\
    fastapi==0.115.0
    uvicorn==0.30.6
    pandas==2.2.2
    numpy==2.1.1
    scikit-learn==1.5.1
    joblib==1.4.2
    python-multipart==0.0.9
    '''), encoding='utf-8')

    (BACKEND / 'main.py').write_text(textwrap.dedent('''\
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Dict, List, Literal

    import joblib
    import numpy as np
    import pandas as pd
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    BASE_DIR = Path(__file__).resolve().parent
    FRONTEND_DIR = BASE_DIR.parent / 'frontend'
    ART_DIR = BASE_DIR / 'artifacts'
    MODEL = joblib.load(ART_DIR / 'demo_supplier_model.joblib')
    MANIFEST = json.loads((ART_DIR / 'feature_manifest.json').read_text(encoding='utf-8'))

    SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = MANIFEST['sector_weights']
    FEATURES: List[str] = MANIFEST['features']

    app = FastAPI(title='AI Supplier Recommender', version='1.0.0')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

    class SupplierInput(BaseModel):
        name: str
        unit_cost: float = Field(..., ge=0)
        logistics_cost: float = Field(0, ge=0)
        lead_time_days: float = Field(..., ge=0)
        quality_score: float = Field(..., ge=0, le=100)
        delivery_score: float = Field(..., ge=0, le=100)
        flexibility_score: float = Field(..., ge=0, le=100)
        innovation_score: float = Field(..., ge=0, le=100)
        esg_score: float = Field(..., ge=0, le=100)
        historical_win_rate: float = Field(0.2, ge=0, le=1)
        historical_volume: float = Field(0, ge=0)
        defect_rate_ppm: float = Field(100, ge=0)
        risk_score: float = Field(30, ge=0, le=100)

    class RecommendationRequest(BaseModel):
        sector: Literal['global', 'automobile', 'electronique'] = 'global'
        product_category: str
        required_model: str
        product_cost_target: float = Field(..., ge=0)
        budget: float = Field(..., gt=0)
        selling_price: float = Field(..., gt=0)
        quantity: int = Field(..., gt=0)
        suppliers: List[SupplierInput]

    def clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def sector_weights(sector: str) -> Dict[str, float]:
        return SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS['global'])

    def engineer_features(request: RecommendationRequest, supplier: SupplierInput) -> Dict[str, float]:
        total_cost_unit = supplier.unit_cost + supplier.logistics_cost
        budget_per_unit = request.budget / request.quantity
        selling_per_unit = request.selling_price / request.quantity
        budget_fit = clip01(1 - max(total_cost_unit - budget_per_unit, 0) / max(budget_per_unit, 1e-6))
        margin_rate = clip01((selling_per_unit - total_cost_unit) / max(selling_per_unit, 1e-6))
        cost_advantage = clip01(1 - total_cost_unit / max(budget_per_unit * 1.25, 1e-6))
        quality_proxy = clip01(0.7 * supplier.quality_score / 100 + 0.3 * (1 - min(supplier.defect_rate_ppm, 4000) / 4000))
        delivery_proxy = clip01(0.7 * supplier.delivery_score / 100 + 0.3 * (1 - min(supplier.lead_time_days, 60) / 60))
        innovation_proxy = clip01(supplier.innovation_score / 100)
        esg_proxy = clip01(supplier.esg_score / 100)
        return {
            'unit_cost': supplier.unit_cost,
            'logistics_cost': supplier.logistics_cost,
            'lead_time_days': supplier.lead_time_days,
            'quality_score': supplier.quality_score,
            'delivery_score': supplier.delivery_score,
            'flexibility_score': supplier.flexibility_score,
            'innovation_score': supplier.innovation_score,
            'esg_score': supplier.esg_score,
            'historical_win_rate': supplier.historical_win_rate,
            'historical_volume': supplier.historical_volume,
            'defect_rate_ppm': supplier.defect_rate_ppm,
            'risk_score': supplier.risk_score,
            'budget_fit': budget_fit,
            'margin_rate': margin_rate,
            'cost_advantage': cost_advantage,
            'quality_proxy': quality_proxy,
            'delivery_proxy': delivery_proxy,
            'innovation_proxy': innovation_proxy,
            'esg_proxy': esg_proxy,
        }

    def business_score(request: RecommendationRequest, supplier: SupplierInput, features: Dict[str, float]) -> Dict[str, float]:
        budget_per_unit = request.budget / request.quantity
        total_cost_unit = supplier.unit_cost + supplier.logistics_cost
        cost_score = clip01(0.55 * features['budget_fit'] + 0.25 * features['margin_rate'] + 0.20 * (1 - min(total_cost_unit, budget_per_unit * 1.5) / max(budget_per_unit * 1.5, 1e-6)))
        criteria = {
            'quality': features['quality_proxy'],
            'cost': cost_score,
            'delivery': features['delivery_proxy'],
            'flexibility': clip01(supplier.flexibility_score / 100),
            'innovation': features['innovation_proxy'],
            'esg': features['esg_proxy'],
        }
        weights = sector_weights(request.sector)
        score = sum(criteria[k] * weights[k] for k in criteria)
        return {'score': float(score), 'criteria': criteria, 'weights': weights}

    def explain_supplier(row: Dict[str, float], supplier: SupplierInput) -> Dict[str, object]:
        strengths = []
        watchouts = []
        if row['criteria_breakdown']['quality'] >= 0.8:
            strengths.append('très bon niveau qualité')
        if row['criteria_breakdown']['delivery'] >= 0.8:
            strengths.append('livraison fiable et rapide')
        if row['criteria_breakdown']['innovation'] >= 0.8:
            strengths.append('fort potentiel d’innovation')
        if row['criteria_breakdown']['esg'] >= 0.75:
            strengths.append('profil ESG solide')
        if row['estimated_tco'] <= row['budget_per_unit']:
            strengths.append('coût total compatible avec le budget')
        if supplier.risk_score > 55:
            watchouts.append('risque fournisseur à surveiller')
        if supplier.defect_rate_ppm > 250:
            watchouts.append('défauts potentiellement élevés')
        if supplier.lead_time_days > 25:
            watchouts.append('lead time plus long que les meilleurs candidats')
        if row['criteria_breakdown']['esg'] < 0.55:
            watchouts.append('performance ESG moyenne')
        if not strengths:
            strengths.append('profil global équilibré')
        if not watchouts:
            watchouts.append('aucun point critique majeur détecté')
        summary = f"{supplier.name} combine un bon compromis entre score IA ({row['ml_probability']:.1%}), score métier ({row['business_score']:.1%}) et coût total estimé ({row['estimated_tco']:.2f} par unité)."
        return {'summary': summary, 'strengths': strengths[:3], 'watchouts': watchouts[:3]}

    @app.get('/')
    def root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / 'index.html')

    @app.get('/health')
    def health() -> Dict[str, object]:
        return {'status': 'ok', 'selected_model': MANIFEST['selected_model'], 'features': FEATURES}

    @app.post('/predict')
    def predict(request: RecommendationRequest) -> Dict[str, object]:
        rows = []
        for supplier in request.suppliers:
            feats = engineer_features(request, supplier)
            X = pd.DataFrame([{k: feats[k] for k in FEATURES}])
            ml_probability = float(MODEL.predict_proba(X)[0, 1])
            business = business_score(request, supplier, feats)
            estimated_tco = float(supplier.unit_cost + supplier.logistics_cost + 0.00005 * supplier.defect_rate_ppm + 0.03 * supplier.risk_score)
            final_score = 0.65 * ml_probability + 0.35 * business['score']
            row = {
                'name': supplier.name,
                'ml_probability': ml_probability,
                'business_score': float(business['score']),
                'final_score': float(final_score),
                'estimated_tco': estimated_tco,
                'criteria_breakdown': business['criteria'],
                'budget_per_unit': request.budget / request.quantity,
            }
            row['explanation'] = explain_supplier(row, supplier)
            rows.append(row)
        rows.sort(key=lambda x: x['final_score'], reverse=True)
        return {
            'request_summary': {
                'sector': request.sector,
                'required_model': request.required_model,
                'product_category': request.product_category,
            },
            'weights_used': sector_weights(request.sector),
            'ranking': rows,
        }
    '''), encoding='utf-8')

    (ROOT / 'README.md').write_text(textwrap.dedent('''\
    # Package IA - Sélection intelligente des fournisseurs

    Ce package contient trois livrables :

    1. `notebooks/supplier_selection_workflow.ipynb`
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

    Le modèle fourni dans `app/backend/artifacts/demo_supplier_model.joblib` est un **modèle de démonstration** entraîné sur des données synthétiques réalistes pour rendre l'application directement utilisable.

    Pour le projet final académique, il faut **remplacer ce modèle** par celui exporté par le notebook après exécution sur les datasets publics massifs + vos KPI fournisseurs internes.
    '''), encoding='utf-8')

    (DOCS / 'datasets_links.md').write_text(textwrap.dedent('''\
    # Datasets recommandés pour le projet fournisseur IA

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
    '''), encoding='utf-8')


def write_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    add = cells.append

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    # Sélection intelligente des fournisseurs

    Ce notebook construit un pipeline **réaliste et soutenable académiquement** pour recommander le meilleur fournisseur à partir :

    1. d'un historique public massif d'attributions de marchés,
    2. de KPI métier inspirés de votre présentation (qualité, coût, livraison, flexibilité, innovation, ESG),
    3. d'une couche d'explicabilité pour l'application web.

    ## Idée clé

    Il n'existe généralement pas de dataset public parfait qui contient directement **"le meilleur fournisseur"** avec toutes les variables OTIF / PPM / ESG / flexibilité.

    La stratégie correcte est donc :

    - utiliser les **datasets publics massifs** pour apprendre des schémas d'attribution,
    - fusionner ensuite les **KPI internes fournisseurs**,
    - entraîner un modèle de classement / classification,
    - produire une **recommandation explicable**.
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Datasets recommandés

    - TED CSV subset (UE) — source officielle
    - GPPD — 72+ millions de contrats harmonisés
    - FPDS / Figshare — quasi 100 millions d'actions contractuelles US

    Ce notebook est prêt pour un mode **hybride** :

    - si vous avez déjà les fichiers bruts, il les charge,
    - sinon il bascule sur une **démo synthétique** pour valider le pipeline de bout en bout.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    import json
    import math
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', 120)

    PROJECT_ROOT = Path.cwd().resolve().parent if (Path.cwd().name == 'notebooks') else Path.cwd().resolve()
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DIR = DATA_DIR / 'raw'
    PROCESSED_DIR = DATA_DIR / 'processed'
    ARTIFACT_DIR = PROJECT_ROOT / 'app' / 'backend' / 'artifacts'

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    SECTOR_WEIGHTS = {
        'global': {'quality': 0.28, 'cost': 0.20, 'delivery': 0.17, 'innovation': 0.13, 'flexibility': 0.12, 'esg': 0.10},
        'automobile': {'quality': 0.30, 'cost': 0.22, 'delivery': 0.20, 'innovation': 0.08, 'flexibility': 0.12, 'esg': 0.08},
        'electronique': {'quality': 0.22, 'cost': 0.18, 'delivery': 0.15, 'innovation': 0.25, 'flexibility': 0.10, 'esg': 0.10},
    }

    FEATURE_COLUMNS = [
        'unit_cost', 'logistics_cost', 'lead_time_days', 'quality_score', 'delivery_score',
        'flexibility_score', 'innovation_score', 'esg_score', 'historical_win_rate',
        'historical_volume', 'defect_rate_ppm', 'risk_score', 'budget_fit', 'margin_rate',
        'cost_advantage', 'quality_proxy', 'delivery_proxy', 'innovation_proxy', 'esg_proxy'
    ]

    def clip01(x):
        return np.clip(x, 0, 1)
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 1 — Déclaration des fichiers attendus

    Placez idéalement vos fichiers ici :

    - `data/raw/ted/can_2018.csv`, `can_2019.csv`, ...
    - `data/raw/ted/cn_2018.csv`, `cn_2019.csv`, ...
    - `data/raw/internal/supplier_kpis.csv`

    `supplier_kpis.csv` devrait contenir au moins :

    - supplier_name
    - quality_score
    - delivery_score
    - flexibility_score
    - innovation_score
    - esg_score
    - defect_rate_ppm
    - lead_time_days
    - risk_score
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    TED_DIR = RAW_DIR / 'ted'
    INTERNAL_DIR = RAW_DIR / 'internal'
    ted_can_files = sorted(TED_DIR.glob('can_*.csv'))
    ted_cn_files = sorted(TED_DIR.glob('cn_*.csv'))
    internal_kpi_file = INTERNAL_DIR / 'supplier_kpis.csv'

    ted_can_files[:3], ted_cn_files[:3], internal_kpi_file.exists()
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 2 — Fonctions utilitaires
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    def create_synthetic_procurement_dataset(n_tenders=8000, suppliers_per_tender=4, seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        sectors = rng.choice(['global', 'automobile', 'electronique'], size=n_tenders, p=[0.45, 0.25, 0.30])
        for tender_id in range(n_tenders):
            sector = sectors[tender_id]
            weights = SECTOR_WEIGHTS[sector]
            quantity = int(rng.integers(100, 5000))
            budget_per_unit = float(rng.uniform(25, 350))
            selling_per_unit = float(budget_per_unit * rng.uniform(1.05, 1.7))
            awarded_pos = int(rng.integers(0, suppliers_per_tender))
            for pos in range(suppliers_per_tender):
                supplier_name = f'Supplier_{int(rng.integers(1, 800))}'
                quality_score = float(rng.uniform(40, 99))
                delivery_score = float(rng.uniform(35, 99))
                flexibility_score = float(rng.uniform(30, 99))
                innovation_score = float(rng.uniform(20, 99))
                esg_score = float(rng.uniform(20, 99))
                defect_rate_ppm = float(rng.uniform(5, 3500))
                lead_time_days = float(rng.integers(3, 55))
                risk_score = float(rng.uniform(2, 95))
                historical_win_rate = float(rng.uniform(0.01, 0.85))
                historical_volume = int(rng.integers(1, 5000))
                unit_cost = float(rng.uniform(20, 320))
                logistics_cost = float(rng.uniform(1, 35))
                total_cost_unit = unit_cost + logistics_cost
                budget_fit = float(clip01(1 - max(total_cost_unit - budget_per_unit, 0) / max(budget_per_unit, 1)))
                margin_rate = float(clip01((selling_per_unit - total_cost_unit) / max(selling_per_unit, 1)))
                cost_advantage = float(clip01(1 - total_cost_unit / max(budget_per_unit * 1.25, 1)))
                quality_proxy = float(clip01(0.7 * quality_score/100 + 0.3 * (1 - defect_rate_ppm/4000)))
                delivery_proxy = float(clip01(0.7 * delivery_score/100 + 0.3 * (1 - lead_time_days/60)))
                innovation_proxy = float(clip01(innovation_score / 100))
                esg_proxy = float(clip01(esg_score / 100))

                utility = (
                    weights['quality'] * quality_proxy +
                    weights['cost'] * (0.55 * budget_fit + 0.45 * margin_rate) +
                    weights['delivery'] * delivery_proxy +
                    weights['flexibility'] * (flexibility_score/100) +
                    weights['innovation'] * innovation_proxy +
                    weights['esg'] * esg_proxy +
                    0.08 * historical_win_rate +
                    0.02 * np.log1p(historical_volume)/np.log(5001) -
                    0.10 * (risk_score/100)
                ) + rng.normal(0, 0.05)

                rows.append({
                    'tender_id': tender_id,
                    'sector': sector,
                    'supplier_name': supplier_name,
                    'quantity': quantity,
                    'budget_per_unit': budget_per_unit,
                    'selling_per_unit': selling_per_unit,
                    'unit_cost': unit_cost,
                    'logistics_cost': logistics_cost,
                    'lead_time_days': lead_time_days,
                    'quality_score': quality_score,
                    'delivery_score': delivery_score,
                    'flexibility_score': flexibility_score,
                    'innovation_score': innovation_score,
                    'esg_score': esg_score,
                    'historical_win_rate': historical_win_rate,
                    'historical_volume': historical_volume,
                    'defect_rate_ppm': defect_rate_ppm,
                    'risk_score': risk_score,
                    'budget_fit': budget_fit,
                    'margin_rate': margin_rate,
                    'cost_advantage': cost_advantage,
                    'quality_proxy': quality_proxy,
                    'delivery_proxy': delivery_proxy,
                    'innovation_proxy': innovation_proxy,
                    'esg_proxy': esg_proxy,
                    'utility_hidden': utility,
                })

        df = pd.DataFrame(rows)
        best_idx = df.groupby('tender_id')['utility_hidden'].idxmax()
        df['label'] = 0
        df.loc[best_idx, 'label'] = 1
        return df


    def standardize_procurement_columns(df):
        mapping = {
            'supplier': 'supplier_name',
            'vendor_name': 'supplier_name',
            'contract_value': 'award_value',
            'award_amount': 'award_value',
            'cpv': 'product_code',
            'cpv_code': 'product_code',
            'buyer': 'buyer_name',
            'buyer_name': 'buyer_name',
            'notice_id': 'notice_id',
        }
        rename = {c: mapping[c] for c in df.columns if c in mapping}
        return df.rename(columns=rename)


    def build_dataset_from_real_files():
        # Squelette volontairement robuste: à adapter aux colonnes réelles des fichiers TED/GPPD.
        frames = []
        for file in ted_can_files:
            part = pd.read_csv(file, low_memory=False)
            part = standardize_procurement_columns(part)
            part['source_file'] = file.name
            frames.append(part)
        if not frames:
            return None
        can_df = pd.concat(frames, ignore_index=True)
        can_df = can_df.drop_duplicates()
        return can_df
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 3 — Chargement des données

    Si aucun fichier brut n'est disponible, on génère une démo pour que le notebook reste exécutable.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    real_df = build_dataset_from_real_files()

    if real_df is None:
        print('Aucun fichier brut détecté -> génération d\'un jeu synthétique de démonstration.')
        df = create_synthetic_procurement_dataset()
        mode = 'synthetic-demo'
    else:
        print('Fichiers détectés -> pipeline réel à poursuivre et adapter selon les colonnes source.')
        df = real_df.copy()
        mode = 'real-data'

    print(mode)
    df.head()
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 4 — Fusion avec KPI internes fournisseurs

    Sur les données réelles, cette étape est essentielle pour injecter les variables de votre présentation :

    - qualité,
    - coût / TCO,
    - livraison,
    - flexibilité,
    - innovation,
    - ESG.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    if mode == 'real-data' and internal_kpi_file.exists():
        supplier_kpis = pd.read_csv(internal_kpi_file)
        supplier_kpis = supplier_kpis.rename(columns={'supplier': 'supplier_name', 'vendor_name': 'supplier_name'})
        df = df.merge(supplier_kpis, on='supplier_name', how='left')
    elif mode == 'synthetic-demo':
        # Les KPI sont déjà inclus dans la démo.
        pass

    df.shape
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 5 — Nettoyage
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    if mode == 'synthetic-demo':
        cleaned = df.copy()
    else:
        cleaned = df.copy()
        # Exemple de règles minimales à ajuster selon les vraies colonnes.
        if 'award_value' in cleaned.columns:
            cleaned = cleaned[(cleaned['award_value'].isna()) | ((cleaned['award_value'] >= 100) & (cleaned['award_value'] <= 10_000_000_000))]
        cleaned = cleaned.drop_duplicates()

    cleaned.head()
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 6 — Construction de la table d'apprentissage

    Cas réel recommandé :

    - créer des paires `(appel d'offres, fournisseur candidat)`,
    - label = 1 si le fournisseur a gagné, 0 sinon,
    - générer des fournisseurs négatifs par échantillonnage dans la même catégorie / pays / période.

    Ici, la démo contient déjà ces paires.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    if mode == 'synthetic-demo':
        model_df = cleaned.copy()
    else:
        model_df = cleaned.copy()
        # TODO réel: construire ici les couples tender-supplier et le label binaire.

    model_df = model_df.dropna(subset=[c for c in FEATURE_COLUMNS if c in model_df.columns] + (['label'] if 'label' in model_df.columns else []))
    model_df.shape, model_df['label'].mean() if 'label' in model_df.columns else None
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 7 — Split temporel ou holdout

    Sur les vraies données, préférez un split **par date**.
    Sur la démo, on fait un holdout classique.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    X = model_df[FEATURE_COLUMNS].copy()
    y = model_df['label'].astype(int).copy()
    groups = model_df['tender_id'].copy() if 'tender_id' in model_df.columns else pd.Series(range(len(model_df)))

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=y
    )

    X_train.shape, X_test.shape
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 8 — Baselines et modèles candidats
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    def top_k_hit_rate(y_true, group_ids, scores, k=1):
        temp = pd.DataFrame({'y': y_true, 'group': group_ids, 'score': scores})
        hits = []
        for _, grp in temp.groupby('group'):
            top = grp.sort_values('score', ascending=False).head(k)
            hits.append(int((top['y'] == 1).any()))
        return float(np.mean(hits)) if hits else 0.0


    candidates = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=260,
            random_state=42,
        )
    }

    metrics = {}
    fitted_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            'roc_auc': roc_auc_score(y_test, proba),
            'avg_precision': average_precision_score(y_test, proba),
            'top1_hit': top_k_hit_rate(y_test.values, g_test.values, proba, k=1),
            'top3_hit': top_k_hit_rate(y_test.values, g_test.values, proba, k=3),
        }
        fitted_models[name] = model

    metrics_df = pd.DataFrame(metrics).T.sort_values(['top1_hit', 'roc_auc', 'avg_precision'], ascending=False)
    metrics_df
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 9 — Choix du modèle retenu

    Critère recommandé :

    - priorité au **Top-1 hit rate**,
    - puis ROC-AUC,
    - puis Average Precision.

    Cela est cohérent avec le besoin métier : recommander **le bon fournisseur en premier**.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    best_model_name = metrics_df.index[0]
    best_model = fitted_models[best_model_name]

    print('Modèle retenu :', best_model_name)
    print(metrics_df.loc[best_model_name])
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 10 — Export des artefacts pour l'application web
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    joblib.dump(best_model, ARTIFACT_DIR / 'demo_supplier_model.joblib')

    manifest = {
        'selected_model': best_model_name,
        'features': FEATURE_COLUMNS,
        'sector_weights': SECTOR_WEIGHTS,
        'metrics': metrics_df.to_dict(orient='index')
    }

    with open(ARTIFACT_DIR / 'feature_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print('Artefacts exportés vers:', ARTIFACT_DIR)
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Étape 11 — Démonstration de scoring fournisseur

    Ce bloc illustre la logique utilisée ensuite par le backend pour classer plusieurs fournisseurs candidats.
    ''')))

    add(nbf.v4.new_code_cell(textwrap.dedent('''
    sample_candidates = X_test.head(5).copy()
    sample_scores = best_model.predict_proba(sample_candidates)[:, 1]
    preview = sample_candidates.copy()
    preview['prediction_probability'] = sample_scores
    preview.sort_values('prediction_probability', ascending=False).head()
    ''')))

    add(nbf.v4.new_markdown_cell(textwrap.dedent('''
    ## Conclusion

    Ce notebook fournit une base cohérente avec votre soutenance :

    - **multicritère**,
    - **hybride IA + règles métier**,
    - **explicable**,
    - **prêt à être connecté à une application web**.

    Pour la version finale académique, exécutez ce notebook sur les vrais fichiers TED / GPPD / FPDS + vos KPI internes.
    ''')))

    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'}
    }

    out = NOTEBOOKS / 'supplier_selection_workflow.ipynb'
    with out.open('w', encoding='utf-8') as f:
        nbf.write(nb, f)


def write_start_script() -> None:
    (ROOT / 'start_backend.sh').write_text(textwrap.dedent('''\
    #!/usr/bin/env bash
    set -euo pipefail
    cd "$(dirname "$0")/app/backend"
    python -m uvicorn main:app --reload
    '''), encoding='utf-8')


def main() -> None:
    metrics = train_demo_model()
    write_backend_files()
    write_notebook()
    write_start_script()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
