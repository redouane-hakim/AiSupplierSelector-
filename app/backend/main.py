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
