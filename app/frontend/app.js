const sectorWeights = {
  global: { quality: 28, cost: 20, delivery: 17, innovation: 13, flexibility: 12, esg: 10 },
  automobile: { quality: 30, cost: 22, delivery: 20, innovation: 8, flexibility: 12, esg: 8 },
  electronique: { quality: 22, cost: 18, delivery: 15, innovation: 25, flexibility: 10, esg: 10 },
};

const sectorSupplierDefaults = {
  global: [
    { name: 'Atlas Components', unit_cost: 110, logistics_cost: 5, lead_time_days: 14, quality_score: 90, delivery_score: 88, flexibility_score: 80, innovation_score: 72, esg_score: 68, historical_win_rate: 0.37, historical_volume: 420, defect_rate_ppm: 35, risk_score: 22 },
    { name: 'Nova Industrial', unit_cost: 104, logistics_cost: 8, lead_time_days: 19, quality_score: 83, delivery_score: 79, flexibility_score: 85, innovation_score: 77, esg_score: 74, historical_win_rate: 0.29, historical_volume: 310, defect_rate_ppm: 70, risk_score: 35 },
    { name: 'Sigma Tech Supply', unit_cost: 118, logistics_cost: 3, lead_time_days: 10, quality_score: 94, delivery_score: 92, flexibility_score: 78, innovation_score: 88, esg_score: 80, historical_win_rate: 0.44, historical_volume: 500, defect_rate_ppm: 18, risk_score: 18 },
  ],
  automobile: [
    { name: 'AutoParts Pro', unit_cost: 115, logistics_cost: 6, lead_time_days: 12, quality_score: 92, delivery_score: 90, flexibility_score: 82, innovation_score: 70, esg_score: 75, historical_win_rate: 0.40, historical_volume: 450, defect_rate_ppm: 25, risk_score: 20 },
    { name: 'CarTech Solutions', unit_cost: 108, logistics_cost: 7, lead_time_days: 16, quality_score: 85, delivery_score: 82, flexibility_score: 88, innovation_score: 75, esg_score: 78, historical_win_rate: 0.32, historical_volume: 380, defect_rate_ppm: 50, risk_score: 28 },
    { name: 'Vehicle Supply Co', unit_cost: 122, logistics_cost: 4, lead_time_days: 8, quality_score: 96, delivery_score: 94, flexibility_score: 76, innovation_score: 85, esg_score: 82, historical_win_rate: 0.48, historical_volume: 520, defect_rate_ppm: 15, risk_score: 16 },
  ],
  electronique: [
    { name: 'ElectroComp', unit_cost: 105, logistics_cost: 5, lead_time_days: 18, quality_score: 87, delivery_score: 85, flexibility_score: 83, innovation_score: 90, esg_score: 80, historical_win_rate: 0.35, historical_volume: 400, defect_rate_ppm: 40, risk_score: 25 },
    { name: 'TechInnovate', unit_cost: 98, logistics_cost: 9, lead_time_days: 22, quality_score: 80, delivery_score: 78, flexibility_score: 90, innovation_score: 95, esg_score: 85, historical_win_rate: 0.28, historical_volume: 350, defect_rate_ppm: 80, risk_score: 38 },
    { name: 'Digital Supply', unit_cost: 125, logistics_cost: 3, lead_time_days: 12, quality_score: 98, delivery_score: 96, flexibility_score: 75, innovation_score: 92, esg_score: 88, historical_win_rate: 0.50, historical_volume: 550, defect_rate_ppm: 12, risk_score: 14 },
  ],
};

let supplierDefaults = sectorSupplierDefaults.global.slice();

const container = document.getElementById('suppliersContainer');
const sectorEl = document.getElementById('sector');
const inputSection = document.getElementById('inputSection');
const resultsSection = document.getElementById('resultsSection');
const newPredictionBtn = document.getElementById('newPredictionBtn');
let scoreChart;
let radarChart;
let activeSupplierIndex = 0;

let needSectionOpen = true;

function renderWeights() {
  const weights = sectorWeights[sectorEl.value];
  document.getElementById('weightsBox').innerHTML = Object.entries(weights)
    .map(([k, v]) => `<div class="weight-chip"><strong>${v}%</strong><span>${k}</span></div>`)
    .join('');
}

function toggleNeedSection() {
  needSectionOpen = !needSectionOpen;
  const needDetails = document.getElementById('needDetails');
  const needArrow = document.getElementById('needArrow');
  if (needSectionOpen) {
    needDetails.classList.remove('closed');
    needArrow.classList.remove('collapsed');
  } else {
    needDetails.classList.add('closed');
    needArrow.classList.add('collapsed');
  }
}

function supplierCardTemplate(index, supplier, expanded = false) {
  const chevron = expanded ? '▼' : '▶';
  return `
    <div class="supplier-card${expanded ? ' open' : ''}" data-index="${index}">
      <div class="supplier-head">
        <button class="toggle-btn" onclick="toggleSupplier(${index})">${chevron}</button>
        <div class="supplier-summary">
          <h3>${supplier.name || `Fournisseur ${index + 1}`}</h3>
          <p>${supplier.unit_cost}€/unité · Qualité ${supplier.quality_score}/100 · Livraison ${supplier.delivery_score}/100</p>
        </div>
        <button class="ghost" onclick="removeSupplier(${index})">Supprimer</button>
      </div>
      <div class="supplier-details${expanded ? ' open' : ''}">
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
    </div>
  `;
}

function inputField(label, key, value, type = 'number') {
  return `<label>${label}<input data-key="${key}" type="${type}" value="${value}" /></label>`;
}

function toggleSupplier(index) {
  activeSupplierIndex = activeSupplierIndex === index ? -1 : index;
  renderSuppliers(supplierDefaults);
}

function renderSuppliers(list) {
  container.innerHTML = list.map((supplier, index) => supplierCardTemplate(index, supplier, index === activeSupplierIndex)).join('');
  // Add event listeners to update supplierDefaults on input change
  list.forEach((supplier, index) => {
    const card = container.children[index];
    card.querySelectorAll('input').forEach(input => {
      input.addEventListener('input', () => {
        const key = input.dataset.key;
        supplierDefaults[index][key] = key === 'name' ? input.value : Number(input.value);
        // Update summary if name changed
        if (key === 'name') {
          const summaryH3 = card.querySelector('.supplier-summary h3');
          if (summaryH3) summaryH3.textContent = input.value || `Fournisseur ${index + 1}`;
        }
      });
    });
  });
}

window.toggleSupplier = toggleSupplier;

window.removeSupplier = (index) => {
  supplierDefaults.splice(index, 1);
  if (activeSupplierIndex >= supplierDefaults.length) activeSupplierIndex = supplierDefaults.length - 1;
  renderSuppliers(supplierDefaults);
};

document.getElementById('addSupplierBtn').addEventListener('click', () => {
  const currentSector = sectorEl.value;
  const defaultSupplier = sectorSupplierDefaults[currentSector][0]; // Use first as template
  supplierDefaults.push({ 
    name: `Nouveau fournisseur ${supplierDefaults.length + 1}`, 
    unit_cost: defaultSupplier.unit_cost, 
    logistics_cost: defaultSupplier.logistics_cost, 
    lead_time_days: defaultSupplier.lead_time_days, 
    quality_score: defaultSupplier.quality_score, 
    delivery_score: defaultSupplier.delivery_score, 
    flexibility_score: defaultSupplier.flexibility_score, 
    innovation_score: defaultSupplier.innovation_score, 
    esg_score: defaultSupplier.esg_score, 
    historical_win_rate: defaultSupplier.historical_win_rate, 
    historical_volume: defaultSupplier.historical_volume, 
    defect_rate_ppm: defaultSupplier.defect_rate_ppm, 
    risk_score: defaultSupplier.risk_score 
  });
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
    <div class="recommendation-body">
      <h3>${best.name}</h3>
      <p class="large"><strong>Choix recommandé</strong></p>
      <p><strong>Pourquoi :</strong> ${best.explanation.summary}</p>
      <div class="recommendation-grid">
        <div><strong>Forces</strong><br>${best.explanation.strengths.join(' · ')}</div>
        <div><strong>Watchouts</strong><br>${best.explanation.watchouts.join(' · ')}</div>
      </div>
      <div class="recommendation-metrics">
        <span>Score final : <strong>${(best.final_score * 100).toFixed(1)}%</strong></span>
        <span>ML : <strong>${(best.ml_probability * 100).toFixed(1)}%</strong></span>
        <span>Métier : <strong>${(best.business_score * 100).toFixed(1)}%</strong></span>
        <span>TCO : <strong>${best.estimated_tco.toFixed(2)} €</strong></span>
      </div>
      <p class="small"><strong>Poids secteur :</strong> ${Object.entries(data.weights_used).map(([k,v]) => `${k} ${Math.round(v*100)}%`).join(' | ')}</p>
    </div>
  `;

  if (scoreChart) scoreChart.destroy();
  scoreChart = new Chart(document.getElementById('scoreChart'), {
    type: 'bar',
    data: {
      labels: data.ranking.map(r => r.name),
      datasets: [{ label: 'Score final', data: data.ranking.map(r => Number((r.final_score * 100).toFixed(2))), backgroundColor: 'rgba(49, 130, 206, 0.6)', borderColor: 'rgba(49, 130, 206, 1)', borderWidth: 1 }]
    },
    options: { responsive: true, plugins: { legend: { labels: { color: '#1f2937' } } }, scales: { x: { ticks: { color: '#1f2937' } }, y: { ticks: { color: '#1f2937' } } } }
  });

  if (radarChart) radarChart.destroy();
  const radarLabels = ['Qualité', 'Coût', 'Livraison', 'Flexibilité', 'Innovation', 'ESG'];
  const radarDatasets = data.ranking.map((row, i) => {
    const alpha = 0.15 + Math.min(0.45, i * 0.1);
    const baseColor = ['rgba(49, 130, 206,', 'rgba(237, 137, 54,', 'rgba(72, 187, 120,', 'rgba(168, 85, 247,', 'rgba(136, 72, 192,', 'rgba(239, 68, 68,'][i % 6];
    return {
      label: row.name,
      data: [row.criteria_breakdown.quality, row.criteria_breakdown.cost, row.criteria_breakdown.delivery, row.criteria_breakdown.flexibility, row.criteria_breakdown.innovation, row.criteria_breakdown.esg].map(v => Number((v * 100).toFixed(1))),
      backgroundColor: `${baseColor} ${alpha})`,
      borderColor: `${baseColor} 1)`,
      borderWidth: 2,
      pointBackgroundColor: `${baseColor} 1)`,
      fill: true,
      tension: 0.4,
    };
  });

  radarChart = new Chart(document.getElementById('radarChart'), {
    type: 'radar',
    data: {
      labels: radarLabels,
      datasets: radarDatasets
    },
    options: { responsive: true, plugins: { legend: { position: 'top', labels: { color: '#1f2937' } } }, scales: { r: { angleLines: { color: '#b8c5d8' }, grid: { color: '#b8c5d8' }, pointLabels: { color: '#1f2937' }, ticks: { color: '#4d5f77', backdropColor: 'transparent' } } } }
  });
}

function showInputPage() {
  inputSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showResultsPage() {
  inputSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

newPredictionBtn.addEventListener('click', () => {
  showInputPage();
});

document.getElementById('predictBtn').addEventListener('click', async () => {
  const payload = gatherRequest();
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  renderResults(data);
  showResultsPage();
});

sectorEl.addEventListener('change', renderWeights);
renderWeights();
renderSuppliers(supplierDefaults);
showInputPage();

document.getElementById('needToggle').addEventListener('click', toggleNeedSection);
