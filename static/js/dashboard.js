/**
 * static/js/dashboard.js
 * Interactive controls, Chart.js degradation forecasting, and real-time ML diagnostic inference.
 */

let forecastChartInstance = null;
let debounceTimer = null;

// Presets data fallback
const LOCAL_PRESETS = {
    brand_new: {
        voltage: 4.12, current: 1.2, temperature: 24.5, capacity: 2.01,
        cycle: 5, time: 18000, resistance: 48.0, impedance: 0.038,
        soc: 92.0, soh: 99.5, age: 4.0
    },
    moderate_aging: {
        voltage: 3.68, current: 1.8, temperature: 31.0, capacity: 1.74,
        cycle: 150, time: 540000, resistance: 76.5, impedance: 0.058,
        soc: 75.0, soh: 86.5, age: 115.0
    },
    second_life: {
        voltage: 3.42, current: 2.1, temperature: 38.5, capacity: 1.48,
        cycle: 280, time: 1008000, resistance: 115.0, impedance: 0.082,
        soc: 60.0, soh: 73.8, age: 210.0
    },
    critical_thermal: {
        voltage: 3.15, current: 2.4, temperature: 48.2, capacity: 1.28,
        cycle: 360, time: 1296000, resistance: 158.0, impedance: 0.115,
        soc: 45.0, soh: 63.5, age: 275.0
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initForecastChart();
    runPrediction(); // Initial diagnostic on load
});

function syncInput(param, val) {
    const numInput = document.getElementById(param);
    const labelSpan = document.getElementById(`${param}_val`);
    if (numInput) numInput.value = val;
    if (labelSpan) {
        if (param === 'voltage') labelSpan.innerText = `${parseFloat(val).toFixed(2)} V`;
        else if (param === 'current') labelSpan.innerText = `${parseFloat(val).toFixed(2)} A`;
        else if (param === 'soc') labelSpan.innerText = `${parseFloat(val).toFixed(1)} %`;
        else if (param === 'cycle') labelSpan.innerText = `${parseInt(val)}`;
        else if (param === 'capacity') labelSpan.innerText = `${parseFloat(val).toFixed(2)} Ah`;
        else if (param === 'temperature') labelSpan.innerText = `${parseFloat(val).toFixed(1)} °C`;
    }
    onParamChange();
}

function syncSlider(param, val) {
    const slider = document.getElementById(`${param}_range`);
    const labelSpan = document.getElementById(`${param}_val`);
    if (slider) slider.value = val;
    if (labelSpan) {
        if (param === 'voltage') labelSpan.innerText = `${parseFloat(val).toFixed(2)} V`;
        else if (param === 'current') labelSpan.innerText = `${parseFloat(val).toFixed(2)} A`;
        else if (param === 'soc') labelSpan.innerText = `${parseFloat(val).toFixed(1)} %`;
        else if (param === 'cycle') labelSpan.innerText = `${parseInt(val)}`;
        else if (param === 'capacity') labelSpan.innerText = `${parseFloat(val).toFixed(2)} Ah`;
        else if (param === 'temperature') labelSpan.innerText = `${parseFloat(val).toFixed(1)} °C`;
    }
    onParamChange();
}

function onParamChange() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        runPrediction();
    }, 250);
}

function loadPreset(presetKey) {
    const p = LOCAL_PRESETS[presetKey];
    if (!p) return;

    for (const [k, v] of Object.entries(p)) {
        const numInput = document.getElementById(k);
        const slider = document.getElementById(`${k}_range`);
        const label = document.getElementById(`${k}_val`);

        if (numInput) numInput.value = v;
        if (slider) slider.value = v;
        if (label) {
            if (k === 'voltage') label.innerText = `${v} V`;
            else if (k === 'current') label.innerText = `${v} A`;
            else if (k === 'soc') label.innerText = `${v} %`;
            else if (k === 'cycle') label.innerText = `${v}`;
            else if (k === 'capacity') label.innerText = `${v} Ah`;
            else if (k === 'temperature') label.innerText = `${v} °C`;
            else if (k === 'soh') label.innerText = `${v} %`;
        }
    }
    runPrediction();
}

function getFormPayload() {
    return {
        voltage: parseFloat(document.getElementById('voltage')?.value || 3.7),
        current: parseFloat(document.getElementById('current')?.value || 1.5),
        temperature: parseFloat(document.getElementById('temperature')?.value || 25.0),
        capacity: parseFloat(document.getElementById('capacity')?.value || 1.8),
        cycle: parseInt(document.getElementById('cycle')?.value || 100),
        time: parseFloat(document.getElementById('time')?.value || 360000),
        resistance: parseFloat(document.getElementById('resistance')?.value || 65.0),
        impedance: parseFloat(document.getElementById('impedance')?.value || 0.05),
        soc: parseFloat(document.getElementById('soc')?.value || 80.0),
        soh: document.getElementById('soh')?.value ? parseFloat(document.getElementById('soh').value) : null,
        age: parseFloat(document.getElementById('age')?.value || 80.0)
    };
}

function runPrediction() {
    const payload = getFormPayload();
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        submitBtn.classList.add('opacity-90');
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(res => {
        if (submitBtn) submitBtn.classList.remove('opacity-90');
        if (res.success) {
            updateDashboardUI(res.data);
        } else {
            console.error('Prediction API Error:', res.error);
        }
    })
    .catch(err => {
        if (submitBtn) submitBtn.classList.remove('opacity-90');
        console.error('Fetch error:', err);
    });
}

function updateDashboardUI(data) {
    const preds = data.predictions;
    const advisories = data.advisories;
    const forecast = data.forecast;

    // 1. Update SOH Circular Gauge
    const sohVal = preds.soh_final;
    document.getElementById('sohDisplay').innerText = `${sohVal.toFixed(1)}%`;
    
    // Circumference = 2 * PI * 48 = 301.59
    const circumference = 301.59;
    const offset = circumference - (sohVal / 100.0) * circumference;
    const circle = document.getElementById('sohCircle');
    if (circle) {
        circle.style.strokeDashoffset = Math.max(0, offset);
        
        // Color transition
        if (sohVal >= 90) {
            circle.setAttribute('class', 'text-emerald-400 transition-all duration-700 ease-out');
        } else if (sohVal >= 80) {
            circle.setAttribute('class', 'text-teal-400 transition-all duration-700 ease-out');
        } else if (sohVal >= 70) {
            circle.setAttribute('class', 'text-amber-400 transition-all duration-700 ease-out');
        } else {
            circle.setAttribute('class', 'text-rose-500 transition-all duration-700 ease-out');
        }
    }

    // Health status badge
    const badge = document.getElementById('healthBadge');
    if (badge) {
        badge.innerText = preds.health_status;
        if (sohVal >= 90) {
            badge.className = 'mt-2 px-2.5 py-1 text-xs font-bold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
        } else if (sohVal >= 80) {
            badge.className = 'mt-2 px-2.5 py-1 text-xs font-bold rounded-full bg-teal-500/10 text-teal-400 border border-teal-500/30';
        } else if (sohVal >= 70) {
            badge.className = 'mt-2 px-2.5 py-1 text-xs font-bold rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30';
        } else {
            badge.className = 'mt-2 px-2.5 py-1 text-xs font-bold rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 animate-pulse';
        }
    }

    // 2. RUL & Range
    document.getElementById('rulCycles').innerText = preds.rul_cycles.toLocaleString();
    document.getElementById('estKm').innerText = preds.est_range_km.toLocaleString();
    document.getElementById('estMiles').innerText = `${preds.est_range_miles.toLocaleString()} mi`;

    const riskElem = document.getElementById('riskLevel');
    if (riskElem) {
        riskElem.innerText = preds.risk_level;
        if (preds.risk_level === 'LOW') riskElem.className = 'font-bold text-emerald-400';
        else if (preds.risk_level === 'MODERATE') riskElem.className = 'font-bold text-amber-400';
        else riskElem.className = 'font-bold text-rose-400 animate-pulse';
    }

    // 3. Update Forecast Chart
    if (forecastChartInstance && forecast) {
        forecastChartInstance.data.labels = forecast.cycles;
        forecastChartInstance.data.datasets[0].data = forecast.capacity;
        forecastChartInstance.data.datasets[1].data = forecast.cycles.map(() => 1.40); // 70% threshold
        forecastChartInstance.update();
    }

    // 4. Update BMS Advisories
    const container = document.getElementById('advisoriesContainer');
    const countBadge = document.getElementById('advisoryCount');
    if (container && advisories) {
        container.innerHTML = '';
        if (countBadge) countBadge.innerText = `${advisories.length} Note${advisories.length > 1 ? 's' : ''}`;

        advisories.forEach(adv => {
            const card = document.createElement('div');
            let borderClass = 'border-slate-800 bg-slate-800/40 text-slate-300';
            let icon = 'info';
            let iconColor = 'text-cyan-400';

            if (adv.type === 'CRITICAL') {
                borderClass = 'border-rose-500/30 bg-rose-500/10 text-rose-200';
                icon = 'alert-octagon';
                iconColor = 'text-rose-400';
            } else if (adv.type === 'WARNING' || adv.type === 'ALERT') {
                borderClass = 'border-amber-500/30 bg-amber-500/10 text-amber-200';
                icon = 'alert-triangle';
                iconColor = 'text-amber-400';
            } else if (adv.type === 'GOOD') {
                borderClass = 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200';
                icon = 'shield-check';
                iconColor = 'text-emerald-400';
            }

            card.className = `p-3 rounded-xl border flex items-start gap-3 text-xs ${borderClass}`;
            card.innerHTML = `
                <i data-lucide="${icon}" class="w-4 h-4 ${iconColor} shrink-0 mt-0.5"></i>
                <div class="space-y-0.5">
                    <div class="font-bold tracking-wide">${adv.title}</div>
                    <div class="text-[11px] opacity-90 leading-relaxed">${adv.detail}</div>
                </div>
            `;
            container.appendChild(card);
        });

        lucide.createIcons();
    }
}

function initForecastChart() {
    const ctx = document.getElementById('forecastChart');
    if (!ctx) return;

    forecastChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [150, 180, 210, 240, 270, 300, 330, 360],
            datasets: [
                {
                    label: 'Projected Capacity (Ah)',
                    data: [1.74, 1.69, 1.64, 1.58, 1.52, 1.46, 1.40, 1.35],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.12)',
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#10b981',
                    pointRadius: 3,
                    borderWidth: 2
                },
                {
                    label: 'EOL Knee Point (1.40 Ah / 70% SOH)',
                    data: [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
                    borderColor: '#ef4444',
                    borderDash: [6, 6],
                    pointRadius: 0,
                    borderWidth: 1.5,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#94a3b8', font: { size: 11 } }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Operating Cycles', color: '#94a3b8', font: { size: 11 } },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#1e293b' }
                },
                y: {
                    title: { display: true, text: 'Capacity (Ah)', color: '#94a3b8', font: { size: 11 } },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#1e293b' }
                }
            }
        }
    });
}
