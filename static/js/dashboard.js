/**
 * static/js/dashboard.js
 * Intuitive controls, Simple vs Advanced modes, Smart Physics Auto-linking,
 * and Parameter Impact / Explainability visualizations.
 */

let forecastChartInstance = null;
let debounceTimer = null;
let isAutoLinkEnabled = true;
let currentMode = 'simple'; // 'simple' or 'advanced'

// Presets aligned with real NASA battery aging
const LOCAL_PRESETS = {
    brand_new: {
        cycle: 5, temperature: 24.5, current: 1.2, soc: 95.0,
        capacity: 1.99, voltage: 4.15, resistance: 49.0, impedance: 0.039,
        time: 18000, age: 4.0, soh: 99.5, style: 'Mild Eco'
    },
    moderate_aging: {
        cycle: 150, temperature: 31.0, current: 1.8, soc: 75.0,
        capacity: 1.74, voltage: 3.68, resistance: 76.5, impedance: 0.058,
        time: 540000, age: 115.0, soh: 86.5, style: 'Standard'
    },
    second_life: {
        cycle: 280, temperature: 38.5, current: 2.1, soc: 60.0,
        capacity: 1.48, voltage: 3.42, resistance: 115.0, impedance: 0.082,
        time: 1008000, age: 210.0, soh: 73.8, style: 'Standard'
    },
    critical_thermal: {
        cycle: 360, temperature: 48.2, current: 2.5, soc: 45.0,
        capacity: 1.28, voltage: 3.15, resistance: 158.0, impedance: 0.115,
        time: 1296000, age: 275.0, soh: 63.5, style: 'Aggressive'
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initForecastChart();
    // Initialize simple controls sync
    updateSimpleControlsUIFromForm();
    runPrediction(); // Initial diagnostic
});

// Mode switcher: Simple vs Advanced
function setDashboardMode(mode) {
    currentMode = mode;
    const tabSimple = document.getElementById('tabSimple');
    const tabAdv = document.getElementById('tabAdvanced');
    const simpleDiv = document.getElementById('simpleControls');
    const advDiv = document.getElementById('advancedControls');

    if (mode === 'simple') {
        tabSimple.className = "px-3 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm transition-all flex items-center gap-1.5";
        tabAdv.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition-all flex items-center gap-1.5";
        simpleDiv.classList.remove('hidden');
        advDiv.classList.add('hidden');
    } else {
        tabAdv.className = "px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 shadow-sm transition-all flex items-center gap-1.5";
        tabSimple.className = "px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition-all flex items-center gap-1.5";
        advDiv.classList.remove('hidden');
        simpleDiv.classList.add('hidden');
    }
}

// Collapsible "How Parameters Affect Life" Guide
function toggleRelationshipGuide() {
    const guide = document.getElementById('relationshipGuide');
    const text = document.getElementById('guideToggleText');
    const icon = document.getElementById('guideToggleIcon');
    if (!guide) return;

    if (guide.classList.contains('hidden')) {
        guide.classList.remove('hidden');
        text.innerText = "Hide Relationship Guide";
        icon.setAttribute('data-lucide', 'chevron-up');
    } else {
        guide.classList.add('hidden');
        text.innerText = "Show Relationship Guide";
        icon.setAttribute('data-lucide', 'chevron-down');
    }
    lucide.createIcons();
}

// Toggle Auto-linking electrochemical physics
function toggleAutoLink(enabled) {
    isAutoLinkEnabled = enabled;
    if (enabled) {
        syncPhysicsFromCycles();
        onParamChange();
    }
}

// Handler when user tweaks Simple Mode inputs
function onSimpleParamChange(type, val) {
    const floatVal = parseFloat(val);

    if (type === 'cycle') {
        document.getElementById('simple_cycle_display').innerText = `${parseInt(val)} cycles`;
        document.getElementById('cycle').value = parseInt(val);
        
        // Cycle badge
        const badge = document.getElementById('simple_cycle_badge');
        if (floatVal < 50) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
            badge.innerText = `New Cell (${parseInt(val)}c)`;
        } else if (floatVal < 200) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-teal-500/10 text-teal-400 border border-teal-500/20";
            badge.innerText = `Mid-Life (${parseInt(val)}c)`;
        } else if (floatVal < 320) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20";
            badge.innerText = `High Mileage (${parseInt(val)}c)`;
        } else {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20 animate-pulse";
            badge.innerText = `End of Life (${parseInt(val)}c)`;
        }

        if (isAutoLinkEnabled) {
            syncPhysicsFromCycles();
        }
    } else if (type === 'temp') {
        document.getElementById('simple_temp_display').innerText = `${floatVal.toFixed(1)} °C`;
        document.getElementById('temperature').value = floatVal;

        const badge = document.getElementById('simple_temp_badge');
        if (floatVal < 18) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20";
            badge.innerText = `Cool (${floatVal.toFixed(1)}°C)`;
        } else if (floatVal <= 32) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
            badge.innerText = `Optimal (${floatVal.toFixed(1)}°C)`;
        } else if (floatVal <= 40) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20";
            badge.innerText = `Elevated Heat (${floatVal.toFixed(1)}°C)`;
        } else {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20 animate-pulse";
            badge.innerText = `Critical Heat (${floatVal.toFixed(1)}°C)`;
        }

        if (isAutoLinkEnabled) {
            syncPhysicsFromCycles();
        }
    } else if (type === 'soc') {
        document.getElementById('simple_soc_display').innerText = `${parseInt(val)}%`;
        document.getElementById('soc').value = parseInt(val);
        if (isAutoLinkEnabled) {
            syncPhysicsFromCycles();
        }
    }

    onParamChange();
}

// Set Driving Style
function setDrivingStyle(currentVal, styleName) {
    document.getElementById('current').value = currentVal;
    const badge = document.getElementById('simple_style_badge');
    badge.innerText = `${styleName} (${currentVal.toFixed(2)} A)`;

    // Highlight button
    const ecoBtn = document.getElementById('styleEco');
    const stdBtn = document.getElementById('styleStd');
    const sportBtn = document.getElementById('styleSport');

    ecoBtn.className = "py-1.5 text-xs rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-emerald-400 transition-all text-center";
    stdBtn.className = "py-1.5 text-xs rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-teal-400 transition-all text-center";
    sportBtn.className = "py-1.5 text-xs rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-amber-400 transition-all text-center";

    if (currentVal < 1.5) {
        ecoBtn.className = "py-1.5 text-xs rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-bold transition-all text-center";
        badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
    } else if (currentVal < 2.2) {
        stdBtn.className = "py-1.5 text-xs rounded-lg bg-teal-500/20 border border-teal-500/40 text-teal-300 font-bold transition-all text-center";
        badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-teal-500/10 text-teal-400 border border-teal-500/20";
    } else {
        sportBtn.className = "py-1.5 text-xs rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 font-bold transition-all text-center";
        badge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20";
    }

    if (isAutoLinkEnabled) {
        syncPhysicsFromCycles();
    }
    onParamChange();
}

// Auto-derives Capacity, Resistance, Impedance, Voltage, Age, Time from cycles and physics
function syncPhysicsFromCycles() {
    const cycle = parseInt(document.getElementById('cycle')?.value || 150);
    const temp = parseFloat(document.getElementById('temperature')?.value || 31.0);
    const soc = parseFloat(document.getElementById('soc')?.value || 75.0);
    const current = parseFloat(document.getElementById('current')?.value || 1.8);

    // Thermal acceleration factor (>30°C degrades faster)
    const thermalStress = temp > 30.0 ? 1.0 + (temp - 30.0) * 0.02 : 1.0;

    // Realistic capacity degradation (nominal 2.0 Ah fresh cell)
    const capFade = ((cycle * 0.00185) + Math.pow(cycle, 1.08) * 0.000075) * thermalStress;
    const capacity = Math.max(1.15, Math.min(2.02, 2.0 - capFade));

    // Internal resistance growth (48 mOhm initial fresh cell)
    const resSurge = (cycle * 0.26 + Math.pow(cycle, 1.32) * 0.008) * thermalStress;
    const resistance = Math.min(320.0, 48.0 + resSurge);

    // Impedance growth
    const impedance = Math.min(0.35, 0.038 + (cycle * 0.00017) * thermalStress);

    // Elapsed time & age
    const timeVal = cycle * 3600.0;
    const ageDays = Math.round(cycle * 0.75 + 2.0);

    // Loaded voltage: OCV(SoC) - (I * R)
    const ocv = 3.2 + (soc / 100.0) * 0.95;
    const loadedVolt = Math.min(4.25, Math.max(2.7, ocv - (current * (resistance / 1000.0))));

    // Update Advanced inputs silently
    document.getElementById('capacity').value = capacity.toFixed(2);
    document.getElementById('resistance').value = resistance.toFixed(1);
    document.getElementById('impedance').value = impedance.toFixed(3);
    document.getElementById('voltage').value = loadedVolt.toFixed(2);
    document.getElementById('time').value = timeVal;
    document.getElementById('age').value = ageDays;
    
    // Clear custom SOH override so model calculates accurate prediction
    document.getElementById('soh').value = '';
}

// When user edits advanced inputs directly
function syncAdvancedInput(param, val) {
    if (param === 'cycle') {
        const cVal = parseInt(val);
        document.getElementById('simple_cycle_range').value = cVal;
        document.getElementById('simple_cycle_display').innerText = `${cVal} cycles`;
    } else if (param === 'temperature') {
        const tVal = parseFloat(val);
        document.getElementById('simple_temp_range').value = tVal;
        document.getElementById('simple_temp_display').innerText = `${tVal.toFixed(1)} °C`;
    } else if (param === 'soc') {
        const sVal = parseInt(val);
        document.getElementById('simple_soc_range').value = sVal;
        document.getElementById('simple_soc_display').innerText = `${sVal}%`;
    } else if (param === 'current') {
        const curVal = parseFloat(val);
        setDrivingStyle(curVal, curVal < 1.5 ? 'Mild Eco' : (curVal < 2.2 ? 'Standard' : 'Aggressive'));
    }
    onParamChange();
}

function updateSimpleControlsUIFromForm() {
    const cycle = parseInt(document.getElementById('cycle')?.value || 150);
    const temp = parseFloat(document.getElementById('temperature')?.value || 31.0);
    const soc = parseInt(document.getElementById('soc')?.value || 75);
    const current = parseFloat(document.getElementById('current')?.value || 1.8);

    document.getElementById('simple_cycle_range').value = cycle;
    document.getElementById('simple_cycle_display').innerText = `${cycle} cycles`;

    document.getElementById('simple_temp_range').value = temp;
    document.getElementById('simple_temp_display').innerText = `${temp.toFixed(1)} °C`;

    document.getElementById('simple_soc_range').value = soc;
    document.getElementById('simple_soc_display').innerText = `${soc}%`;

    setDrivingStyle(current, current < 1.5 ? 'Mild Eco' : (current < 2.2 ? 'Standard' : 'Aggressive'));
}

function onParamChange() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        runPrediction();
    }, 200);
}

function loadPreset(presetKey) {
    const p = LOCAL_PRESETS[presetKey];
    if (!p) return;

    for (const [k, v] of Object.entries(p)) {
        const elem = document.getElementById(k);
        if (elem) elem.value = v;
    }

    // Sync simple mode controls
    updateSimpleControlsUIFromForm();
    runPrediction();
}

function getFormPayload() {
    return {
        voltage: parseFloat(document.getElementById('voltage')?.value || 3.68),
        current: parseFloat(document.getElementById('current')?.value || 1.8),
        temperature: parseFloat(document.getElementById('temperature')?.value || 31.0),
        capacity: parseFloat(document.getElementById('capacity')?.value || 1.74),
        cycle: parseInt(document.getElementById('cycle')?.value || 150),
        time: parseFloat(document.getElementById('time')?.value || 540000),
        resistance: parseFloat(document.getElementById('resistance')?.value || 76.5),
        impedance: parseFloat(document.getElementById('impedance')?.value || 0.058),
        soc: parseFloat(document.getElementById('soc')?.value || 75.0),
        soh: document.getElementById('soh')?.value ? parseFloat(document.getElementById('soh').value) : null,
        age: parseFloat(document.getElementById('age')?.value || 115.0)
    };
}

function runPrediction() {
    const payload = getFormPayload();
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) submitBtn.classList.add('opacity-90');

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
        }
    })
    .catch(err => {
        if (submitBtn) submitBtn.classList.remove('opacity-90');
        console.error('Fetch error:', err);
    });
}

function updateDashboardUI(data) {
    const preds = data.predictions;
    const impacts = data.impacts || {};
    const advisories = data.advisories;
    const forecast = data.forecast;
    const inputs = data.inputs;

    // 1. Verdict & Status Banner
    const sohVal = preds.soh_final;
    document.getElementById('sohDisplay').innerText = `${sohVal.toFixed(1)}%`;
    document.getElementById('capacityRetentionText').innerText = `${inputs.capacity.toFixed(2)} / 2.00 Ah`;

    // Title and summary
    const titleText = document.getElementById('healthTitleText');
    const descText = document.getElementById('summaryDescText');
    const badge = document.getElementById('healthBadge');
    const yearsText = document.getElementById('estYearsText');
    const urgencyText = document.getElementById('urgencyText');

    if (sohVal >= 90) {
        titleText.innerText = "Excellent Condition";
        descText.innerText = "Battery operates with minimal wear. Capable of peak performance & full duty cycle.";
        badge.className = "text-xs px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold";
        badge.innerText = "Like-New (Optimal)";
    } else if (sohVal >= 80) {
        titleText.innerText = "Good Condition";
        descText.innerText = "Normal cyclic aging observed. Meets all automotive specifications for daily driving.";
        badge.className = "text-xs px-2.5 py-1 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/40 font-bold";
        badge.innerText = "Normal Aging";
    } else if (sohVal >= 70) {
        titleText.innerText = "Degraded Condition";
        descText.innerText = "Capacity approaching the 70% EV retirement threshold. Ideal candidate for stationary solar/grid storage.";
        badge.className = "text-xs px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold";
        badge.innerText = "Second-Life Candidate";
    } else {
        titleText.innerText = "Critical / End-of-Life";
        descText.innerText = "Capacity is below safe automotive threshold. High internal resistance increases risk of rapid discharge.";
        badge.className = "text-xs px-2.5 py-1 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold animate-pulse";
        badge.innerText = "Replacement Required";
    }

    const yearsRemaining = preds.est_years_remaining || Math.round(preds.rul_cycles / 180.0 * 10) / 10;
    yearsText.innerText = `~${yearsRemaining} Years Left`;
    urgencyText.innerText = preds.urgency || (sohVal >= 75 ? "No action needed" : "Plan replacement soon");

    // 2. Circular Gauge
    const circumference = 301.59;
    const offset = circumference - (sohVal / 100.0) * circumference;
    const circle = document.getElementById('sohCircle');
    if (circle) {
        circle.style.strokeDashoffset = Math.max(0, offset);
        if (sohVal >= 90) circle.setAttribute('class', 'text-emerald-400 transition-all duration-700 ease-out');
        else if (sohVal >= 80) circle.setAttribute('class', 'text-teal-400 transition-all duration-700 ease-out');
        else if (sohVal >= 70) circle.setAttribute('class', 'text-amber-400 transition-all duration-700 ease-out');
        else circle.setAttribute('class', 'text-rose-500 transition-all duration-700 ease-out');
    }

    // 3. RUL & Range
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

    // 4. Parameter Impact Breakdown (Cause and Effect)
    const capLoss = impacts.capacity_loss_pct !== undefined ? impacts.capacity_loss_pct : Math.round((2.0 - inputs.capacity) / 2.0 * 100);
    document.getElementById('impactCapLoss').innerText = `-${capLoss.toFixed(1)}%`;
    document.getElementById('impactCapBar').style.width = `${Math.min(100, capLoss * 2)}%`;
    document.getElementById('impactCapDesc').innerText = `Nominal cell was 2.00 Ah; currently holds ${inputs.capacity.toFixed(2)} Ah.`;

    const cycleWear = impacts.cycle_wear_level || (inputs.cycle < 100 ? 'Low' : (inputs.cycle < 250 ? 'Moderate' : 'High'));
    document.getElementById('impactCycleWear').innerText = cycleWear;
    document.getElementById('impactCycleBar').style.width = `${Math.min(100, (inputs.cycle / 350) * 100)}%`;
    document.getElementById('impactCycleDesc').innerText = `${inputs.cycle} cycles completed out of ~360 total expected cycles.`;

    const tempScore = impacts.thermal_score || (inputs.temperature <= 32 ? 'Optimal' : (inputs.temperature <= 40 ? 'Warm' : 'Severe'));
    document.getElementById('impactThermalScore').innerText = `${tempScore} (${inputs.temperature.toFixed(1)}°C)`;
    document.getElementById('impactThermalBar').style.width = `${Math.min(100, Math.max(10, (inputs.temperature / 50) * 100))}%`;
    document.getElementById('impactThermalDesc').innerText = impacts.thermal_impact || `Operating at ${inputs.temperature.toFixed(1)}°C.`;

    const resHealth = impacts.resistance_health || `${inputs.resistance.toFixed(1)} mΩ`;
    document.getElementById('impactResHealth').innerText = `${inputs.resistance.toFixed(1)} mΩ`;
    document.getElementById('impactResBar').style.width = `${Math.min(100, Math.max(15, (inputs.resistance / 160) * 100))}%`;
    document.getElementById('impactResDesc').innerText = inputs.resistance < 80 ? 'Low resistance. High acceleration power & efficiency.' : 'Increased resistance causes voltage sag under acceleration.';

    // 5. Update Forecast Chart
    if (forecastChartInstance && forecast) {
        forecastChartInstance.data.labels = forecast.cycles;
        forecastChartInstance.data.datasets[0].data = forecast.capacity;
        forecastChartInstance.data.datasets[1].data = forecast.cycles.map(() => 1.40); // 70% threshold
        forecastChartInstance.update();
    }

    // 6. Update BMS Advisories
    const container = document.getElementById('advisoriesContainer');
    const countBadge = document.getElementById('advisoryCount');
    if (container && advisories) {
        container.innerHTML = '';
        countBadge.innerText = `${advisories.length} Note${advisories.length > 1 ? 's' : ''}`;

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
                    label: 'EOL Cutoff Knee Point (1.40 Ah / 70% SOH)',
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
                    title: { display: true, text: 'Operating Charge Cycles', color: '#94a3b8', font: { size: 11 } },
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
