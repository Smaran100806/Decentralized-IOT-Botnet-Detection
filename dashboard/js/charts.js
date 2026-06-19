/**
 * charts.js
 * Chart.js configurations and rendering for all dashboard sections.
 */

Chart.defaults.color = '#8890a8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.borderColor = '#272b38';

const DashboardCharts = {
  charts: {},
  
  init(data) {
    this.renderFeatureOverlap(data.preprocessing);
    this.renderClassDistribution(data.preprocessing);
    this.renderSpectral(data.graph);
    this.renderFederated(data.federated);
    
    // Attack confusion matrix (initially DDoS-ICMP)
    if (data.attacks && data.attacks.attacks && data.attacks.attacks['ddos_icmp']) {
      this.renderConfusionMatrix(data.attacks.attacks['ddos_icmp'].confusion_matrix, 'ddos_icmp');
    }
  },
  
  renderFeatureOverlap(prep) {
    const ctx = document.getElementById('chart-features');
    if (!ctx) return;
    
    const table = prep.feature_table || [];
    const labels = table.slice(0, 15).map(r => r.feature);
    const pearson = table.slice(0, 15).map(r => r.pearson_r);
    const mi = table.slice(0, 15).map(r => r.mutual_info);
    
    this.charts.features = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          { label: 'Pearson |r|', data: pearson, backgroundColor: 'rgba(79,157,232,0.7)', borderRadius: 4 },
          { label: 'Mutual Info', data: mi, backgroundColor: 'rgba(72,196,142,0.7)', borderRadius: 4 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { color: '#e8eaf0' } },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          y: { beginAtZero: true, grid: { color: '#1e2333' } },
          x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } }
        }
      }
    });
  },
  
  renderClassDistribution(prep) {
    const ctx = document.getElementById('chart-classes');
    if (!ctx) return;
    
    const dist = prep.class_distribution || [];
    // Sort by pct desc, take top 6, group rest as "Other"
    dist.sort((a,b) => b.pct - a.pct);
    const top = dist.slice(0, 6);
    const otherPct = dist.slice(6).reduce((sum, item) => sum + item.pct, 0);
    
    if (otherPct > 0) {
      top.push({ family: 'Other', pct: otherPct });
    }
    
    this.charts.classes = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: top.map(d => d.family),
        datasets: [{
          data: top.map(d => d.pct),
          backgroundColor: ['#4f9de8', '#48c48e', '#e05c5c', '#a48ee8', '#e8914f', '#e8c84f', '#5a6180'],
          borderWidth: 0,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { color: '#e8eaf0', font: { size: 10 } } },
          tooltip: { callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%` } }
        },
        cutout: '70%'
      }
    });
  },
  
  renderSpectral(graph) {
    const ctx = document.getElementById('chart-spectral');
    if (!ctx) return;
    
    const evs = graph.eigenvalues || [];
    
    this.charts.spectral = new Chart(ctx, {
      type: 'line',
      data: {
        labels: evs.map((_, i) => `λ${i}`),
        datasets: [{
          label: 'Eigenvalue',
          data: evs,
          borderColor: '#4f9de8',
          backgroundColor: 'rgba(79,157,232,0.1)',
          borderWidth: 2,
          pointBackgroundColor: '#0a0c12',
          pointBorderColor: '#4f9de8',
          pointBorderWidth: 2,
          pointRadius: 4,
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { grid: { color: '#1e2333' } },
          x: { grid: { color: '#1e2333' } }
        }
      }
    });
  },
  
  renderFederated(fed) {
    const ctx = document.getElementById('chart-fl');
    if (!ctx) return;
    
    const rounds = fed.rounds || [];
    const labels = rounds.map(r => `Round ${r.round}`);
    const f1s = rounds.map(r => r.global_f1);
    
    this.charts.fl = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Global Model F1 Score',
          data: f1s,
          borderColor: '#48c48e',
          borderWidth: 3,
          pointBackgroundColor: '#48c48e',
          pointRadius: 5,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e8eaf0' } } },
        scales: {
          y: { grid: { color: '#1e2333' }, min: Math.max(0.9, Math.min(...f1s) - 0.02) },
          x: { grid: { color: '#1e2333' } }
        }
      }
    });
  },
  
  renderConfusionMatrix(cm, type) {
    const ctx = document.getElementById('chart-confusion');
    if (!ctx) return;
    
    // Chart.js doesn't have a native heatmap, so we use a scatter/bubble workaround or 
    // a simplified bar chart if heatmap is too complex. For simplicity, we use a 
    // double bar chart showing True Positives, False Positives etc.
    
    if (this.charts.confusion) {
      this.charts.confusion.destroy();
    }
    
    let color = '#4f9de8'; // default
    if (type === 'ddos_icmp') color = '#e05c5c';
    if (type === 'mirai_greeth') color = '#48c48e';
    
    // cm is [[TN, FP], [FN, TP]]
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    
    this.charts.confusion = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
        datasets: [{
          label: 'Samples',
          data: [tn, fp, fn, tp],
          backgroundColor: [
            'rgba(90, 97, 128, 0.7)', // TN: Grey
            'rgba(224, 92, 92, 0.7)',  // FP: Red
            'rgba(232, 145, 79, 0.7)', // FN: Orange
            color                      // TP: Attack Color
          ],
          borderWidth: 0,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { type: 'logarithmic', grid: { color: '#1e2333' } }, // Log scale because TP/TN are huge, FP/FN tiny
          x: { grid: { display: false } }
        }
      }
    });
  }
};
