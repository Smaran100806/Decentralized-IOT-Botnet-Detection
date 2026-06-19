/**
 * app.js
 * Main application controller. Wires up routing, data injection, and module initialization.
 */

const App = {
  async init() {
    console.log("Initializing Dashboard...");
    
    // Setup routing
    this.setupNavigation();
    
    // Load Data
    const data = await DataLoader.load();
    
    // Initialize modules
    NodeControl.init();
    NetworkGraph.init();
    DashboardCharts.init(data);
    
    // Inject Data
    this.injectOverview(data.overview);
    this.injectPreprocessing(data.preprocessing);
    this.injectBaseline(data.baseline);
    this.injectSpectral(data.graph);
    this.injectFederated(data.federated);
    
    // Setup Attack Section
    this.setupAttackTabs(data.attacks);
    
    // Show initial section
    this.handleRouting();
    
    // Initial animation
    setTimeout(() => document.getElementById('section-overview').classList.add('visible'), 100);
  },
  
  setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
      item.addEventListener('click', (e) => {
        // Only prevent default if it's an internal link
        if (item.getAttribute('href').startsWith('#')) {
          e.preventDefault();
          
          // Update nav state
          navItems.forEach(nav => nav.classList.remove('active'));
          item.classList.add('active');
          
          // Update sections
          const targetId = item.getAttribute('href').substring(1);
          document.querySelectorAll('.section').forEach(sec => {
            sec.classList.remove('visible');
            setTimeout(() => {
              if (sec.id !== targetId) sec.style.display = 'none';
            }, 500); // Wait for fade out
          });
          
          const targetSec = document.getElementById(targetId);
          if (targetSec) {
            targetSec.style.display = 'block';
            // Trigger reflow
            void targetSec.offsetWidth;
            targetSec.classList.add('visible');
          }
        }
      });
    });
    
    window.addEventListener('hashchange', this.handleRouting.bind(this));
  },
  
  handleRouting() {
    const hash = window.location.hash;
    if (!hash) return;
    
    const navItem = document.querySelector(`.nav-item[href="${hash}"]`);
    if (navItem) navItem.click();
  },
  
  injectOverview(overview) {
    if (!overview) return;
    
    const stats = overview.stats || {};
    document.getElementById('ov-samples').textContent = (stats.total_samples / 1000000).toFixed(1) + 'M';
    document.getElementById('ov-attacks').textContent = stats.attack_types;
    document.getElementById('ov-models').textContent = stats.models_trained;
    document.getElementById('ov-f1').textContent = stats.best_binary_f1.toFixed(4);
    
    // Pipeline Flow
    const flowContainer = document.getElementById('pipeline-flow-container');
    if (flowContainer && overview.pipeline_stages) {
      let html = '';
      overview.pipeline_stages.forEach((stage, idx) => {
        const icons = {
          'preprocess': '⚙️', 'baseline': '📈', 'graph': '🌐', 
          'spectral': '📉', 'attack': '⚔️', 'federated': '🔗'
        };
        const activeClass = idx === 0 ? 'active' : 'done';
        
        html += `
          <div class="pipeline-step">
            <div class="pipeline-node ${activeClass}">
              <div class="pipeline-node-icon">${icons[stage.id] || '•'}</div>
              <div class="pipeline-node-label">${stage.label}</div>
              <div class="pipeline-node-status">${stage.status}</div>
            </div>
            ${idx < overview.pipeline_stages.length - 1 ? '<div class="pipeline-arrow">→</div>' : ''}
          </div>
        `;
      });
      flowContainer.innerHTML = html;
    }
    
    // Targeted Threats List
    const threatsList = document.getElementById('targeted-threats-list');
    if (threatsList && overview.targeted_attacks) {
      let html = '';
      overview.targeted_attacks.forEach((threat, idx) => {
        html += `
          <div class="step-item">
            <div class="step-num" style="background:${threat.color}20; color:${threat.color}; border-color:${threat.color};">${idx + 1}</div>
            <div class="step-content">
              <div class="step-title">${threat.label}</div>
              <div class="step-detail">Accounts for ${threat.pct}% of the dataset. Target of dedicated binary classifier.</div>
            </div>
          </div>
        `;
      });
      threatsList.innerHTML = html;
    }
  },
  
  injectPreprocessing(prep) {
    if (!prep) return;
    
    const pipelineList = document.getElementById('prep-pipeline-list');
    if (pipelineList && prep.pipeline_steps) {
      let html = '';
      prep.pipeline_steps.forEach(step => {
        html += `
          <div class="step-item">
            <div class="step-num">${step.step}</div>
            <div class="step-content">
              <div class="step-title">${step.name}</div>
              <div class="step-detail">${step.detail}</div>
            </div>
          </div>
        `;
      });
      pipelineList.innerHTML = html;
    }
  },
  
  injectBaseline(baseline) {
    if (!baseline) return;
    
    const tableBody = document.getElementById('baseline-table');
    if (tableBody && baseline.binary_results) {
      let html = '';
      baseline.binary_results.forEach(res => {
        const isBest = res.model === baseline.best_binary_model;
        html += `
          <tr ${isBest ? 'style="background:rgba(79,157,232,0.1);"' : ''}>
            <td class="mono"><strong>${res.model}</strong> ${isBest ? '<span class="badge badge-high" style="margin-left:8px;">BEST</span>' : ''}</td>
            <td>${res.accuracy.toFixed(4)}</td>
            <td>${res.precision.toFixed(4)}</td>
            <td>${res.recall.toFixed(4)}</td>
            <td class="val-good">${res.f1.toFixed(4)}</td>
          </tr>
        `;
      });
      tableBody.innerHTML = html;
    }
  },
  
  injectSpectral(graph) {
    if (!graph) return;
    
    if (graph.partition_summary) {
      document.getElementById('sp-partitions').textContent = graph.partition_summary.n_partitions || '-';
      document.getElementById('sp-cross').textContent = graph.partition_summary.cross_partition_edges || '-';
    }
    
    if (graph.spectral_summary) {
      document.getElementById('sp-topk').textContent = graph.spectral_summary.top_k || '-';
      const fv = graph.fiedler_value || graph.spectral_summary.per_partition_summary?.['1']?.fiedler_value || 0;
      document.getElementById('sp-fiedler').textContent = fv.toFixed(4);
    }
  },
  
  injectFederated(fed) {
    if (!fed) return;
    
    document.getElementById('fl-fw').textContent = fed.fl_architecture?.framework || 'Flower';
    document.getElementById('fl-clients').textContent = fed.n_clients || 2;
    document.getElementById('fl-privacy').textContent = fed.privacy_note || 'Additive Masking';
  },
  
  setupAttackTabs(attackData) {
    if (!attackData || !attackData.attacks) return;
    
    const tabs = document.querySelectorAll('.tab-btn');
    
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => {
          t.classList.remove('active-ddos', 'active-syn', 'active-mirai');
          t.style.background = ''; t.style.color = ''; t.style.borderColor = '';
        });
        
        const type = tab.getAttribute('data-attack');
        
        // Add specific active class based on type
        if (type === 'ddos_icmp') tab.classList.add('active-ddos');
        else if (type === 'ddos_syn') tab.classList.add('active-syn');
        else if (type === 'mirai_greeth') tab.classList.add('active-mirai');
        
        this.renderAttackContent(type, attackData);
      });
    });
    
    // Render initial
    this.renderAttackContent('ddos_icmp', attackData);
  },
  
  renderAttackContent(type, attackData) {
    const data = attackData.attacks[type];
    if (!data) return;
    
    document.getElementById('ac-title').textContent = data.attack_display;
    document.getElementById('ac-best-model').textContent = `Best: ${data.best.model} (F1: ${data.best.f1.toFixed(4)})`;
    
    const trainInfo = data.rows ? 
      `${(data.rows.train_total/1000).toFixed(0)}K rows (Class balanced)` : 
      'Training Data Info';
    document.getElementById('ac-train-info').textContent = trainInfo;
    
    // Signatures (from signal strengths)
    const sigContainer = document.getElementById('ac-signatures');
    const signals = attackData.signal_strengths[type] || [];
    
    let sigHtml = '';
    let fillClass = 'fill-accent';
    if (type === 'ddos_icmp') fillClass = 'fill-red';
    if (type === 'mirai_greeth') fillClass = 'fill-cyan';
    
    signals.slice(0, 6).forEach(sig => {
      sigHtml += `
        <div class="feat-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div class="feat-name">${sig.feature}</div>
            <div class="badge badge-${sig.importance}">${sig.importance}</div>
          </div>
          <div class="signal-bar-wrap">
            <div class="signal-label"><span>Signal strength</span> <span>${sig.signal}%</span></div>
            <div class="signal-track"><div class="signal-fill ${fillClass}" style="width:${sig.signal}%"></div></div>
          </div>
        </div>
      `;
    });
    sigContainer.innerHTML = sigHtml;
    
    // Metrics table
    const tableBody = document.getElementById('ac-metrics-table');
    let tblHtml = '';
    
    // Sort results by F1
    const sortedResults = [...data.results].sort((a,b) => b.f1 - a.f1);
    
    sortedResults.forEach(res => {
      const isBest = res.model === data.best.model;
      tblHtml += `
        <tr ${isBest ? 'style="background:rgba(255,255,255,0.05);"' : ''}>
          <td class="mono"><strong>${res.model}</strong></td>
          <td>${res.accuracy.toFixed(4)}</td>
          <td>${res.precision.toFixed(4)}</td>
          <td>${res.recall.toFixed(4)}</td>
          <td class="${isBest ? 'val-good' : ''}">${res.f1.toFixed(4)}</td>
        </tr>
      `;
    });
    tableBody.innerHTML = tblHtml;
    
    // Extra features list
    const efContainer = document.getElementById('ac-extra-features');
    if (data.extra_features && data.extra_features.length > 0) {
      let efHtml = '<span style="font-size:11px; color:var(--text-muted); margin-top:5px;">Added Features:</span>';
      data.extra_features.forEach(ef => {
        efHtml += `<div class="metric-pill">${ef}</div>`;
      });
      efContainer.innerHTML = efHtml;
    } else {
      efContainer.innerHTML = '';
    }
    
    // Update chart
    if (window.DashboardCharts) {
      DashboardCharts.renderConfusionMatrix(data.confusion_matrix, type);
    }
  }
};

// Boot
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});
