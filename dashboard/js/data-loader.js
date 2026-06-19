/**
 * data-loader.js
 * Fetches dashboard_data.json and provides a global API for other modules.
 */

const DataLoader = {
  data: null,
  
  async load() {
    try {
      const response = await fetch('./data/dashboard_data.json');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      this.data = await response.json();
      return this.data;
    } catch (e) {
      console.error('Error loading dashboard data:', e);
      // Create fallback dummy data if not running on a server
      this.createFallbackData();
      return this.data;
    }
  },
  
  createFallbackData() {
    console.warn("Using fallback data for dashboard");
    this.data = {
      overview: {
        stats: { total_samples: 7845673, attack_types: 34, models_trained: 16, best_binary_f1: 0.9967 },
        pipeline_stages: [
          {id: 'preprocess', label: 'Preprocessing', status: 'done', detail: 'Clean + scale 7.8M rows'},
          {id: 'baseline', label: 'Baseline', status: 'done', detail: 'Best F1: 0.9967'},
          {id: 'attack', label: 'Attack Models', status: 'done', detail: 'OVR models for 3 attacks'}
        ],
        targeted_attacks: [
          {key: 'ddos_icmp', label: 'DDoS-ICMP Flood', color: '#e05c5c', pct: 15.3},
          {key: 'ddos_syn', label: 'DDoS-SYN Flood', color: '#4f9de8', pct: 8.7},
          {key: 'mirai_greeth', label: 'Mirai-Greeth_flood', color: '#48c48e', pct: 5.0}
        ]
      },
      preprocessing: {
        feature_table: [],
        class_distribution: [],
        pipeline_steps: [],
        total_rows: { train: 5491971 },
        feature_count: 17
      },
      baseline: { binary_results: [] },
      attacks: { attacks: {}, signal_strengths: {} },
      graph: { graph_summary: {nodes:20}, partition_summary: {}, spectral_summary: {}, eigenvalues: [1,2,3,4,5,6,7,8] },
      federated: { rounds: [], n_clients: 2, n_rounds: 2, final_eval: {} }
    };
  },
  
  getOverview() { return this.data?.overview || {}; },
  getPreprocessing() { return this.data?.preprocessing || {}; },
  getBaseline() { return this.data?.baseline || {}; },
  getAttacks() { return this.data?.attacks || {}; },
  getGraph() { return this.data?.graph || {}; },
  getFederated() { return this.data?.federated || {}; }
};
