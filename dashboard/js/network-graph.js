/**
 * network-graph.js
 * Uses D3.js to render a force-directed graph of the IoT botnet.
 */

const NetworkGraph = {
  svg: null,
  simulation: null,
  nodesData: [],
  linksData: [],
  width: 0,
  height: 0,
  isSimulating: false,
  simInterval: null,
  
  init() {
    const wrap = document.getElementById('graph-canvas-wrap');
    if (!wrap) return;
    
    this.width = wrap.clientWidth;
    this.height = wrap.clientHeight;
    
    this.svg = d3.select('#graph-svg')
      .attr('viewBox', [0, 0, this.width, this.height]);
      
    // Set up zoom
    this.g = this.svg.append('g');
    this.svg.call(d3.zoom()
      .extent([[0, 0], [this.width, this.height]])
      .scaleExtent([0.1, 4])
      .on('zoom', (e) => this.g.attr('transform', e.transform)));
      
    this.generateData();
    this.render();
    
    // Listen for block events
    window.addEventListener('node-blocked', (e) => this.updateNodeState(e.detail.nodeId, true));
    window.addEventListener('node-unblocked', (e) => this.updateNodeState(e.detail.nodeId, false));
    
    // Simulate traffic button
    const simBtn = document.getElementById('btn-sim-traffic');
    if (simBtn) {
      simBtn.addEventListener('click', () => {
        this.isSimulating = !this.isSimulating;
        if (this.isSimulating) {
          simBtn.innerHTML = '<span class="live-dot" style="margin-right:4px;"></span> Stop Simulating';
          simBtn.style.background = 'var(--red-dim)';
          simBtn.style.color = 'var(--red)';
          simBtn.style.borderColor = 'var(--red)';
          this.startSimulation();
        } else {
          simBtn.innerHTML = '<span class="live-dot" style="margin-right:4px;"></span> Simulate Traffic';
          simBtn.style.background = 'var(--accent)';
          simBtn.style.color = '#0a0c12';
          simBtn.style.borderColor = 'var(--accent)';
          this.stopSimulation();
        }
      });
    }
    
    // Reset graph button
    const resetBtn = document.getElementById('btn-reset-graph');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        this.generateData();
        this.render();
      });
    }
  },
  
  generateData() {
    // 50 nodes based on actual distribution
    const N = 50;
    this.nodesData = [];
    this.linksData = [];
    
    const types = [
      { t: 'benign', color: 'var(--cyan)', p: 0.42 },
      { t: 'ddos_icmp', color: 'var(--red)', p: 0.15 },
      { t: 'ddos_syn', color: 'var(--accent)', p: 0.09 },
      { t: 'mirai', color: 'var(--purple)', p: 0.05 },
      { t: 'other', color: 'var(--orange)', p: 0.29 }
    ];
    
    for (let i = 0; i < N; i++) {
      let r = Math.random(), acc = 0, type = 'benign', color = 'var(--cyan)';
      for (let t of types) {
        acc += t.p;
        if (r <= acc) { type = t.t; color = t.color; break; }
      }
      this.nodesData.push({
        id: i,
        type: type,
        color: color,
        baseColor: color,
        val: type === 'benign' ? Math.random() * 5 + 2 : Math.random() * 15 + 8,
        blocked: false
      });
    }
    
    // Add links (mostly within same type, some cross)
    for (let i = 0; i < N; i++) {
      let numLinks = Math.floor(Math.random() * 3) + 1;
      if (this.nodesData[i].type !== 'benign') numLinks += 2; // attackers have more links
      
      for (let l = 0; l < numLinks; l++) {
        let target = Math.floor(Math.random() * N);
        if (target !== i) {
          // Check if link exists
          if (!this.linksData.some(link => (link.source === i && link.target === target) || (link.source === target && link.target === i))) {
            this.linksData.push({
              source: i,
              target: target,
              value: Math.random() * 4 + 1
            });
          }
        }
      }
    }
    
    // Count links for UI
    this.nodesData.forEach(n => {
      n.links = this.linksData.filter(l => l.source === n.id || l.target === n.id).length;
    });
  },
  
  render() {
    this.g.selectAll('*').remove();
    
    if (this.simulation) this.simulation.stop();
    
    this.simulation = d3.forceSimulation(this.nodesData)
      .force('link', d3.forceLink(this.linksData).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collide', d3.forceCollide().radius(d => d.val + 5));
      
    this.link = this.g.append('g')
      .selectAll('line')
      .data(this.linksData)
      .join('line')
      .attr('stroke', 'var(--border-light)')
      .attr('stroke-width', d => Math.sqrt(d.value))
      .attr('stroke-opacity', 0.6);
      
    this.node = this.g.append('g')
      .selectAll('circle')
      .data(this.nodesData)
      .join('circle')
      .attr('r', d => d.val)
      .attr('fill', d => d.color)
      .attr('stroke', '#0a0c12')
      .attr('stroke-width', 1.5)
      .attr('id', d => `node-${d.id}`)
      .attr('style', d => {
        if (d.type === 'ddos_icmp') return 'animation: malicious-pulse 2s infinite;';
        if (d.type === 'mirai') return 'animation: mirai-pulse 3s infinite;';
        return '';
      })
      .call(this.drag(this.simulation))
      .on('click', (e, d) => {
        if (window.NodeControl) window.NodeControl.selectNode(d);
        // Highlight logic
        this.node.attr('stroke', '#0a0c12').attr('stroke-width', 1.5);
        d3.select(e.currentTarget).attr('stroke', '#fff').attr('stroke-width', 3);
      })
      .on('mouseover', (e, d) => this.showTooltip(e, d))
      .on('mouseout', () => this.hideTooltip());
      
    this.simulation.on('tick', () => {
      this.link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
        
      this.node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
    });
  },
  
  drag(simulation) {
    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }
    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }
    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
    return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  },
  
  updateNodeState(nodeId, isBlocked) {
    const node = this.nodesData.find(n => n.id === nodeId);
    if (!node) return;
    
    node.blocked = isBlocked;
    node.color = isBlocked ? 'var(--grey)' : node.baseColor;
    
    d3.select(`#node-${nodeId}`)
      .transition().duration(300)
      .attr('fill', node.color)
      .style('animation', isBlocked ? 'none' : null);
      
    // Fade links connected to blocked node
    this.link.transition().duration(300).attr('stroke-opacity', d => {
      if (d.source.blocked || d.target.blocked) return 0.1;
      return 0.6;
    });
  },
  
  startSimulation() {
    this.simInterval = setInterval(() => {
      // Pick random non-blocked benign node to turn malicious
      const avail = this.nodesData.filter(n => !n.blocked && n.type === 'benign');
      if (avail.length > 0) {
        const n = avail[Math.floor(Math.random() * avail.length)];
        n.type = 'ddos_icmp';
        n.baseColor = 'var(--red)';
        n.color = 'var(--red)';
        n.val += 5; // grows in traffic volume
        
        d3.select(`#node-${n.id}`)
          .transition().duration(500)
          .attr('fill', n.color)
          .attr('r', n.val)
          .style('animation', 'malicious-pulse 2s infinite');
          
        // Re-simulate briefly to adjust for size
        this.simulation.alpha(0.3).restart();
      }
    }, 2000);
  },
  
  stopSimulation() {
    if (this.simInterval) clearInterval(this.simInterval);
  },
  
  showTooltip(e, d) {
    const tooltip = document.getElementById('node-tooltip');
    if (!tooltip) return;
    
    tooltip.innerHTML = `
      <div style="font-weight:700; margin-bottom:4px;">Node #${d.id}</div>
      <div style="font-size:11px; color:var(--text-sub);">Type: <span style="color:${d.baseColor}">${d.type.toUpperCase()}</span></div>
      <div style="font-size:11px; color:var(--text-sub);">Volume: ${Math.round(d.val*100)} kbps</div>
      ${d.blocked ? '<div style="font-size:10px; color:var(--red); margin-top:4px; font-weight:700;">BLOCKED</div>' : ''}
    `;
    
    // Position
    const wrapRect = document.getElementById('graph-canvas-wrap').getBoundingClientRect();
    const x = e.clientX - wrapRect.left + 15;
    const y = e.clientY - wrapRect.top + 15;
    
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.classList.add('show');
  },
  
  hideTooltip() {
    const tooltip = document.getElementById('node-tooltip');
    if (tooltip) tooltip.classList.remove('show');
  }
};
