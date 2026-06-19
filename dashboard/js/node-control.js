/**
 * node-control.js
 * Manages the blocked nodes state and UI for the network graph section.
 */

const NodeControl = {
  blockedNodes: new Set(),
  
  init() {
    this.logEl = document.getElementById('blocked-log');
    this.blockBtn = document.getElementById('btn-block-node');
    this.unblockBtn = document.getElementById('btn-unblock-node');
    this.countEl = document.getElementById('gc-blocked');
    this.selectedNodeId = null;
    
    if (this.blockBtn) {
      this.blockBtn.addEventListener('click', () => {
        if (this.selectedNodeId !== null) this.blockNode(this.selectedNodeId);
      });
    }
    
    if (this.unblockBtn) {
      this.unblockBtn.addEventListener('click', () => {
        if (this.selectedNodeId !== null) this.unblockNode(this.selectedNodeId);
      });
    }
  },
  
  selectNode(node) {
    this.selectedNodeId = node.id;
    const panel = document.getElementById('selected-node-panel');
    if (!panel) return;
    
    panel.style.display = 'block';
    document.getElementById('sn-id').textContent = `Node #${node.id}`;
    document.getElementById('sn-links').textContent = node.links || 0;
    document.getElementById('sn-vol').textContent = Math.round(node.vol || node.val * 100);
    
    const typeLabel = document.getElementById('sn-type-label');
    if (node.type === 'benign') {
      typeLabel.textContent = 'Benign Traffic';
      typeLabel.style.color = 'var(--cyan)';
    } else {
      typeLabel.textContent = `Malicious: ${node.type.toUpperCase()}`;
      typeLabel.style.color = 'var(--red)';
    }
    
    this.updateButtons();
  },
  
  updateButtons() {
    if (this.selectedNodeId === null) return;
    const isBlocked = this.blockedNodes.has(this.selectedNodeId);
    
    if (isBlocked) {
      this.blockBtn.style.display = 'none';
      this.unblockBtn.style.display = 'block';
    } else {
      this.blockBtn.style.display = 'block';
      this.unblockBtn.style.display = 'none';
    }
  },
  
  blockNode(nodeId) {
    if (this.blockedNodes.has(nodeId)) return;
    
    this.blockedNodes.add(nodeId);
    this.updateCount();
    this.updateButtons();
    this.logAction(nodeId, 'Blocked');
    
    // Dispatch event for graph to update
    window.dispatchEvent(new CustomEvent('node-blocked', { detail: { nodeId } }));
  },
  
  unblockNode(nodeId) {
    if (!this.blockedNodes.has(nodeId)) return;
    
    this.blockedNodes.delete(nodeId);
    this.updateCount();
    this.updateButtons();
    this.logAction(nodeId, 'Unblocked');
    
    // Dispatch event for graph to update
    window.dispatchEvent(new CustomEvent('node-unblocked', { detail: { nodeId } }));
  },
  
  updateCount() {
    if (this.countEl) this.countEl.textContent = this.blockedNodes.size;
  },
  
  logAction(nodeId, action) {
    if (!this.logEl) return;
    
    if (this.blockedNodes.size === 1 && action === 'Blocked') {
      // Clear empty message
      this.logEl.innerHTML = '';
    } else if (this.blockedNodes.size === 0) {
      this.logEl.innerHTML = '<div style="font-size:11px; color:var(--text-muted); text-align:center; padding:20px 0;">No nodes blocked yet.</div>';
      return;
    }
    
    if (action === 'Unblocked') {
      const el = document.getElementById(`log-item-${nodeId}`);
      if (el) el.remove();
      return;
    }
    
    const time = new Date().toLocaleTimeString([], {hour12:false});
    const item = document.createElement('div');
    item.className = 'blocked-item';
    item.id = `log-item-${nodeId}`;
    item.innerHTML = `
      <div style="width:8px; height:8px; border-radius:50%; background:var(--red);"></div>
      <div style="flex:1;">Node #${nodeId}</div>
      <div class="block-time">${time}</div>
    `;
    
    this.logEl.prepend(item);
  }
};
