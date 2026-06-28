(() => {
  'use strict';

  const $ = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

  const state = {
    models: [],
    activeModel: null,
    loading: false,
    generating: false,
    config: {},
    gpu: null,
    lastResult: null,
    hasImage: false,
    progressTimer: null,
  };

  const el = {
    sidebar:        $('#sidebar'),
    modelList:      $('#model-list'),
    statusGpu:      $('#status-gpu'),
    statusVram:     $('#status-vram'),
    statusModel:    $('#status-model'),
    statusCache:    $('#status-cache'),
    outputPanel:    $('#output-panel'),
    outputPlaceholder: $('#output-placeholder'),
    outputResult:   $('#output-result'),
    resultImage:    $('#result-image'),
    resultLabel:    $('#result-label'),
    resultMeta:     $('#result-meta'),
    genLoading:     $('#gen-loading'),
    progressWrapper: $('#progress-wrapper'),
    progressFill:   $('#progress-fill'),
    progressText:   $('#progress-text'),
    promptForm:     $('#prompt-form'),
    promptInput:    $('#prompt-input'),
    negativePrompt: $('#negative-prompt'),
    btnSend:        $('#btn-send'),
    btnToggleAdvanced: $('#btn-toggle-advanced'),
    promptAdvanced: $('#prompt-advanced'),
    configBadge:    $('#config-badge'),
    sizeBadge:      $('#size-badge'),
    paramSteps:     $('#param-steps'),
    paramGuidance:  $('#param-guidance'),
    paramSeed:      $('#param-seed'),
    paramWidth:     $('#param-width'),
    paramHeight:    $('#param-height'),
    toastContainer: $('#toast-container'),
    btnFolder:      $('#btn-folder'),
    btnClear:       $('#btn-clear'),
    cacheInput:     $('#cache-input'),
    btnSetupSave:   $('#btn-setup-save'),
    nf4Toggle:      $('#nf4-toggle'),
    nf4Info:        $('#nf4-info'),
    btnNf4Info:     $('#btn-nf4-info'),
  };

  // ---- Sidebar resize ----

  let resizeDragging = false;
  let resizeStartX = 0;
  let resizeStartW = 0;

  function initSidebarResize() {
    const handle = document.createElement('div');
    handle.className = 'sidebar-resize-handle';
    el.sidebar.appendChild(handle);
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      resizeDragging = true;
      resizeStartX = e.clientX;
      resizeStartW = el.sidebar.offsetWidth;
      document.body.style.cursor = 'col-resize';
      handle.classList.add('dragging');
    });
    document.addEventListener('mousemove', (e) => {
      if (!resizeDragging) return;
      const delta = e.clientX - resizeStartX;
      const newW = Math.max(280, resizeStartW + delta);
      el.sidebar.style.width = newW + 'px';
      el.sidebar.style.minWidth = newW + 'px';
    });
    document.addEventListener('mouseup', () => {
      if (!resizeDragging) return;
      resizeDragging = false;
      document.body.style.cursor = '';
      handle.classList.remove('dragging');
    });
  }

  // ---- Init ----

  async function init() {
    initSidebarResize();
    const grid = document.createElement('div');
    grid.className = 'gen-grid';
    for (let i = 0; i < 36; i++) {
      const dot = document.createElement('i');
      grid.appendChild(dot);
    }
    el.genLoading.appendChild(grid);
    setupTextarea();
    setupEventListeners();
    await fetchStatus();
    await fetchModels();
    autoResizeTextarea();
  }

  // ---- API helpers ----

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  async function apiSaveConfig(key, value) {
    return api('/api/config', {
      method: 'POST',
      body: JSON.stringify({ key, value: JSON.stringify(value) }),
    });
  }

  // ---- Data fetching ----

  async function fetchStatus() {
    try {
      const data = await api('/api/status');
      state.config = data.config || {};
      state.gpu = data.gpu;
      el.statusGpu.textContent = data.gpu ? data.gpu.name : 'Not detected';
      el.statusGpu.title = data.gpu ? data.gpu.name : '';
      el.statusVram.textContent = data.gpu ? `${data.gpu.vram_gb} GB` : '--';
      el.statusCache.textContent = data.cache_dir || '--';
      if (data.cache_dir) el.cacheInput.value = data.cache_dir;

      if (data.active_model) {
        state.activeModel = data.active_model.key;
        el.statusModel.textContent = data.active_model.name;
        highlightActiveModel();
      } else {
        el.statusModel.textContent = 'None loaded';
      }

      if (state.config.use_nf4 !== undefined) {
        el.nf4Toggle.checked = state.config.use_nf4;
      }

      updateConfigBadges();
      updateSendButton();
    } catch (err) {
      console.error('Status fetch failed:', err);
      el.statusGpu.textContent = 'Error';
    }
  }

  async function fetchModels() {
    try {
      const data = await api('/api/models');
      state.models = data.models;
      renderModelList();
    } catch (err) {
      console.error('Models fetch failed:', err);
      showToast('Could not load model list', 'error');
    }
  }

  // ---- Parse description ----

  function parseModelMeta(desc) {
    const parts = desc.split('·').map(s => s.trim());
    return { params: parts[0] || '', speed: parts[1] || '' };
  }

  // ---- Model rendering ----

  function renderModelList() {
    el.modelList.innerHTML = state.models.map(m => {
      const meta = parseModelMeta(m.description);
      const isActive = state.activeModel === m.key;
      const isOtherLoading = state.loading && !isActive;
      return `
        <div class="model-item ${isActive ? 'active' : ''} ${isOtherLoading ? 'loading-state' : ''}" data-model="${m.key}">
          <div class="model-item-row">
            <span class="model-item-name">${esc(m.name)}</span>
            <span class="model-item-status">
              <span class="model-spinner u-hidden"></span>
              <span class="model-check u-hidden">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              </span>
            </span>
          </div>
          <div class="model-item-meta">
            <span class="model-meta-badge">${esc(meta.params)}</span>
            <span class="model-meta-badge">${esc(meta.speed)}</span>
          </div>
        </div>
      `;
    }).join('');

    $$('.model-item', el.modelList).forEach(item => {
      item.addEventListener('click', () => {
        const key = item.dataset.model;
        if (key && key !== state.activeModel && !state.loading) {
          loadModelAction(key);
        }
      });
    });
  }

  function highlightActiveModel() {
    $$('.model-item', el.modelList).forEach(item => {
      item.classList.toggle('active', item.dataset.model === state.activeModel);
    });
  }

  // ---- Model actions ----

  async function loadModelAction(key) {
    if (state.loading) return;
    const model = state.models.find(m => m.key === key);
    if (!model) return;

    state.loading = true;
    stopProgressPolling();
    setModelListLoading(key);
    setUIEnabled(false);

    try {
      await api('/api/load-model', { method: 'POST', body: JSON.stringify({ model_key: key }) });
      state.activeModel = key;
      state.hasImage = false;
      el.statusModel.textContent = model.name;
      el.outputResult.classList.add('u-hidden');
      el.outputPlaceholder.classList.remove('u-hidden');
      highlightActiveModel();
      updateConfigBadges();
      populateParams(model);
      updateSendButton();
      setModelLoaded(key);
      showToast(`${model.name} loaded`, 'success');
    } catch (err) {
      setModelLoadFailed(key);
      showToast(err.message, 'error');
      console.error('Model load failed:', err);
    } finally {
      state.loading = false;
      setUIEnabled(true);
    }
  }

  function setModelListLoading(loadingKey) {
    $$('.model-item', el.modelList).forEach(item => {
      const key = item.dataset.model;
      if (key === loadingKey) {
        item.classList.add('model-loading');
        const spinner = item.querySelector('.model-spinner');
        const check = item.querySelector('.model-check');
        if (spinner) spinner.classList.remove('u-hidden');
        if (check) check.classList.add('u-hidden');
        item.classList.remove('active');
      } else {
        item.classList.add('loading-state');
      }
      const spinner = item.querySelector('.model-spinner');
      const check = item.querySelector('.model-check');
      if (key !== loadingKey && spinner) spinner.classList.add('u-hidden');
      if (key !== loadingKey && check) check.classList.add('u-hidden');
    });
  }

  function setModelLoaded(key) {
    $$('.model-item', el.modelList).forEach(item => {
      item.classList.remove('loading-state', 'model-loading');
      const itemKey = item.dataset.model;
      const spinner = item.querySelector('.model-spinner');
      const check = item.querySelector('.model-check');
      if (spinner) spinner.classList.add('u-hidden');
      if (check) check.classList.add('u-hidden');
      if (itemKey === key && check) check.classList.remove('u-hidden');
    });
    highlightActiveModel();
  }

  function setModelLoadFailed(key) {
    $$('.model-item', el.modelList).forEach(item => {
      item.classList.remove('loading-state', 'model-loading');
      const spinner = item.querySelector('.model-spinner');
      const check = item.querySelector('.model-check');
      if (spinner) spinner.classList.add('u-hidden');
      if (check) check.classList.add('u-hidden');
    });
  }

  function setUIEnabled(enabled) {
    el.promptInput.disabled = !enabled;
  }

  function populateParams(model) {
    if (model) {
      el.paramSteps.value = model.default_steps;
      el.paramGuidance.value = model.default_guidance;
      el.paramWidth.value = model.default_width;
      el.paramHeight.value = model.default_height;
      el.sizeBadge.textContent = `${model.default_width}x${model.default_height}`;
    }
  }

  function updateConfigBadges() {
    el.configBadge.textContent = state.activeModel || 'none';
  }

  function updateSendButton() {
    const hasModel = !!state.activeModel;
    const hasPrompt = el.promptInput.value.trim().length > 0;
    el.btnSend.disabled = !hasModel || !hasPrompt || state.generating || state.loading;
  }

  // ---- Generation ----

  async function handleGenerate(e) {
    e.preventDefault();
    if (state.generating || state.loading) return;
    if (!state.activeModel) {
      showToast('Select a model first', 'warning');
      return;
    }
    const prompt = el.promptInput.value.trim();
    if (!prompt) return;

    state.generating = true;
    setGeneratingUI(true);

    const negativePrompt = el.negativePrompt.value.trim();
    const seedVal = parseInt(el.paramSeed.value) || -1;
    const widthVal = parseInt(el.paramWidth.value) || 1024;
    const heightVal = parseInt(el.paramHeight.value) || 1024;
    const stepsVal = parseInt(el.paramSteps.value) || 4;
    const guidanceVal = parseFloat(el.paramGuidance.value) || 0;

    startProgressPolling(stepsVal);

    try {
      const result = await api('/api/generate', {
        method: 'POST',
        body: JSON.stringify({
          prompt,
          negative_prompt: negativePrompt,
          seed: seedVal,
          width: widthVal,
          height: heightVal,
          steps: stepsVal,
          guidance: guidanceVal,
        }),
      });
      state.lastResult = result;
      state.hasImage = true;
      displayResult(result);
      showToast(`Generated in ${result.time}s`, 'success');
    } catch (err) {
      showToast(err.message || 'Generation failed', 'error');
      console.error('Generation failed:', err);
    } finally {
      stopProgressPolling();
      state.generating = false;
      setGeneratingUI(false);
    }
  }

  function setGeneratingUI(active) {
    if (active) {
      el.btnSend.classList.add('generating');
      el.btnSend.disabled = true;
      el.promptInput.disabled = true;

      if (!state.hasImage) {
        el.outputPlaceholder.classList.add('u-hidden');
      }

      el.genLoading.classList.remove('u-hidden');
      if (state.hasImage) {
        el.resultImage.style.opacity = '0.08';
      }
      el.progressWrapper.classList.remove('u-hidden');
    } else {
      el.btnSend.classList.remove('generating');
      el.promptInput.disabled = false;
      el.genLoading.classList.add('u-hidden');
      if (state.hasImage) {
        el.resultImage.style.opacity = '1';
      }
      el.progressWrapper.classList.add('u-hidden');
      el.progressFill.style.width = '0%';
      updateSendButton();
    }
  }

  function startProgressPolling(totalSteps) {
    stopProgressPolling();
    state.progressTimer = setInterval(async () => {
      try {
        const p = await api('/api/progress');
        if (p.active && p.total > 0) {
          el.progressFill.style.width = Math.round((p.step / p.total) * 100) + '%';
          el.progressText.textContent = `Step ${p.step} / ${p.total}`;
        }
      } catch (_) {}
    }, 1500);
  }

  function stopProgressPolling() {
    if (state.progressTimer) {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
    }
  }

  function displayResult(result) {
    const imgUrl = `/api/image/${result.filename}`;
    el.resultImage.src = imgUrl;
    el.resultImage.alt = result.filename;
    el.resultLabel.textContent = result.filename;
    el.resultMeta.innerHTML =
      `<span>${result.width}x${result.height}</span>` +
      `<span>${result.time}s</span>` +
      `<span>seed ${result.seed}</span>`;
    el.outputPlaceholder.classList.add('u-hidden');
    el.outputResult.classList.remove('u-hidden');
    el.sizeBadge.textContent = `${result.width}x${result.height}`;
  }

  async function openFolder() {
    if (!state.lastResult) return;
    try {
      await api('/api/open-folder', {
        method: 'POST',
        body: JSON.stringify({ filename: state.lastResult.filename }),
      });
    } catch (err) {
      showToast('Could not open folder', 'error');
    }
  }

  // ---- Textarea ----

  function setupTextarea() {
    el.promptInput.addEventListener('input', () => {
      autoResizeTextarea();
      updateSendButton();
    });
    el.promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        el.promptForm.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });
  }

  function autoResizeTextarea() {
    const ta = el.promptInput;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
  }

  // ---- Event listeners ----

  function setupEventListeners() {
    el.promptForm.addEventListener('submit', handleGenerate);
    el.btnFolder.addEventListener('click', openFolder);
    el.btnClear.addEventListener('click', () => {
      state.hasImage = false;
      state.lastResult = null;
      el.resultImage.src = '';
      el.outputResult.classList.add('u-hidden');
      el.outputPlaceholder.classList.remove('u-hidden');
    });

    el.btnToggleAdvanced.addEventListener('click', () => {
      const hidden = el.promptAdvanced.classList.contains('u-hidden');
      el.promptAdvanced.classList.toggle('u-hidden');
      el.btnToggleAdvanced.querySelector('span').textContent = hidden ? 'Hide' : 'Advanced';
    });

    el.paramSeed.addEventListener('focus', () => {
      if (el.paramSeed.value === '-1') el.paramSeed.value = '';
    });
    el.paramSeed.addEventListener('blur', () => {
      if (el.paramSeed.value === '') el.paramSeed.value = '-1';
    });

    el.paramWidth.addEventListener('change', () => {
      const w = el.paramWidth.value;
      const h = el.paramHeight.value;
      el.sizeBadge.textContent = `${w}x${h}`;
    });
    el.paramHeight.addEventListener('change', () => {
      const w = el.paramWidth.value;
      const h = el.paramHeight.value;
      el.sizeBadge.textContent = `${w}x${h}`;
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        el.promptAdvanced.classList.add('u-hidden');
        el.btnToggleAdvanced.querySelector('span').textContent = 'Advanced';
      }
    });

    el.btnSetupSave.addEventListener('click', async () => {
      const dir = el.cacheInput.value.trim();
      if (!dir) return;
      try {
        await apiSaveConfig('cache_dir', dir);
        showToast('Cache directory saved. Restart server to apply.', 'success');
      } catch (err) {
        showToast('Failed to save cache directory', 'error');
      }
    });

    el.nf4Toggle.addEventListener('change', async () => {
      const val = el.nf4Toggle.checked;
      try {
        await apiSaveConfig('use_nf4', val);
        showToast(val ? 'NF4 enabled' : 'NF4 disabled', 'info');
      } catch (err) {
        showToast('Failed to save NF4 setting', 'error');
        el.nf4Toggle.checked = !val;
      }
    });

    el.btnNf4Info.addEventListener('click', () => {
      el.nf4Info.classList.toggle('u-hidden');
    });
  }

  // ---- Toast ----

  function showToast(message, type = 'info') {
    const icons = {
      success: '<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
      error: '<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
      warning: '<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
      info: '<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `${icons[type] || icons.info}<span>${esc(message)}</span>`;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('leaving');
      toast.addEventListener('animationend', () => toast.remove());
    }, 3500);
  }

  function esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  document.addEventListener('DOMContentLoaded', init);
})();
