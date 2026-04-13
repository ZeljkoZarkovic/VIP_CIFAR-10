const MODEL_DATA = [
  {
    model_name: "Model_1_Baseline",
    label: "Baseline",
    config: { num_conv_blocks: 3, filters: [32, 64, 128], kernel_size: 3, dense_units: 256, dropout_rate: 0.5, learning_rate: 0.001 },
    fold_results: [
      { fold: 1, val_accuracy: 0.7220, val_loss: 0.8493, train_accuracy: 0.9346, train_loss: 0.1966 },
      { fold: 2, val_accuracy: 0.7275, val_loss: 0.8923, train_accuracy: 0.9564, train_loss: 0.1319 },
      { fold: 3, val_accuracy: 0.7325, val_loss: 0.7951, train_accuracy: 0.9645, train_loss: 0.1121 },
      { fold: 4, val_accuracy: 0.7295, val_loss: 0.8607, train_accuracy: 0.9640, train_loss: 0.1123 },
      { fold: 5, val_accuracy: 0.7310, val_loss: 0.8798, train_accuracy: 0.9590, train_loss: 0.1278 },
    ],
    avg_val_accuracy: 0.7285, avg_val_loss: 0.8554,
    avg_train_time: 577.98, avg_inference_time: 1.777,
    model_size_params: 816938, model_memory_mb: 3.12,
  },
  {
    model_name: "Model_2_Deep",
    label: "Deep",
    config: { num_conv_blocks: 4, filters: [32, 64, 128, 256], kernel_size: 3, dense_units: 512, dropout_rate: 0.5, learning_rate: 0.001 },
    fold_results: [
      { fold: 1, val_accuracy: 0.7040, val_loss: 0.8925, train_accuracy: 0.9712, train_loss: 0.0865 },
      { fold: 2, val_accuracy: 0.7465, val_loss: 1.0974, train_accuracy: 0.9826, train_loss: 0.0525 },
      { fold: 3, val_accuracy: 0.6865, val_loss: 0.9545, train_accuracy: 0.9595, train_loss: 0.1194 },
      { fold: 4, val_accuracy: 0.7300, val_loss: 0.9520, train_accuracy: 0.9865, train_loss: 0.0458 },
      { fold: 5, val_accuracy: 0.7330, val_loss: 1.2105, train_accuracy: 0.9879, train_loss: 0.0429 },
    ],
    avg_val_accuracy: 0.7200, avg_val_loss: 1.0214,
    avg_train_time: 678.27, avg_inference_time: 2.185,
    model_size_params: 1708074, model_memory_mb: 6.52,
  },
  {
    model_name: "Model_3_Wide",
    label: "Wide",
    config: { num_conv_blocks: 3, filters: [64, 128, 256], kernel_size: 3, dense_units: 512, dropout_rate: 0.5, learning_rate: 0.001 },
    fold_results: [
      { fold: 1, val_accuracy: 0.7065, val_loss: 0.9300, train_accuracy: 0.9841, train_loss: 0.0501 },
      { fold: 2, val_accuracy: 0.7165, val_loss: 0.9149, train_accuracy: 0.9926, train_loss: 0.0293 },
      { fold: 3, val_accuracy: 0.7025, val_loss: 0.9624, train_accuracy: 0.9908, train_loss: 0.0361 },
      { fold: 4, val_accuracy: 0.6845, val_loss: 1.0137, train_accuracy: 0.9923, train_loss: 0.0361 },
      { fold: 5, val_accuracy: 0.7465, val_loss: 1.0259, train_accuracy: 0.9961, train_loss: 0.0141 },
    ],
    avg_val_accuracy: 0.7113, avg_val_loss: 0.9694,
    avg_train_time: 1015.58, avg_inference_time: 3.053,
    model_size_params: 3253834, model_memory_mb: 12.41,
  },
  {
    model_name: "Model_4_Small",
    label: "Small",
    config: { num_conv_blocks: 2, filters: [32, 64], kernel_size: 3, dense_units: 128, dropout_rate: 0.3, learning_rate: 0.001 },
    fold_results: [
      { fold: 1, val_accuracy: 0.6615, val_loss: 1.0480, train_accuracy: 0.9749, train_loss: 0.0993 },
      { fold: 2, val_accuracy: 0.6705, val_loss: 1.1290, train_accuracy: 0.9884, train_loss: 0.0510 },
      { fold: 3, val_accuracy: 0.6425, val_loss: 1.0413, train_accuracy: 0.9669, train_loss: 0.1165 },
      { fold: 4, val_accuracy: 0.6395, val_loss: 1.0696, train_accuracy: 0.9672, train_loss: 0.1194 },
      { fold: 5, val_accuracy: 0.6315, val_loss: 1.1684, train_accuracy: 0.9801, train_loss: 0.0783 },
    ],
    avg_val_accuracy: 0.6491, avg_val_loss: 1.0913,
    avg_train_time: 306.55, avg_inference_time: 1.301,
    model_size_params: 592554, model_memory_mb: 2.26,
  },
  {
    model_name: "Model_5_LargeKernel",
    label: "Large Kernel",
    config: { num_conv_blocks: 3, filters: [32, 64, 128], kernel_size: 5, dense_units: 256, dropout_rate: 0.5, learning_rate: 0.0005 },
    fold_results: [
      { fold: 1, val_accuracy: 0.7070, val_loss: 0.9227, train_accuracy: 0.9749, train_loss: 0.0935 },
      { fold: 2, val_accuracy: 0.6760, val_loss: 1.0505, train_accuracy: 0.9806, train_loss: 0.0739 },
      { fold: 3, val_accuracy: 0.7050, val_loss: 0.9627, train_accuracy: 0.9886, train_loss: 0.0502 },
      { fold: 4, val_accuracy: 0.6970, val_loss: 0.9920, train_accuracy: 0.9850, train_loss: 0.0568 },
      { fold: 5, val_accuracy: 0.6755, val_loss: 1.0274, train_accuracy: 0.9751, train_loss: 0.0938 },
    ],
    avg_val_accuracy: 0.6921, avg_val_loss: 0.9911,
    avg_train_time: 1063.79, avg_inference_time: 2.607,
    model_size_params: 1326378, model_memory_mb: 5.06,
  },
];

const CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];
const CLASS_ICONS = { airplane:'✈️', automobile:'🚗', bird:'🐦', cat:'🐱', deer:'🦌', dog:'🐶', frog:'🐸', horse:'🐴', ship:'🚢', truck:'🚛' };

//NAVIGATION
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = link.dataset.section;
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    link.classList.add('active');
    document.getElementById(target).classList.add('active');
  });
});

//CLASSIFIER
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImg  = document.getElementById('previewImg');
const clearBtn    = document.getElementById('clearBtn');
const classifyBtn = document.getElementById('classifyBtn');
const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const resultsContent     = document.getElementById('resultsContent');

let currentFile = null;

//Drag/drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  if (file.size > 5 * 1024 * 1024) { alert('Fajl je prevelik (max 5MB)'); return; }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropZone.style.display = 'none';
    previewArea.style.display = 'block';
    classifyBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
  currentFile = null;
  previewImg.src = '';
  fileInput.value = '';
  dropZone.style.display = 'block';
  previewArea.style.display = 'none';
  classifyBtn.disabled = true;
  resultsPlaceholder.style.display = 'flex';
  resultsContent.style.display = 'none';
  document.querySelectorAll('.class-chip').forEach(c => c.classList.remove('highlight'));
});

classifyBtn.addEventListener('click', () => {
  if (!currentFile) return;
  const btnText    = classifyBtn.querySelector('.btn-text');
  const btnSpinner = classifyBtn.querySelector('.btn-spinner');
  btnText.style.display = 'none';
  btnSpinner.style.display = 'inline';
  classifyBtn.disabled = true;

  const formData = new FormData();
  formData.append('image', currentFile);
  formData.append('model', document.getElementById('modelSelect').value);

  fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) throw new Error(data.error);
      showResults(data.predictions);
    })
    .catch(err => {
      alert('Greška: ' + err.message);
    })
    .finally(() => {
      btnText.style.display = 'inline';
      btnSpinner.style.display = 'none';
      classifyBtn.disabled = false;
    });
});


function showResults(predictions) {
  const top = predictions[0];
  document.getElementById('predIcon').textContent = CLASS_ICONS[top.class] || '🔍';
  document.getElementById('predLabel').textContent = top.class.charAt(0).toUpperCase() + top.class.slice(1);
  document.getElementById('predConfidence').textContent = `Pouzdanost: ${(top.confidence * 100).toFixed(1)}%`;

  const barsEl = document.getElementById('confidenceBars');
  barsEl.innerHTML = '';
  predictions.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'conf-bar-row';
    row.innerHTML = `
      <span class="conf-bar-label">${p.class}</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill ${i === 0 ? 'top' : ''}" style="width:0%"></div>
      </div>
      <span class="conf-bar-value">${(p.confidence * 100).toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
    
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector('.conf-bar-fill').style.width = (p.confidence * 100) + '%';
      });
    });
  });

    // Highlight class chip
  document.querySelectorAll('.class-chip').forEach(c => {
    c.classList.toggle('highlight', c.dataset.class === top.class);
  });

  resultsPlaceholder.style.display = 'none';
  resultsContent.style.display = 'block';
}