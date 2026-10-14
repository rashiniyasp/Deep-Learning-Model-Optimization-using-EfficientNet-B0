/**
 * HTIC IIT Madras - Fetal Ultrasound Optimization Showcase
 * Interactive Gallery, Lightbox, Filter and Dynamic Interactions
 */

// Plot gallery data mapping all 17 empirical artifacts
const GALLERY_ITEMS = [
  {
    id: "pareto_frontier",
    title: "Latency vs Accuracy Pareto Frontier",
    category: "deployment",
    categoryLabel: "Deployment & Optimization",
    image: "assets/pareto_frontier_latency_f1.png",
    description: "Multi-objective Pareto frontier showing PyTorch FP32, PyTorch INT8, ONNX FP32, and ONNX INT8 trade-offs on AMD Ryzen 7 7730U CPU. Demonstrates optimal operational points."
  },
  {
    id: "deployment_opt",
    title: "Deployment Latency, FPS & Model Footprint",
    category: "deployment",
    categoryLabel: "Deployment & Optimization",
    image: "assets/deployment_optimization.png",
    description: "Quad-panel benchmark breakdown: Mean inference latency (ms), Real-time throughput (FPS), Checkpoint disk footprint (MB), and Test Accuracy retention across runtime engines."
  },
  {
    id: "calibration_scaling",
    title: "Temperature Scaling Probability Calibration",
    category: "calibration",
    categoryLabel: "Uncertainty Calibration",
    image: "assets/calibration_temperature_scaling.png",
    description: "Optimization trajectory of post-hoc Temperature Scaling (T* = 0.8765) using L-BFGS, showing before vs. after reliability diagrams and confidence histograms."
  },
  {
    id: "calibration_summary",
    title: "ECE Calibration Summary",
    category: "calibration",
    categoryLabel: "Uncertainty Calibration",
    image: "assets/calibration_summary.png",
    description: "Expected Calibration Error (ECE) reduction from 13.27% down to 9.46% (28.8% relative reduction), rectifying overconfidence in critical clinical predictions."
  },
  {
    id: "reliability_diag",
    title: "Pre-Calibration Reliability Diagram",
    category: "calibration",
    categoryLabel: "Uncertainty Calibration",
    image: "assets/reliability_diagram.png",
    description: "Baseline Expected Calibration Error (ECE = 13.27%) and Maximum Calibration Error (MCE) across 15 confidence bins on 5,271 test ultrasound images."
  },
  {
    id: "gradcam_gallery",
    title: "Grad-CAM Anatomical Landmark Explainability",
    category: "explainability",
    categoryLabel: "Explainability & Grad-CAM",
    image: "assets/gradcam_gallery.png",
    description: "Native PyTorch Grad-CAM saliency maps overlaid on original ultrasound frames across all 6 anatomical planes, verifying model attention on acoustic landmarks."
  },
  {
    id: "tsne_planes",
    title: "1,280-dim Penultimate Feature t-SNE Projections",
    category: "manifold",
    categoryLabel: "Latent Manifold Projections",
    image: "assets/tsne_anatomical_planes.png",
    description: "t-Distributed Stochastic Neighbor Embedding of penultimate 1,280-dim features for 5,271 test images. Demonstrates clean anatomical clustering and intra-cluster tightness."
  },
  {
    id: "umap_planes",
    title: "1,280-dim Penultimate Feature UMAP Projections",
    category: "manifold",
    categoryLabel: "Latent Manifold Projections",
    image: "assets/umap_anatomical_planes.png",
    description: "Uniform Manifold Approximation and Projection (UMAP) preserving global cluster topology and continuum geometry across maternal-fetal anatomical planes."
  },
  {
    id: "confusion_matrix",
    title: "Normalized Multi-Class Confusion Matrix",
    category: "training",
    categoryLabel: "Training & Convergence",
    image: "assets/confusion_matrix.png",
    description: "Full test set evaluation (5,271 images, 94.67% accuracy). Shows near-zero inter-plane confusion with slight mutual overlap in Brain trans-thalamic vs trans-ventricular scans."
  },
  {
    id: "per_class_metrics",
    title: "Per-Class Precision, Recall & F1-Score",
    category: "training",
    categoryLabel: "Training & Convergence",
    image: "assets/per_class_metrics.png",
    description: "Comprehensive classification breakdown across Fetal Abdomen, Fetal Brain, Fetal Femur, Fetal Thorax, Maternal Cervix, and Other planes."
  },
  {
    id: "per_group_accuracy",
    title: "Clinical Subgroup & Bias Auditing",
    category: "training",
    categoryLabel: "Training & Convergence",
    image: "assets/per_group_accuracy.png",
    description: "Auditing classification accuracy across clinical ultrasound device manufacturers and sonographer operators, ensuring invariance to acoustic artifacts."
  },
  {
    id: "training_curves",
    title: "AdamW Loss & Accuracy Convergence Curves",
    category: "training",
    categoryLabel: "Training & Convergence",
    image: "assets/training_curves.png",
    description: "Epoch-by-epoch cross-entropy loss with Label Smoothing (0.1) and top-1 accuracy curves. Demonstrates smooth convergence to optimal validation score at epoch 24."
  },
  {
    id: "sample_images",
    title: "Clinical Ultrasound Dataset Exemplars",
    category: "eda",
    categoryLabel: "Exploratory Data Analysis",
    image: "assets/sample_images.png",
    description: "High-resolution ultrasound scans illustrating fetal biometric planes (trans-thalamic brain, femur length, abdominal circumference, thoracic view, cervix, and other)."
  },
  {
    id: "class_distribution",
    title: "Class Imbalance Distribution (Zenodo 3904280)",
    category: "eda",
    categoryLabel: "Exploratory Data Analysis",
    image: "assets/class_distribution.png",
    description: "Pre-training frequency distribution across all 12,400 ultrasound images, revealing significant class imbalance (Fetal Brain 3,092 vs Other 6,538 vs Cervix 1,626)."
  },
  {
    id: "metadata_distribution",
    title: "Acquisition Equipment & Operator Breakdown",
    category: "eda",
    categoryLabel: "Exploratory Data Analysis",
    image: "assets/metadata_distribution.png",
    description: "Breakdown of ultrasound scanner hardware models and clinical sonographer experience tiers across the cohort."
  },
  {
    id: "image_sizes",
    title: "Native Ultrasound Image Resolutions",
    category: "eda",
    categoryLabel: "Exploratory Data Analysis",
    image: "assets/image_sizes.png",
    description: "Scatter and histogram of variable input resolutions from clinical PACS exports prior to standardized 224x224 bicubic letterbox normalization."
  },
  {
    id: "brain_subclasses",
    title: "Fetal Brain Plane Subclass Breakdown",
    category: "eda",
    categoryLabel: "Exploratory Data Analysis",
    image: "assets/brain_subclasses.png",
    description: "Fine-grained anatomical plane stratification within Fetal Brain: Trans-thalamic, Trans-ventricular, and Trans-cerebellar planes."
  }
];

document.addEventListener("DOMContentLoaded", () => {
  renderGallery("all");
  setupFilters();
  setupLightbox();
  setupSmoothScroll();
});

// Render cards into the gallery grid
function renderGallery(filterCategory) {
  const grid = document.getElementById("galleryGrid");
  if (!grid) return;

  const filtered = filterCategory === "all"
    ? GALLERY_ITEMS
    : GALLERY_ITEMS.filter(item => item.category === filterCategory);

  grid.innerHTML = filtered.map(item => `
    <div class="plot-card" data-id="${item.id}" onclick="openLightbox('${item.id}')">
      <div class="plot-img-wrapper">
        <img src="${item.image}" alt="${item.title}" loading="lazy">
        <div class="plot-overlay">
          <span>🔍 View High-Resolution</span>
        </div>
      </div>
      <div class="plot-info">
        <span class="plot-category">${item.categoryLabel}</span>
        <h4 class="plot-title">${item.title}</h4>
        <p class="plot-desc">${item.description}</p>
      </div>
    </div>
  `).join("");
}

// Setup category filter buttons
function setupFilters() {
  const buttons = document.querySelectorAll(".filter-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const filter = btn.getAttribute("data-filter");
      renderGallery(filter);
    });
  });
}

// Setup Lightbox Modal
function setupLightbox() {
  const modal = document.getElementById("lightboxModal");
  const closeBtn = document.getElementById("lightboxClose");

  if (!modal || !closeBtn) return;

  closeBtn.addEventListener("click", closeLightbox);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeLightbox();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeLightbox();
    }
  });
}

function openLightbox(itemId) {
  const item = GALLERY_ITEMS.find(i => i.id === itemId);
  if (!item) return;

  const modal = document.getElementById("lightboxModal");
  const img = document.getElementById("lightboxImg");
  const title = document.getElementById("lightboxTitle");
  const caption = document.getElementById("lightboxCaption");

  img.src = item.image;
  img.alt = item.title;
  title.textContent = item.title;
  caption.textContent = item.description;

  modal.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const modal = document.getElementById("lightboxModal");
  modal.classList.remove("active");
  document.body.style.overflow = "auto";
}

// Smooth scroll & active navigation highlighting
function setupSmoothScroll() {
  const links = document.querySelectorAll(".nav-links a");
  const sections = document.querySelectorAll("section[id]");

  window.addEventListener("scroll", () => {
    let current = "";
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 100;
      if (window.pageYOffset >= sectionTop) {
        current = section.getAttribute("id");
      }
    });

    links.forEach(link => {
      link.classList.remove("active");
      if (link.getAttribute("href") === `#${current}`) {
        link.classList.add("active");
      }
    });
  });
}
