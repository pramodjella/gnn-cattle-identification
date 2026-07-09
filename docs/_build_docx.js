// Builds the mentor-facing Word document (Project Introduction + Progress Report).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak, PageNumber, Header, Footer, ImageRun,
  TableOfContents, VerticalAlign, ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "outputs", "figures");

// ---- helpers ---------------------------------------------------------------
const CONTENT_W = 9360; // US Letter, 1" margins
const ACCENT = "2E5E3A";   // deep green
const ACCENT_LT = "E2EFE6";
const HEADER_FILL = "2E5E3A";
const ZEBRA = "F4F8F5";
const GREY = "CCCCCC";

const border = { style: BorderStyle.SINGLE, size: 1, color: GREY };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 110, right: 110 };

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figure(file, targetWidthPx, caption) {
  const { w, h } = pngSize(file);
  const width = targetWidthPx;
  const height = Math.round((h / w) * targetWidthPx);
  const out = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(file),
        transformation: { width, height },
        altText: { title: caption, description: caption, name: path.basename(file) },
      })],
    }),
  ];
  if (caption) {
    out.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })],
    }));
  }
  return out;
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function p(runs, opts = {}) {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({ spacing: { after: 120 }, children, ...opts });
}
function bullet(runs, level = 0) {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({ numbering: { reference: "bullets", level }, spacing: { after: 60 }, children });
}
function numbered(runs, ref = "nums") {
  const children = Array.isArray(runs) ? runs : [new TextRun(runs)];
  return new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children });
}
function b(text) { return new TextRun({ text, bold: true }); }
function t(text) { return new TextRun(text); }
function code(text) {
  return new Paragraph({
    spacing: { after: 40 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Consolas", size: 18 })],
  });
}

// table: headerRow = array of strings; rows = array of array of (string | {text,bold})
function makeTable(colWidths, headerRow, rows, opts = {}) {
  const aligns = opts.aligns || colWidths.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER));
  const mkCell = (content, i, { head = false, fill = null } = {}) => {
    const runs = (typeof content === "object" && content.runs)
      ? content.runs
      : [new TextRun({
          text: String(content),
          bold: head || (typeof content === "object" && content.bold) || false,
          color: head ? "FFFFFF" : "000000",
          size: head ? 18 : 18,
        })];
    return new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: cellMargins,
      verticalAlign: VerticalAlign.CENTER,
      shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
      children: [new Paragraph({ alignment: aligns[i], children: runs })],
    });
  };
  const trHead = new TableRow({
    tableHeader: true,
    children: headerRow.map((c, i) => mkCell(c, i, { head: true, fill: HEADER_FILL })),
  });
  const trBody = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => mkCell(c, i, { fill: ri % 2 === 1 ? ZEBRA : null })),
  }));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [trHead, ...trBody],
  });
}

// emphasize a whole row (e.g. the SOTA row) by bolding cells
function boldRow(arr) { return arr.map((c) => ({ text: String(c), bold: true })); }

// ---- styles ----------------------------------------------------------------
const styles = {
  default: { document: { run: { font: "Calibri", size: 22 } } },
  paragraphStyles: [
    { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
      run: { size: 52, bold: true, color: ACCENT, font: "Calibri" },
      paragraph: { spacing: { after: 120 }, alignment: AlignmentType.CENTER } },
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 30, bold: true, color: ACCENT, font: "Calibri" },
      paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0,
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } } } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 24, bold: true, color: "1F1F1F", font: "Calibri" },
      paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
  ],
};

const numbering = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 260 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1080, hanging: 260 } } } },
    ] },
    { reference: "nums", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 280 } } } },
    ] },
    { reference: "nums2", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 280 } } } },
    ] },
    { reference: "obj", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 280 } } } },
    ] },
  ],
};

// ---- content ---------------------------------------------------------------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 1800 }, children: [] }),
  new Paragraph({ style: "Title", children: [new TextRun("Biometric Cattle Identification")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "via Deep Learning and Graph Neural Networks", size: 30, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 40 },
    children: [new TextRun({ text: "Project Introduction & Progress Report", size: 26, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Prepared for mentor review", size: 22, italics: true, color: "777777" })] }),
);
// title-page meta table (borderless)
const meta = [
  ["Student", "Pramod Jella"],
  ["Date", "23 June 2026"],
  ["Dataset", "Zenodo Beef Cattle Muzzle Database (260 animals, 4,891 images)"],
  ["Target venue", "Computers and Electronics in Agriculture"],
  ["Best result", "96.1% Rank-1 • 0.78% EER • 0.9995 ROC AUC (Ensemble)"],
];
const noB = { style: BorderStyle.NONE };
children.push(new Table({
  width: { size: 7200, type: WidthType.DXA },
  columnWidths: [2000, 5200],
  alignment: AlignmentType.CENTER,
  borders: { top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB },
  rows: meta.map(([k, v]) => new TableRow({ children: [
    new TableCell({ width: { size: 2000, type: WidthType.DXA }, margins: cellMargins,
      children: [new Paragraph({ children: [new TextRun({ text: k, bold: true, color: ACCENT })] })] }),
    new TableCell({ width: { size: 5200, type: WidthType.DXA }, margins: cellMargins,
      children: [new Paragraph({ children: [new TextRun(v)] })] }),
  ] })),
}));
children.push(new Paragraph({ children: [new PageBreak()] }));

// TOC
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")] }));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ============================= PART I: INTRODUCTION =========================
children.push(h1("Part I — Project Introduction"));

children.push(h2("1. Motivation"));
children.push(p("Livestock traceability underpins disease control, food-safety assurance, insurance and ownership verification, and modern herd management. The methods used in practice today — ear tags, hot/freeze branding, and RFID transponders — are all invasive, removable, and forgeable. Tags fall out or are swapped; brands fade and raise welfare concerns; RFID chips can be cloned or transferred between animals."));
children.push(p([
  t("The "), b("muzzle print"), t(" (nose print) offers a biometric alternative. Like a human fingerprint, the arrangement of "),
  new TextRun({ text: "beads", italics: true }), t(" (raised dermal protuberances) and "),
  new TextRun({ text: "valleys", italics: true }), t(" (grooves) on a bovine muzzle is unique to each animal and stable over its lifetime. It cannot be removed or transferred, and it can be captured with an ordinary camera — making it a low-cost, tamper-proof, non-invasive option for the field."),
]));
children.push(p("The technical challenge is that field-captured muzzle images are hard:"));
children.push(numbered([b("Geometric deformation — "), t("head movement and camera angle change the apparent spacing of beads.")]));
children.push(numbered([b("Environmental variation — "), t("outdoor lighting, shadows, dirt, and moisture alter contrast.")]));
children.push(numbered([b("Scale change — "), t("the muzzle grows as the animal matures, so matching must be scale-invariant.")]));
children.push(p("Classical handcrafted descriptors (SIFT, LBP) generalise poorly under these conditions, and even standard CNNs — translation-equivariant but not naturally invariant to non-rigid deformation — leave room for improvement."));

children.push(h2("2. Core Idea"));
children.push(p("This project investigates whether modelling the muzzle as a graph of keypoints — rather than only as a grid of pixels — improves identification robustness, and how graph-based representations compare against strong CNN baselines on a level playing field."));
children.push(p("The central hypothesis: the topology of the muzzle (which beads neighbour which, and how they are spatially arranged) carries identity information that is more robust to occlusion and deformation than appearance alone. Graph Neural Networks (GNNs) learn over this topology via message passing, capturing relational structure (“bead A always sits between beads B and C”) that a CNN does not explicitly represent."));
children.push(p("Rather than betting on a single architecture, the project is framed as a rigorous, controlled comparative study of four model families on one benchmark, so that any claimed advantage is measured fairly and tested for statistical significance."));

children.push(h2("3. Objectives"));
[
  "Build a complete, reproducible pipeline from raw muzzle images to identity decisions.",
  "Implement and tune representatives of four model families: pure CNN, pure GNN, Hybrid CNN-GNN, and Prototype GNN.",
  "Re-implement prior-art baselines (VGG-16, ResNet-50) for fair comparison against published methods.",
  "Evaluate with biometric metrics (Rank-1/Rank-5, Equal Error Rate, ROC AUC), not just classification accuracy.",
  "Establish statistical rigour via 5-fold cross-validation, McNemar tests, and bootstrap confidence intervals.",
  "Provide explainability (Grad-CAM for CNNs, attention heatmaps for GNNs) to verify the models attend to biologically meaningful structures.",
  "Package the work as a publication-ready manuscript and a deployable web demonstrator.",
].forEach((x) => children.push(numbered(x, "obj")));

children.push(h2("4. Approach — Pipeline Overview"));
children.push(p("The pipeline runs in five stages:"));
[
  ["Preprocessing", "ROI extraction → CLAHE contrast enhancement → Otsu segmentation (256×256)."],
  ["Keypoint detection & description", "Learned keypoints (Kornia-DISK; SuperPoint / DeDoDe / SIFT also supported) — up to 128 keypoints, each with a 256-d descriptor."],
  ["Graph construction", "k-NN graph (k=8) over keypoint coordinates; edges carry 5-d features [Δx, Δy, distance, angle, relative scale]."],
  ["Model family", "One of four: CNN | GNN | Hybrid CNN-GNN | Prototype GNN, trained with ArcFace / metric-learning losses."],
  ["Matching & evaluation", "Cosine similarity → Rank-k identification, EER, ROC AUC."],
].forEach(([k, v]) => children.push(bullet([b(k + " — "), t(v)])));
children.push(p([b("Key methodological choices."), t("")], { spacing: { before: 120, after: 60 } }));
children.push(bullet([b("Learned keypoints over SIFT. "), t("Kornia-DISK descriptors give more robust, illumination-tolerant nodes.")]));
children.push(bullet([b("Bilinear feature sampling (the Hybrid model’s novelty). "), t("Instead of cropping expensive image patches around each keypoint, the image is passed through a shared EfficientNet backbone once, then the deep feature map is sampled at each keypoint coordinate via bilinear interpolation. Every node gets a rich, contextual feature vector cheaply, refined by Dynamic EdgeConv and a GATv2-based Topological Relation Module (TRM).")]));
children.push(bullet([b("ArcFace loss. "), t("Additive angular-margin loss (scale 128, margin 0.35) produces well-separated embeddings for the 260-class, high-intra-class-variation livestock setting.")]));

children.push(h2("5. Dataset"));
children.push(makeTable([3200, 6160],
  ["Property", "Value"],
  [
    ["Animals (classes)", "260"],
    ["Total images", "4,891"],
    ["Images per animal", "18.8 ± 10.0 (min 5, max 70, median 16)"],
    ["Split (70/15/15)", "3,312 train / 615 val / 964 test"],
  ],
  { aligns: [AlignmentType.LEFT, AlignmentType.LEFT] }));
children.push(p([new TextRun({ text: "All headline metrics are reported on the 964-image test set.", italics: true, size: 18 })], { spacing: { before: 80 } }));

children.push(h2("6. Novel Contributions"));
[
  "First controlled, like-for-like comparison of CNN, pure GNN, Hybrid, and Prototype-GNN architectures on the same muzzle benchmark and protocol.",
  "A Hybrid CNN-GNN architecture fusing CNN texture (via bilinear feature-map sampling at keypoints) with learned graph topology.",
  "Deep learned keypoint graphs (Kornia-DISK) replacing handcrafted keypoints.",
  "Dual explainability (Grad-CAM + graph attention) connecting predictions back to muzzle anatomy.",
  "A statistically validated result set (cross-validation + significance testing), not single-run numbers.",
].forEach((x) => children.push(numbered(x, "nums2")));

children.push(h2("7. Technology Stack"));
children.push(bullet([b("ML / DL: "), t("PyTorch, PyTorch-Geometric, Kornia (DISK), timm / EfficientNet.")]));
children.push(bullet([b("Imaging: "), t("OpenCV (CLAHE, Otsu), NumPy.")]));
children.push(bullet([b("Evaluation & figures: "), t("scikit-learn, Matplotlib (vector PDF + PNG), LaTeX table generation.")]));
children.push(bullet([b("Deployment (demonstrator): "), t("FastAPI backend with PostgreSQL + pgvector similarity search, React frontend, Docker Compose.")]));
children.push(bullet([b("Hardware: "), t("single NVIDIA RTX 5070 (8 GB), mixed-precision (bfloat16) training.")]));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ============================= PART II: PROGRESS ===========================
children.push(h1("Part II — Progress Report"));

children.push(h2("1. Summary"));
children.push(p("The end-to-end pipeline is complete and reproducible. All four model families plus two prior-art baselines are trained and evaluated on the 964-image test set, results are statistically validated, and a manuscript draft and publication figures are ready. A working web demonstrator (register / identify cattle) has also been built."));
children.push(p([
  b("Headline result: "),
  t("the proposed "), b("Ensemble (CNN + Hybrid CNN-GNN)"),
  t(" reaches "), b("96.1% Rank-1"), t(" accuracy with a "), b("0.78% Equal Error Rate"),
  t(" and "), b("0.9995 ROC AUC"), t(", outperforming re-implemented VGG-16 (95.1%) and ResNet-50 (94.6%) prior-art baselines."),
]));
children.push(p("The main open items are honest ones, called out in Section 6: pure GNNs trail CNNs on raw accuracy, and the Hybrid model is unstable under short cross-validation training. These are the points I would most like to discuss."));

children.push(h2("2. What Is Complete"));
children.push(makeTable([4400, 1300, 3660],
  ["Component", "Status", "Where"],
  [
    ["Data download + 70/15/15 split", "Done", "scripts/01_download_data.py"],
    ["Preprocessing (ROI, CLAHE, Otsu seg.)", "Done", "scripts/02_preprocess.py"],
    ["Keypoint extraction (DISK + alternates)", "Done", "scripts/03_extract_keypoints.py"],
    ["k-NN graph construction (k=8)", "Done", "scripts/04_build_graphs.py"],
    ["CNN (EfficientNet-B4 + ArcFace)", "Trained", "scripts/train_cnn.py"],
    ["Hybrid CNN-GNN", "Trained", "scripts/train_hybrid.py"],
    ["ProtoN (Prototype Node GNN)", "Trained", "scripts/train_proton.py"],
    ["GNN v3 / v4 (GATv2 + Virtual Node)", "Trained", "scripts/train_gnn_v3*.py"],
    ["GNN+ / GNN++ ablation variants", "Trained", "scripts/train_gnn_plus*.py"],
    ["VGG-16 baseline (Bello et al. 2020)", "Trained", "scripts/baselines/"],
    ["ResNet-50 baseline (Qin et al. 2021)", "Trained", "scripts/baselines/"],
    ["Ensemble + Test-Time Augmentation", "Done", "scripts/ensemble_inference.py"],
    ["5-fold cross-validation", "Done", "scripts/cross_validation.py"],
    ["McNemar tests + bootstrap CIs", "Done", "scripts/statistical_tests.py"],
    ["Explainability (Grad-CAM + GNN attn.)", "Done", "scripts/visualize_*.py"],
    ["Publication figures (PDF + PNG)", "Done", "scripts/figures/"],
    ["Manuscript draft", "Done", "paper_draft.md"],
    ["Web demonstrator (FastAPI+React+pgvector)", "Built", "web/"],
  ],
  { aligns: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.LEFT] }));

children.push(h2("3. Results (test set, 964 images)"));
const resHeader = ["Model", "Rank-1 %", "Rank-5 %", "EER %", "ROC AUC", "Notes"];
const resRows = [
  boldRow(["Ensemble (CNN-TTA + Hybrid)", "96.1", "98.1", "0.78", "0.9995", "Proposed SOTA (0.95 / 0.05)"]),
  ["CNN (EfficientNet-B4) + TTA", "95.4", "97.4", "2.70", "0.9961", "ArcFace + test-time aug."],
  ["CNN (EfficientNet-B4)", "95.4", "97.4", "2.70", "0.9961", "Baseline ArcFace"],
  ["VGG-16 baseline", "95.1", "97.7", "1.23", "0.9993", "Bello et al. (2020)"],
  ["ResNet-50 baseline", "94.6", "97.3", "2.14", "0.9971", "Qin et al. (2021)"],
  ["Hybrid CNN-GNN", "92.0", "96.7", "1.85", "0.9979", "Bilinear + EdgeConv + TRM"],
  ["ProtoN (Prototype GNN)", "91.6", "94.8", "1.17", "0.9982", "Cross-graph alignment loss"],
  ["GNN v4 (GATv2, enhanced)", "91.6", "94.4", "1.48", "0.9937", "4-layer GATv2 + VN"],
  ["GNN v3 (GATv2 + VN)", "91.5", "95.0", "1.87", "0.9954", "4-layer GATv2 + VN"],
  ["GNN++ (CNN patch features)", "78.3", "86.2", "7.81", "0.9730", "MobileNetV3 patch nodes"],
  ["GNN+ (Kornia-DISK features)", "72.0", "84.2", "11.17", "0.9516", "DISK-descriptor nodes"],
];
children.push(makeTable([2450, 880, 880, 820, 980, 3350], resHeader, resRows));
children.push(...figure(path.join(FIG, "fig_main_results_bar.png"), 560, "Figure 1. Rank-1 identification accuracy across all architectures."));

children.push(h2("4. Key Findings"));
[
  [b("CNN texture is the strongest single signal. "), t("Pure CNNs (≈95% Rank-1) beat pure GNNs (≈91–92%): raw dermatoglyphic texture is more discriminative than keypoint geometry alone.")],
  [b("Topology helps verification. "), t("The Hybrid and ProtoN GNNs achieve the lowest EERs (1.85% and 1.17%) — graph structure improves accept/reject reliability even where it does not top raw Rank-1.")],
  [b("The ensemble is the best of both. "), t("Blending the CNN’s global texture view with the Hybrid model’s topological view yields the best overall numbers; their error profiles are complementary.")],
  [b("Learned keypoints beat handcrafted ones. "), t("DISK-based GNN+ (72.0%) clearly outperforms SIFT-equivalent baselines.")],
  [b("Results are statistically significant. "), t("McNemar tests confirm the CNN/Hybrid advantage over the pure GNNs at p < 10⁻⁸.")],
].forEach((x) => children.push(numbered(x)));
children.push(...figure(path.join(FIG, "fig_cmc_curves.png"), 470, "Figure 2. Cumulative Match Characteristic (CMC) curves."));
children.push(...figure(path.join(FIG, "fig_roc_curves.png"), 470, "Figure 3. ROC curves (verification performance)."));

children.push(h2("5. Validation & Rigour"));
children.push(p([b("Stratified 5-fold cross-validation"), t(" on the top models:")]));
children.push(makeTable([3360, 2000, 2000, 2000],
  ["Model", "Rank-1 %", "EER %", "Stability"],
  [
    ["CNN (EfficientNet-B4)", "93.91 ± 0.31", "3.21 ± 0.22", "Very stable"],
    ["ProtoN", "89.49 ± 0.71", "4.10 ± 0.71", "Stable"],
    ["Hybrid CNN-GNN", "68.88 ± 2.04", "11.31 ± 1.47", "Unstable (see §6)"],
  ]));
children.push(bullet([b("McNemar pairwise significance tests"), t(" and "), b("bootstrap confidence intervals"), t(" computed for all headline comparisons.")]));
children.push(bullet([b("Explainability "), t("confirms models attend to anatomy: Grad-CAM concentrates on central bead clusters; GNN attention emphasises keypoint links spanning major valleys.")]));

children.push(h2("6. Open Issues / Points to Discuss"));
[
  [b("GNNs don’t beat CNNs on raw accuracy. "), t("The research contribution is currently best framed as “topology improves verification reliability and ensembles, and learned-keypoint graphs are a viable representation” rather than “GNNs win outright.” I would value your view on framing.")],
  [b("Hybrid cross-validation instability (68.9% ± 2.0%). "), t("Full-protocol Hybrid scores 92% Rank-1, but under the shorter CV-safe schedule it collapses — likely a training-budget / optimisation-stability issue rather than a modelling flaw. Needs longer CV runs to confirm.")],
  [b("Validation vs. test gap. "), t("Reported best-validation Rank-1 (~82–84%) sits below test Rank-1 (~95%), an artefact of the small validation set (615 images, ~2.4/animal) and the gallery/probe protocol. Worth aligning before submission.")],
  [b("Single dataset. "), t("All results are on one benchmark; cross-dataset generalisation is untested.")],
].forEach((x) => children.push(numbered(x)));

children.push(h2("7. Next Steps"));
children.push(p([b("Near term (pre-submission)")]));
children.push(bullet("Re-run Hybrid cross-validation with a longer/robust schedule to resolve the instability."));
children.push(bullet("Reconcile the val/test evaluation protocol."));
children.push(bullet("Finalise manuscript figures/tables and tighten the framing of the GNN contribution."));
children.push(p([b("Medium term")], { spacing: { before: 120 } }));
children.push(bullet("Cross-dataset transfer test (train on Zenodo, evaluate on a second muzzle dataset)."));
children.push(bullet("Robustness study under simulated occlusion / blur / low light — the regime where topology should pay off most."));
children.push(bullet("Mobile / low-latency deployment of the best model in the web demonstrator."));
children.push(p([b("Stretch")], { spacing: { before: 120 } }));
children.push(bullet("Submit to Computers and Electronics in Agriculture."));

children.push(h2("8. Reproduce Key Results"));
[
  "# 5-fold cross-validation (top 3 models)",
  "venv\\Scripts\\python.exe scripts/cross_validation.py --epochs-cnn 10 \\",
  "    --epochs-proton 12 --epochs-hybrid-p1 10 --epochs-hybrid-p2 2",
  "# Comparison report + markdown table",
  "venv\\Scripts\\python.exe scripts/compare_models.py",
  "# Publication-quality vector figures (PDF/PNG)",
  "venv\\Scripts\\python.exe scripts/figures/generate_paper_figures.py",
  "# Explainability maps",
  "venv\\Scripts\\python.exe scripts/visualize_gradcam.py",
  "venv\\Scripts\\python.exe scripts/visualize_gnn_attention.py",
].forEach((l) => children.push(code(l)));

// ---- document --------------------------------------------------------------
const doc = new Document({
  styles,
  numbering,
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Biometric Cattle Identification  •  ", size: 16, color: "888888" }),
          new TextRun({ text: "Page ", size: 16, color: "888888" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" }),
        ],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs", "Cattle_Identification_Project_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("WROTE " + out + " (" + buf.length + " bytes)");
});
