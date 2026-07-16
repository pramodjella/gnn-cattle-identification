// Build the team-review Word version of the domain paper from paper_draft.md.
// Usage:  npm install docx   &&   node docs/_build_paper_docx.js
// (run from the repo root; embeds rendered PNG figures after their tables).
const fs = require('fs');
const path = require('path');
const docx = require('docx');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel,
        AlignmentType, BorderStyle, WidthType, ShadingType, LevelFormat, TableOfContents, ImageRun } = docx;
const ROOT = path.resolve(__dirname, '..');
const FIGDIR = path.join(ROOT, 'outputs/figures');
const ARCH_IMG = path.join(FIGDIR, 'architecture/hybrid_architecture.png');
// figures inserted after their table caption: {captionMatch: [path, number, caption, w, h]}
const TABLE_FIGS = [
  { m: /Table 3:/, img: FIGDIR + '/extension/fig_cross_dataset.png', n: 2, w: 520, h: 221,
    cap: 'Zero-shot cross-dataset transfer: identification (Rank-1) and verification (EER) on two external muzzle datasets.' },
  { m: /Table 4:/, img: FIGDIR + '/extension/fig_snorm.png', n: 3, w: 440, h: 255,
    cap: 'Label-free S-norm test-time calibration reduces cross-domain verification EER on both datasets.' },
  { m: /Table 5:/, img: FIGDIR + '/extension/fig_corruption_rank1.png', n: 4, w: 560, h: 182,
    cap: 'Rank-1 vs corruption severity: the validation-tuned fusion (yellow) tracks or exceeds the CNN across all corruptions, while the Hybrid and per-sample quality-aware gate collapse under spatter.' },
  { m: /Table 6:/, img: FIGDIR + '/extension/fig_stage1_attribution.png', n: 5, w: 360, h: 606,
    cap: 'Stage 1 attribution across case types (one representative per row): muzzle image, CNN Grad-CAM, and GNN per-keypoint importance.' },
];

const SRC = path.join(ROOT, 'paper_draft.md');
const OUT = path.join(ROOT, 'docs/Cattle_Muzzle_Biometrics_Paper.docx');
const lines = fs.readFileSync(SRC, 'utf8').split(/\r?\n/);

const CW = 9360; // content width DXA (US Letter, 1" margins)
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };

// inline: split **bold**, `code`, strip $math$
function runs(text) {
  text = text.replace(/\$\$?/g, '')
             .replace(/\\,|\\;|\\!|\\:/g, '')
             .replace(/\\le/g, '≤').replace(/\\ge/g, '≥').replace(/\\times/g, '×')
             .replace(/\\theta/g, 'θ').replace(/\\alpha/g, 'α').replace(/\\Delta/g, 'Δ')
             .replace(/\\to/g, '→').replace(/\\gg/g, '≫').replace(/\\sim/g, '~').replace(/\\cos/g, 'cos')
             .replace(/_\{([^}]*)\}/g, '_$1').replace(/\^\{([^}]*)\}/g, '^$1')
             .replace(/\\%/g, '%').replace(/\\_/g, '_').replace(/\\&/g, '&')
             .replace(/\\[a-zA-Z]+/g, '');
  const out = [];
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const p of parts) {
    if (!p) continue;
    if (p.startsWith('**') && p.endsWith('**')) out.push(new TextRun({ text: p.slice(2, -2), bold: true }));
    else if (p.startsWith('`') && p.endsWith('`')) out.push(new TextRun({ text: p.slice(1, -1), font: 'Consolas' }));
    else out.push(new TextRun(p));
  }
  return out.length ? out : [new TextRun(text)];
}

function table(rows) {
  const cells = rows.map(r => r.replace(/^\||\|$/g, '').split('|').map(c => c.trim()));
  const header = cells[0];
  const body = cells.slice(2); // skip separator row
  const ncol = header.length;
  const colW = Math.floor(CW / ncol);
  const colWidths = Array(ncol).fill(colW); colWidths[ncol - 1] = CW - colW * (ncol - 1);
  const mkRow = (arr, head) => new TableRow({
    children: arr.map((c, i) => new TableCell({
      borders, width: { size: colWidths[i], type: WidthType.DXA },
      shading: head ? { fill: "D9E7F5", type: ShadingType.CLEAR } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: runs(c.replace(/\*\*/g, '')).map(r =>
        new TextRun({ text: r.text || c, bold: head })), alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })]
    }))
  });
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: colWidths,
    rows: [mkRow(header, true), ...body.map(r => mkRow(r, false))] });
}

const children = [];
let i = 0;
let lastLine = '';
// title (first # line)
while (i < lines.length) {
  const L = lines[i];
  if (/^#\s+/.test(L)) { children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: runs(L.replace(/^#\s+/, '')) })); i++; break; }
  i++;
}
children.push(new Paragraph({ children: [new TextRun({ text: 'Draft manuscript — for internal team review', italics: true, color: '666666' })] }));
children.push(new Paragraph({ children: [new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-3' })] }));

for (; i < lines.length; i++) {
  let L = lines[i];
  if (/^\s*$/.test(L)) continue;
  if (/^---+$/.test(L)) continue;
  if (/^```/.test(L)) { // code block
    i++; const buf = [];
    while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
    const blockText = buf.join('\n');
    if (/Muzzle Image|EfficientNet/.test(blockText) && fs.existsSync(ARCH_IMG)) {
      // architecture ASCII diagram -> embed rendered figure
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(ARCH_IMG),
          transformation: { width: 540, height: 424 },
          altText: { title: 'Hybrid CNN-GNN architecture', description: 'Hybrid CNN-GNN pipeline', name: 'arch' } })] }));
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
        children: [new TextRun({ text: 'Figure 1. Hybrid CNN-GNN architecture: bilinear feature sampling at DISK keypoints feeds a graph reasoning head.', italics: true, size: 18, color: '444444' })] }));
    } else {
      for (const b of buf) children.push(new Paragraph({ children: [new TextRun({ text: b || ' ', font: 'Consolas', size: 18 })] }));
    }
    continue;
  }
  const mH = L.match(/^(#{2,4})\s+(.*)/);
  if (mH) {
    const lvl = mH[1].length; // ## ->1, ### ->2, #### ->3
    const hl = lvl === 2 ? HeadingLevel.HEADING_1 : lvl === 3 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3;
    children.push(new Paragraph({ heading: hl, children: runs(mH[2]) }));
    continue;
  }
  if (/^\s*\|.*\|\s*$/.test(L)) { // table
    const rows = []; while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { rows.push(lines[i]); i++; } i--;
    children.push(table(rows));
    children.push(new Paragraph({ text: '' }));
    // embed a result figure after its table (by caption match)
    const tf = TABLE_FIGS.find(f => f.m.test(lastLine) && fs.existsSync(f.img));
    if (tf) {
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 40 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(tf.img), transformation: { width: tf.w, height: tf.h },
          altText: { title: 'Figure ' + tf.n, description: tf.cap, name: 'fig' + tf.n } })] }));
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
        children: [new TextRun({ text: `Figure ${tf.n}. ${tf.cap}`, italics: true, size: 18, color: '444444' })] }));
    }
    continue;
  }
  const mUL = L.match(/^\s*[-*]\s+(.*)/);
  if (mUL) { children.push(new Paragraph({ numbering: { reference: 'bul', level: 0 }, children: runs(mUL[1]) })); continue; }
  const mOL = L.match(/^\s*\d+\.\s+(.*)/);
  if (mOL) { children.push(new Paragraph({ numbering: { reference: 'num', level: 0 }, children: runs(mOL[1]) })); continue; }
  lastLine = L;
  children.push(new Paragraph({ children: runs(L), spacing: { after: 120 } }));
}

const doc = new Document({
  styles: { default: { document: { run: { font: 'Calibri', size: 21 } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal', next: 'Normal', run: { size: 34, bold: true, font: 'Calibri' }, paragraph: { spacing: { after: 120 } } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, color: '1F3864', font: 'Calibri' }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, color: '2E5496', font: 'Calibri' }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 22, bold: true, italics: true, font: 'Calibri' }, paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 2 } },
    ] },
  numbering: { config: [
    { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
    { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
  ] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children }]
});

Packer.toBuffer(doc).then(b => {
  let out = OUT;
  try { fs.writeFileSync(out, b); }
  catch (e) { out = OUT.replace(/\.docx$/, '_new.docx'); fs.writeFileSync(out, b);
    console.log('(primary locked; wrote fallback)'); }
  console.log('wrote', out, b.length, 'bytes');
});
