// Build paper/build/paper.html and paper/build/paper.pdf from paper/draft/*.md (4 Nov 2026).
// Usage (from /data/arc): node paper/build_pdf.cjs  (needs markdown-it from the preinstalled node_modules; chromium).
const fs = require('fs'), path = require('path'), { execFileSync } = require('child_process');
const MD = require((() => { for (const p of ['markdown-it', '/data/node_modules/markdown-it', '/vercel/sandbox/node_modules/markdown-it']) { try { return require.resolve(p); } catch (e) {} } throw new Error('markdown-it not found'); })());
const md = new MD({ html: true, typographer: false });
const R = path.resolve(__dirname, '..');
const files = fs.readdirSync(path.join(R, 'paper', 'draft')).filter(f => f.endsWith('.md')).sort();
let body = files.map(f => fs.readFileSync(path.join(R, 'paper', 'draft', f), 'utf8').replace(/<!--[\s\S]*?-->/g, '')).join('\n\n');
// Order sections by their number ("## 4." / "### 4.3"); unnumbered chunks (abstract) stay first. Stable sort.
{ const chunks = []; for (const line of body.split('\n')) { const h = line.match(/^#{2,3} (\d+)(?:\.(\d+))?[.\s]/);
    if (h || !chunks.length) chunks.push({ key: h ? +h[1] * 100 + +(h[2] || 0) : -1, lines: [] }); chunks[chunks.length - 1].lines.push(line); }
  body = chunks.map((c, i) => ({ ...c, i })).sort((x, y) => x.key - y.key || x.i - y.i).map(c => c.lines.join('\n')).join('\n'); }
const FIGS = [
  ['s37_noise_and_hazard.png', 'Fig. 1. Left: five byte-identical runs of our B12 notebook placed on the final public ARC-AGI-3 board. Right: level-clear hazard in token time (dev vs holdout games).'],
  ['s37_survival_fit.png', 'Fig. 2. Survival of level attempts in generated-token time: observed (Kaplan-Meier) vs the fitted crack model.'],
  ['s37_pergame_lottery.png', 'Fig. 3. Per-game level counts across the six valid runs of one agent: the lottery is per game.'],
  ['s37_noise_over_time.png', 'Fig. 4. Variance of per-game levels over the run: fixed clock vs fixed tokens per game.'],
  ['s37_v5_dose_response.png', 'Fig. 5. Levels vs play time: four runs at 121 min per game, the V5 run at twice the play time, the sealed prediction (127, 90% 110-145) and three alternatives.'],
  ['s37_arc2_plateau.png', 'Fig. 6. ARC-AGI-2: the plateau and its rise with entries, against best-of-n draws of one notebook.'],
];
// Place each figure after the first block (paragraph, list or table) that cites "Fig. N"; uncited figures go to the end.
const blocks = body.split(/\n\s*\n/), isList = b => /^\s*(\d+\.|[-*])\s/.test(b), after = {}, unplaced = [];
FIGS.forEach(([f, c], i) => {
  let k = blocks.findIndex(b => new RegExp(`Fig\\. ?${i + 1}\\b`).test(b.replace(/^#.*$/gm, '')));
  if (k < 0) { unplaced.push(i); return; }
  while (isList(blocks[k]) && k + 1 < blocks.length && isList(blocks[k + 1])) k++;
  (after[k] = after[k] || []).push(i);
});
const figHtml = i => `<figure><img src="../figs/${FIGS[i][0]}"><figcaption>${FIGS[i][1]}</figcaption></figure>`;
body = blocks.map((b, k) => [b, ...(after[k] || []).map(figHtml)].join('\n\n')).join('\n\n');
const figOrder = Object.keys(after).map(Number).sort((x, y) => x - y).flatMap(k => after[k]);
if (figOrder.some((f, j) => j && f < figOrder[j - 1])) console.warn('WARNING: figures are not cited in numeric order:', figOrder.map(i => i + 1).join(','));
if (unplaced.length) body += '\n\n## Figures not cited in the text\n\n' + unplaced.map(figHtml).join('\n\n');
// References: entries "- [Key] text" (+ indented continuation lines) in paper/REFERENCES.md; list the ones the draft cites.
const refs = {}; let cur = null;
for (const line of fs.readFileSync(path.join(R, 'paper', 'REFERENCES.md'), 'utf8').split('\n')) {
  const m = line.match(/^- \[([A-Za-z0-9]+)\]\s*(.*)$/);
  if (m) { cur = m[1]; refs[cur] = m[2]; } else if (cur && /^\s{2,}\S/.test(line)) refs[cur] += ' ' + line.trim(); else cur = null;
}
const cited = [...new Set([...body.matchAll(/\[([A-Za-z][A-Za-z0-9]*)\]/g)].map(m => m[1]))];
const PLACEHOLDERS = ['V5', 'PRIVATE'];
const missing = cited.filter(k => !refs[k] && !PLACEHOLDERS.includes(k)), uncited = Object.keys(refs).filter(k => !cited.includes(k));
const refList = Object.keys(refs).filter(k => cited.includes(k)).sort((a, b) => a.localeCompare(b)).map(k => `- [${k}] ${refs[k]}`).join('\n');
const table = t => { const p = path.join(R, 'paper', 'tables', t); return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : `(missing ${t})`; };
body += '\n\n## Table 1. Our ARC-AGI-3 submissions\n\n' + table('t1_arc3_submissions.md');
body += '\n\n## Table 2. Our ARC-AGI-2 submissions\n\n' + table('t2_arc2_submissions.md');
body += '\n\n## References\n\n' + refList;
body += '\n\n## Appendix A. The 32 negative ledger rows\n\n' + table('tA_negative_results.md');
body += '\n\n## Appendix B. All 39 development runs\n\n' + table('tB_dev_runs.md');
body += '\n\n## Appendix C. The agent and its slot rule\n\n' + table('tC_agent.md').replace(/<!--[\s\S]*?-->/g, '');
body += '\n\n## Appendix D. The sealed tests in full\n\n' + table('tD_sealed.md').replace(/<!--[\s\S]*?-->/g, '');
body += '\n\n## Appendix E. More detail on Sections 4.3 and 4.6 to 4.8\n\n' + table('tE_detail.md').replace(/<!--[\s\S]*?-->/g, '');
const css = `body{font-family:'Liberation Serif','Times New Roman',serif;font-size:10.5pt;line-height:1.38;max-width:7.1in;margin:0 auto;color:#111}
h1{font-size:17pt;margin:0 0 4pt} h2{font-size:13pt;margin:16pt 0 4pt;border-bottom:1px solid #ccc} h3{font-size:11pt;margin:12pt 0 3pt}
p{margin:5pt 0;text-align:justify} li{margin:2pt 0} figure{margin:10pt 0;page-break-inside:avoid} img{max-width:100%;max-height:6.8in;display:block;margin:0 auto}
pre{font-size:8.2pt;line-height:1.3;white-space:pre-wrap;margin:6pt 0} figcaption{font-size:9pt;color:#333} table{border-collapse:collapse;font-size:7.6pt;margin:6pt 0} td,th{border:1px solid #bbb;padding:2pt 3pt;vertical-align:top}
.draft{color:#a00;font-size:9pt;border:1px solid #a00;padding:4pt;margin-bottom:8pt}`;
// --final (6 Nov 2026): no DRAFT banner or build-time stamp; lists any placeholder left in the text.
const FINAL = process.argv.includes('--final');
const left = (body.match(/\[(PRIVATE|LINK|TODO)[^\]]*\]/g) || []);
if (FINAL && left.length) console.warn('WARNING --final: placeholders left:', JSON.stringify(left));
const title = '<h1>The ARC-AGI-3 Lottery: a testable model of LLM-agent scores under Kaggle compute</h1>' +
  (FINAL ? '<p>Shreekumar Shah (Kaggle: shreekumarshah). November 2026. Code, data and sealed files: <a href="https://github.com/Shreekumar-Shah-AICTE/arc-agi-3-lottery">github.com/Shreekumar-Shah-AICTE/arc-agi-3-lottery</a></p>' :
  '<p>Shreekumar Shah (Kaggle: shreekumarshah). Draft v0, ' + new Date().toISOString().slice(0, 16).replace('T', ' ') + ' UTC.</p>' +
  '<div class="draft">DRAFT. The placeholder [PRIVATE] marks results that do not exist yet. Every number is checked against paper/results.json.</div>');
const html = `<!doctype html><html><head><meta charset="utf-8"><title>ARC-AGI-3 Lottery${FINAL ? '' : ' (draft)'}</title><style>${css}</style></head><body>${title}${md.render(body)}</body></html>`;
const out = path.join(R, 'paper', 'build', 'paper.html'); fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, html);
execFileSync('chromium', ['--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer', `--print-to-pdf=${path.join(R, 'paper', 'build', 'paper.pdf')}`, 'file://' + out], { stdio: 'ignore' });
console.log('built', out, 'and paper.pdf from', files.length, 'draft files;', 'references cited', cited.filter(k => refs[k]).length, '| missing', JSON.stringify(missing), '| listed but uncited', JSON.stringify(uncited), '| figures not cited', JSON.stringify(unplaced.map(i => i + 1)));
if (missing.length) process.exitCode = 1;
