// index.js — digital chessboard driven by ISO timestamps in moves.json
console.log('index.js loaded');

const FILE_URL = '/moves.json';
const DEFAULT_DT = 2.0; // seconds for missing timestamps

// Glyphs
const GLYPHS = {
  white: { king: '♔', queen: '♕', rook: '♖', bishop: '♗', knight: '♘', pawn: '♙' },
  black: { king: '♚', queen: '♛', rook: '♜', bishop: '♝', knight: '♞', pawn: '♟' }
};

// Algebraic helpers
const files = ['a','b','c','d','e','f','g','h'];
const ranks = ['8','7','6','5','4','3','2','1']; // top → bottom
function algebraicToIdx(square) {
  const f = files.indexOf(square[0]);
  const r = ranks.indexOf(square[1]);
  if (f < 0 || r < 0) return null;
  return { row: r, col: f };
}
function idxToAlgebraic(row, col) { return files[col] + ranks[row]; }

// DOM
const boardEl = document.getElementById('board');
const coordsLeftEl = document.getElementById('coords-left');
const coordsBottomEl = document.getElementById('coords-bottom');
const timeSlider = document.getElementById('time-slider');
const timeLabel = document.getElementById('time-label');
const btnReset = document.getElementById('btn-reset');
const btnPrev  = document.getElementById('btn-prev');
const btnNext  = document.getElementById('btn-next');
const btnPlay  = document.getElementById('btn-play');
const btnPause = document.getElementById('btn-pause');
const snapMoves = document.getElementById('snap-moves');
const statusEl = document.getElementById('status');
const logList = document.getElementById('move-log');

// State
let squares = [];
let boardState = {};
let moves = [];        // {turn, from,to,action,color,piece,t,iso}
let timer = null;
let tCurrent = 0;
let lastFrom = null, lastTo = null;

// Build board + coordinate labels
(function buildBoard() {
  for (let r = 0; r < 8; r++) {
    const div = document.createElement('div');
    div.textContent = ranks[r];
    coordsLeftEl.appendChild(div);
  }
  for (let c = 0; c < 8; c++) {
    const div = document.createElement('div');
    div.textContent = files[c];
    coordsBottomEl.appendChild(div);
  }
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement('div');
      cell.className = 'square ' + ((r + c) % 2 ? 'dark' : 'light');
      cell.dataset.rc = `${r}-${c}`;
      boardEl.appendChild(cell);
      squares.push(cell);
    }
  }
})();

function setStartPosition() {
  boardState = {};
  // pawns
  for (let c = 0; c < 8; c++) {
    boardState[idxToAlgebraic(6, c)] = { color: 'white', piece: 'pawn' }; // rank 2
    boardState[idxToAlgebraic(1, c)] = { color: 'black', piece: 'pawn' }; // rank 7
  }
  // back ranks
  const back = ['rook','knight','bishop','queen','king','bishop','knight','rook'];
  for (let c = 0; c < 8; c++) {
    boardState[idxToAlgebraic(7, c)] = { color: 'white', piece: back[c] }; // rank 1
    boardState[idxToAlgebraic(0, c)] = { color: 'black', piece: back[c] }; // rank 8
  }
  lastFrom = lastTo = null;
  renderBoard();
}

function renderBoard() {
  for (const cell of squares) {
    cell.textContent = '';
    cell.classList.remove('last-from','last-to');
  }
  for (const sq in boardState) {
    const p = boardState[sq];
    const idx = algebraicToIdx(sq);
    if (!idx || !p) continue;
    const i = idx.row * 8 + idx.col;
    squares[i].textContent = GLYPHS[p.color][p.piece];
  }
  if (lastFrom) {
    const f = algebraicToIdx(lastFrom);
    squares[f.row * 8 + f.col].classList.add('last-from');
  }
  if (lastTo) {
    const t = algebraicToIdx(lastTo);
    squares[t.row * 8 + t.col].classList.add('last-to');
  }
}

function applyMove(mv) {
  const { from, to } = mv;
  const piece = boardState[from];
  if (!piece) return;
  delete boardState[from];
  boardState[to] = { color: piece.color, piece: piece.piece };
  lastFrom = from; lastTo = to;
}

// figure out current move index given tCurrent
function currentMoveIndex() {
  // index of the last move whose t <= tCurrent
  let idx = -1;
  for (let i = 0; i < moves.length; i++) {
    if (moves[i].t <= tCurrent + 1e-6) idx = i;
    else break;
  }
  return idx;
}

function renderAtTime(t) {
  tCurrent = t;
  timeSlider.value = t.toFixed(2);
  timeLabel.textContent = `${(+t).toFixed(2)}s`;
  setStartPosition();
  for (const mv of moves) {
    if (mv.t <= t + 1e-6) applyMove(mv);
    else break;
  }
  renderBoard();
  const ci = currentMoveIndex();
  highlightLogIndex(ci);
}

function seekToIndex(i) {
  if (!moves.length) return;
  const clamp = Math.max(0, Math.min(i, moves.length - 1));
  renderAtTime(moves[clamp].t);
}

function stepToMove(dir) {
  // dir: -1 prev, +1 next
  const ci = currentMoveIndex();
  const target = Math.max(0, Math.min(moves.length - 1, ci + dir));
  seekToIndex(target);
}

function play() {
  if (timer) return;
  timer = setInterval(() => {
    const maxT = moves.length ? moves[moves.length - 1].t : 0;
    const next = tCurrent + 0.1;
    if (next >= maxT) { renderAtTime(maxT); pause(); }
    else { renderAtTime(next); }
  }, 100);
}
function pause() { if (timer) clearInterval(timer); timer = null; }

// Controls
btnReset.onclick = () => { pause(); renderAtTime(0); };
btnPrev .onclick = () => { pause(); stepToMove(-1); };
btnNext .onclick = () => { pause(); stepToMove(+1); };
btnPlay .onclick = () => { play(); };
btnPause.onclick = () => { pause(); };

timeSlider.oninput = () => {
  const val = parseFloat(timeSlider.value || '0');
  if (snapMoves.checked && moves.length) {
    let best = moves[0].t, bestd = Math.abs(val - best);
    for (const m of moves) {
      const d = Math.abs(val - m.t);
      if (d < bestd) { best = m.t; bestd = d; }
    }
    renderAtTime(best);
  } else {
    renderAtTime(val);
  }
};

// timestamps → unix seconds
function toUnixSeconds(x) {
  if (x == null) return null;
  if (typeof x === 'number') return x;
  if (typeof x === 'string') {
    const t = Date.parse(x);
    return isNaN(t) ? null : t / 1000;
  }
  return null;
}

// ===== Move Log =====
function formatSANish(m) {
  const arrow = ' \u2192 ';
  const act = m.action || 'move';
  const who = `${m.color} ${m.piece}`;
  const turnTxt = (m.turn != null) ? `Turn ${m.turn} — ` : '';
  return `${turnTxt}${who}: ${m.from} ${arrow} ${m.to} (${act})`;
}

function renderMoveLog(movesArr) {
  logList.innerHTML = '';
  movesArr.forEach((m, i) => {
    const li = document.createElement('li');
    li.textContent = formatSANish(m);
    li.dataset.idx = i;
    if ((m.action || '').toLowerCase() === 'capture') li.classList.add('capture');
    li.addEventListener('click', () => seekToIndex(i));
    logList.appendChild(li);
  });
}

function highlightLogIndex(idx) {
  [...logList.children].forEach((li, i) => {
    if (i === idx) li.classList.add('current');
    else li.classList.remove('current');
  });
  const cur = logList.children[idx];
  if (cur) cur.scrollIntoView({ block: 'nearest' });
}

// init
async function init() {
  try {
    const res = await fetch(FILE_URL, { cache: 'no-store' });
    const raw = await res.json();

    // collect unix times, fill missing
    let unixTimes = raw.map(m => toUnixSeconds(m.timestamp));
    const anyT = unixTimes.some(v => v != null);
    if (!anyT) {
      const now = Date.now() / 1000;
      unixTimes = raw.map((_, i) => now + i * DEFAULT_DT);
    } else {
      for (let i = 0; i < unixTimes.length; i++) {
        if (unixTimes[i] == null) {
          if (i > 0 && unixTimes[i-1] != null) unixTimes[i] = unixTimes[i-1] + DEFAULT_DT;
          else {
            const firstIdx = unixTimes.findIndex(v => v != null);
            const base = unixTimes[firstIdx] ?? (Date.now()/1000);
            unixTimes[i] = base + i * DEFAULT_DT;
          }
        }
      }
    }

    const t0 = Math.min(...unixTimes);

    // normalize moves (keep turn)
    moves = raw.map((m, i) => ({
      turn: m.turn ?? null,
      from: String(m.from).toLowerCase(),
      to: String(m.to).toLowerCase(),
      color: m.color,
      piece: m.piece,
      action: m.action,
      iso: m.timestamp ?? null,
      t: Math.max(0, unixTimes[i] - t0)
    })).sort((a,b) => a.t - b.t);

    const maxT = moves.length ? moves[moves.length - 1].t : 0;
    timeSlider.min = 0;
    timeSlider.max = maxT.toFixed(2);
    timeSlider.step = 0.01;

    setStartPosition();
    renderMoveLog(moves);
    renderAtTime(0);

    const firstISO = moves[0]?.iso || '(synthetic)';
    statusEl.textContent = `Loaded ${moves.length} moves. Start: ${firstISO}. Duration: ${maxT.toFixed(2)}s`;
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Failed to load moves.json';
  }
}

init();
