console.log("app.js loaded");

// URL to moves.json
const FILE_URL = "/moves.json";
const DEFAULT_DT = 2.0;

// Glyphs
const GLYPHS = {
  white: { king: "♔", queen: "♕", rook: "♖", bishop: "♗", knight: "♘", pawn: "♙" },
  black: { king: "♚", queen: "♛", rook: "♜", bishop: "♝", knight: "♞", pawn: "♟" }
};

// Board helpers
const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];

function idx(square) {
  return { col: files.indexOf(square[0]), row: ranks.indexOf(square[1]) };
}

function pos(row, col) {
  return files[col] + ranks[row];
}

// Parse timestamps like "00:05:504"
function parseTimestamp(ts) {
  if (!ts) return NaN;
  const parts = ts.split(":").map(Number);
  if (parts.length !== 3) return NaN;
  const [mm, ss, ms] = parts;
  return mm * 60 + ss + ms / 1000;
}

// Format seconds → mm:ss.ms
function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t % 1) * 1000);
  return `${m}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
}

// DOM
const boardEl = document.getElementById("board");
const coordsLeftEl = document.getElementById("coords-left");
const coordsBottomEl = document.getElementById("coords-bottom");
const timeSlider = document.getElementById("time-slider");
const timeLabel = document.getElementById("time-label");
const btnReset = document.getElementById("btn-reset");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnPlay = document.getElementById("btn-play");
const btnPause = document.getElementById("btn-pause");
const snapMoves = document.getElementById("snap-moves");
const statusEl = document.getElementById("status");
const logList = document.getElementById("move-log");

// State
let squares = [];
let boardState = {};
let moves = [];
let tCurrent = 0;
let timer = null;
let lastFrom = null, lastTo = null;

// Build board
(function buildBoard() {
  ranks.forEach(r => {
    const d = document.createElement("div");
    d.textContent = r;
    coordsLeftEl.appendChild(d);
  });

  files.forEach(f => {
    const d = document.createElement("div");
    d.textContent = f;
    coordsBottomEl.appendChild(d);
  });

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = document.createElement("div");
      sq.className = "square " + ((r + c) % 2 ? "dark" : "light");
      sq.dataset.pos = pos(r, c);
      boardEl.appendChild(sq);
      squares.push(sq);
    }
  }
})();

// Start position
function setStartPosition() {
  boardState = {};

  for (let c = 0; c < 8; c++) {
    boardState[pos(6, c)] = { color: "white", piece: "pawn" };
    boardState[pos(1, c)] = { color: "black", piece: "pawn" };
  }

  const back = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];
  for (let c = 0; c < 8; c++) {
    boardState[pos(7, c)] = { color: "white", piece: back[c] };
    boardState[pos(0, c)] = { color: "black", piece: back[c] };
  }

  lastFrom = lastTo = null;
  renderBoard();
}

// Render board
function renderBoard() {
  squares.forEach(sq => {
    sq.textContent = "";
    sq.classList.remove("last-from", "last-to");
  });

  for (let square in boardState) {
    const p = boardState[square];
    const { row, col } = idx(square);
    squares[row * 8 + col].textContent = GLYPHS[p.color][p.piece];
  }

  if (lastFrom) {
    const f = idx(lastFrom);
    squares[f.row * 8 + f.col].classList.add("last-from");
  }
  if (lastTo) {
    const t = idx(lastTo);
    squares[t.row * 8 + t.col].classList.add("last-to");
  }
}

// Apply a move
function applyMove(mv) {
  const piece = boardState[mv.from];
  if (!piece) return;
  delete boardState[mv.from];
  boardState[mv.to] = piece;
  lastFrom = mv.from;
  lastTo = mv.to;
}

// Render at time
function renderAtTime(t) {
  tCurrent = t;
  timeSlider.value = t;
  timeLabel.textContent = formatTime(t);

  setStartPosition();

  for (let m of moves) {
    if (m.t <= t + 1e-6) applyMove(m);
    else break;
  }

  renderBoard();
  updateLogHighlight();
}

// Find current move index
function currentMoveIndex() {
  let idx = -1;
  for (let i = 0; i < moves.length; i++) {
    if (moves[i].t <= tCurrent) idx = i;
    else break;
  }
  return idx;
}

// Highlight log
function updateLogHighlight() {
  const ci = currentMoveIndex();
  [...logList.children].forEach((li, i) => {
    li.classList.toggle("current", i === ci);
  });
  if (ci >= 0) logList.children[ci].scrollIntoView({ block: "nearest" });
}

// Prev
btnPrev.onclick = () => {
  const ci = currentMoveIndex();
  const target = Math.max(0, ci - 1);
  renderAtTime(moves[target].t);
};

// Next
btnNext.onclick = () => {
  const ci = currentMoveIndex();
  const target = Math.min(moves.length - 1, ci + 1);
  renderAtTime(moves[target].t);
};

// Play
btnPlay.onclick = () => {
  if (timer) return;
  timer = setInterval(() => {
    const maxT = moves[moves.length - 1].t;
    if (tCurrent >= maxT) return btnPause.onclick();
    renderAtTime(tCurrent + 0.1);
  }, 100);
};

// Pause
btnPause.onclick = () => {
  clearInterval(timer);
  timer = null;
};

// Reset
btnReset.onclick = () => renderAtTime(0);

// Slider
timeSlider.oninput = () => {
  const val = parseFloat(timeSlider.value);
  if (snapMoves.checked) {
    let best = moves[0].t;
    let bestd = Math.abs(val - best);
    moves.forEach(m => {
      const d = Math.abs(val - m.t);
      if (d < bestd) {
        best = m.t;
        bestd = d;
      }
    });
    renderAtTime(best);
  } else {
    renderAtTime(val);
  }
};

// Load moves.json
async function init() {
  try {
    const res = await fetch(FILE_URL, { cache: "no-store" });
    const raw = await res.json();

    let times = raw.map(m => parseTimestamp(m.timestamp));

    // Fallback if timestamps missing
    if (times.some(isNaN)) {
      times = raw.map((_, i) => i * DEFAULT_DT);
    }

    moves = raw.map((m, i) => ({
      ...m,
      from: m.from.toLowerCase(),
      to: m.to.toLowerCase(),
      t: times[i] // ⭐ DO NOT NORMALIZE — KEEP REAL TIMESTAMP
    }));

    logList.innerHTML = "";
    moves.forEach((m, i) => {
      const li = document.createElement("li");
      li.className = "move-item";
      li.textContent = `Turn ${m.turn} — ${m.color} ${m.piece}: ${m.from} → ${m.to}`;
      li.onclick = () => renderAtTime(m.t);
      logList.appendChild(li);
    });

    const maxT = Math.max(...moves.map(m => m.t));
    timeSlider.max = maxT;

    renderAtTime(0);
    statusEl.textContent = `Loaded ${moves.length} moves.`;

  } catch (err) {
    console.error(err);
    statusEl.textContent = "ERROR loading moves.json";
  }
}

init();
