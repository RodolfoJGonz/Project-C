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

// Parse timestamps like "00:00:500"
function parseTimestamp(ts) {
  if (!ts) return NaN;
  const parts = ts.split(":").map(Number);
  if (parts.length !== 3) return NaN;
  const [mm, ss, ms] = parts;
  return mm * 60 + ss + ms / 1000;
}

// Format for label
function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t % 1) * 1000);
  return `${m}:${s.toString().padStart(2,"0")}.${ms.toString().padStart(3,"0")}`;
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

const video = document.getElementById("chess-video");

// State
let squares = [];
let boardState = {};
let moves = [];
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

// Apply move
function applyMove(mv) {
  const piece = boardState[mv.from];
  if (!piece) return;
  delete boardState[mv.from];
  boardState[mv.to] = piece;
  lastFrom = mv.from;
  lastTo = mv.to;
}

// Render at given time (video controls this)
function renderAtTime(t) {
  timeSlider.value = t;
  timeLabel.textContent = formatTime(t);

  setStartPosition();

  for (let m of moves) {
    if (m.t <= t + 1e-6) applyMove(m);
    else break;
  }

  renderBoard();
  updateLogHighlight(t);
}

// Highlight log
function updateLogHighlight(t) {
  let idx = -1;
  for (let i = 0; i < moves.length; i++) {
    if (moves[i].t <= t) idx = i;
  }

  [...logList.children].forEach((li, i) => {
    li.classList.toggle("current", i === idx);
  });

  if (idx >= 0) logList.children[idx].scrollIntoView({ block: "nearest" });
}

// VIDEO → BOARD sync
video.addEventListener("timeupdate", () => {
  renderAtTime(video.currentTime);
});

// PLAY BUTTON sync
btnPlay.onclick = () => video.play();
btnPause.onclick = () => video.pause();

// RESET sync
btnReset.onclick = () => {
  video.pause();
  video.currentTime = 0;
  renderAtTime(0);
};

// NEXT / PREV sync
btnNext.onclick = () => {
  const t = video.currentTime;
  let idx = moves.findIndex(m => m.t > t);
  if (idx === -1) idx = moves.length - 1;
  video.currentTime = moves[idx].t;
};

btnPrev.onclick = () => {
  const t = video.currentTime;
  let idx = moves.findIndex(m => m.t >= t) - 1;
  if (idx < 0) idx = 0;
  video.currentTime = moves[idx].t;
};

// Slider manually controlling video
timeSlider.oninput = () => {
  const val = parseFloat(timeSlider.value);
  video.currentTime = val;
};

// Load moves.json
async function init() {
  try {
    const res = await fetch(FILE_URL, { cache: "no-store" });
    const raw = await res.json();

    let times = raw.map(m => parseTimestamp(m.timestamp));
    if (times.some(isNaN)) times = raw.map((_, i) => i * DEFAULT_DT);

    moves = raw.map((m, i) => ({
      ...m,
      from: m.from.toLowerCase(),
      to: m.to.toLowerCase(),
      t: times[i]
    }));

    logList.innerHTML = "";
    moves.forEach((m, i) => {
      const li = document.createElement("li");
      li.textContent = `Turn ${m.turn} — ${m.color} ${m.piece}: ${m.from} → ${m.to}`;
      li.onclick = () => (video.currentTime = m.t);
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
