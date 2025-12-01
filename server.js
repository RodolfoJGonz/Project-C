const express = require('express');
const fs = require('fs');
const path = require('path');
const sqlite3 = require('sqlite3');
const { open } = require('sqlite');

const app = express();
const PORT = 8082;

app.use(express.json());

const publicDir = path.join(__dirname); // where styles.css, index.js, moves.json live
app.use(express.static(publicDir));

// Set up EJS views properly
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views')); // ✅ points to your /views folder


let db;
let sim = { list: [], idx: 0, timer: null };

(async () => {
  db = await open({ filename: 'chess.db', driver: sqlite3.Database });

  await db.exec(`
    CREATE TABLE IF NOT EXISTS moves (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      turn INTEGER,
      color TEXT,
      piece TEXT,
      from_sq TEXT,
      to_sq TEXT,
      action TEXT,
      timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    );
  `);

  // main page
  app.get('/', (_req, res) => res.render('index'));

  // API: list moves
  app.get('/api/moves', async (req, res) => {
    const limit = parseInt(req.query.limit || '200', 10);
    const rows = await db.all(`
      SELECT id, turn, color, piece, from_sq, to_sq, action, timestamp
      FROM moves ORDER BY id ASC
    `);
    res.json({ moves: rows.slice(-limit) });
  });

  // API: add a move
  app.post('/api/moves', async (req, res) => {
    const m = req.body || {};
    const from_sq = m.from_sq || m.from;
    const to_sq   = m.to_sq   || m.to;

    if (!m.turn || !m.color || !m.piece || !from_sq || !to_sq) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const action = m.action || 'move';
    const ts = m.timestamp || new Date().toISOString();

    const r = await db.run(
      `INSERT INTO moves (turn,color,piece,from_sq,to_sq,action,timestamp)
       VALUES (?,?,?,?,?,?,?)`,
      [m.turn, m.color, m.piece, from_sq, to_sq, action, ts]
    );
    res.json({ status: 'ok', id: r.lastID });
  });

  // --- simple simulator using moves.json in project root ---
  const MOVES_JSON = path.join(__dirname, 'moves.json');

  function loadSim() {
    try {
      const txt = fs.readFileSync(MOVES_JSON, 'utf8');
      const arr = JSON.parse(txt);
      sim.list = (Array.isArray(arr) ? arr : []).map(x => ({
        turn: x.turn,
        color: x.color,
        piece: x.piece || x.piece_type,
        from_sq: x.from_sq || x.from,
        to_sq: x.to_sq || x.to,
        action: x.action || 'move',
        timestamp: x.timestamp || new Date().toISOString()
      })).filter(x => x.turn && x.color && x.piece && x.from_sq && x.to_sq);
      sim.idx = 0;
    } catch {
      sim.list = []; sim.idx = 0;
    }
  }

  async function insertNext() {
    if (sim.idx >= sim.list.length) return false;
    const m = sim.list[sim.idx++];
    await db.run(
      `INSERT INTO moves (turn,color,piece,from_sq,to_sq,action,timestamp)
       VALUES (?,?,?,?,?,?,?)`,
      [m.turn, m.color, m.piece, m.from_sq, m.to_sq, m.action, m.timestamp]
    );
    return true;
  }

  app.post('/api/sim/reset', async (_req, res) => {
    await db.run('DELETE FROM moves');
    loadSim();
    res.json({ status: 'reset', total: sim.list.length });
  });

  app.post('/api/sim/next', async (_req, res) => {
    if (sim.list.length === 0) loadSim();
    const ok = await insertNext();
    res.json({ status: ok ? 'inserted' : 'done', idx: sim.idx, total: sim.list.length });
  });

  app.post('/api/sim/play', async (req, res) => {
    const ms = Math.max(800, Math.min(5000, parseInt(req.body?.interval || '1500', 10)));
    if (sim.timer) return res.json({ status: 'already-playing' });
    if (sim.list.length === 0) loadSim();
    sim.timer = setInterval(async () => {
      const ok = await insertNext();
      if (!ok) { clearInterval(sim.timer); sim.timer = null; }
    }, ms);
    res.json({ status: 'playing', interval: ms });
  });

  app.post('/api/sim/stop', (_req, res) => {
    if (sim.timer) clearInterval(sim.timer);
    sim.timer = null;
    res.json({ status: 'stopped' });
  });

  app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
})();
