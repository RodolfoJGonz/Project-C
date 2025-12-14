const express = require("express");
const path = require("path");

const app = express();
const PORT = 8082;

// Serve static files
app.use(express.static(path.join(__dirname, "public")));

// View engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Home page
app.get("/", (req, res) => {
  res.render("index");
});

// Serve moves.json
app.get("/moves.json", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "moves.json"));
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
