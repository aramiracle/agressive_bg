/* ============================
   WEBSOCKET CLIENT
============================ */
const WS_URL = "ws://localhost:8765";
let socket = null;

/* ============================
   CUBE DECISION LISTENER
============================ */
const btnTake = document.getElementById("btnTake");
const btnPass = document.getElementById("btnPass");
const btnDouble = document.getElementById("btnDouble");

btnTake.onclick = () => {
    if (actionPending) return;
    if (send({ type: "take_double" })) actionPending = true;
};

btnPass.onclick = () => {
    if (actionPending) return;
    if (confirm("Are you sure you want to pass? You will lose the current cube value.")) {
        if (send({ type: "refuse_double" })) actionPending = true;
    }
};

btnDouble.onclick = () => {
    if (actionPending || awaitingCube) return;
    if (confirm("Offer to double the stakes?")) {
        if (send({ type: "double" })) actionPending = true;
    }
};
/* ============================
   STATE
============================ */
let turn = 1;
let boardState = new Array(24).fill(0);
let bar = [0, 0];
let off = [0, 0];
let dice = [];
let cubeValue = 1;
let cubeOwner = 0;
let legalMoves = [];
let selectedPoint = null;
let gameMode = "human_vs_ai";
let gameOver = false;
let hasRolledThisTurn = false;  // Track if dice were rolled this turn
let autoplayEnabled = false;
let autoplayInterval = null;
let matchTarget = 7;
let matchScores = { white: 0, black: 0 };
let crawford = false;
let canDouble = false;
let awaitingCube = false;   // a double is pending and the human must respond
let actionPending = false;  // an action was sent, awaiting the server's state reply

/* ============================
   UI ELEMENTS
============================ */
const pipsWhite = document.getElementById("pipsWhite");
const pipsBlack = document.getElementById("pipsBlack");
const status = document.getElementById("status");
const cubeValEl = document.getElementById("cubeValue");
const cubeOwnerEl = document.getElementById("cubeOwner");
const cubeBox = document.getElementById("cubeBox");
const die1 = document.getElementById("die1");
const die2 = document.getElementById("die2");
const die3 = document.getElementById("die3");
const die4 = document.getElementById("die4");
const scoreWhite = document.getElementById("scoreWhite");
const scoreBlack = document.getElementById("scoreBlack");
const matchTargetSelect = document.getElementById("matchTargetSelect");
const crawfordIndicator = document.getElementById("crawfordIndicator");

const topLeft = document.getElementById("topLeft");
const topRight = document.getElementById("topRight");
const bottomLeft = document.getElementById("bottomLeft");
const bottomRight = document.getElementById("bottomRight");

const barBlack = document.getElementById("barBlack");
const barWhite = document.getElementById("barWhite");

const offWhiteEl = document.getElementById("offWhite");
const offBlackEl = document.getElementById("offBlack");
const bearoffWhiteStack = document.getElementById("bearoffWhiteStack");
const bearoffBlackStack = document.getElementById("bearoffBlackStack");
const bearoffWhite = document.getElementById("bearoffWhite");
const bearoffBlack = document.getElementById("bearoffBlack");

const btnRoll = document.getElementById("btnRoll");
const btnEnd = document.getElementById("btnEnd");
const btnNew = document.getElementById("btnNew");
const btnNewMatch = document.getElementById("btnNewMatch");
const btnAI = document.getElementById("btnAI");
const btnAutoplay = document.getElementById("btnAutoplay");
const autoplaySpeed = document.getElementById("autoplaySpeed");
const aiModeSelect = document.getElementById("aiMode");
const connStatus = document.getElementById("connStatus");
const cardWhite = document.getElementById("cardWhite");
const cardBlack = document.getElementById("cardBlack");
const btnLoadModel = document.getElementById("btnLoadModel");
const modelFile = document.getElementById("modelFile");
const modelStatus = document.getElementById("modelStatus");

function setModelStatusState(state) {
  modelStatus.classList.remove("is-ok", "is-error", "is-loading");
  modelStatus.classList.add(state);
}

/* ============================
   SOCKET
============================ */
function connectAI() {
  console.log("Attempting to connect to:", WS_URL);
  
  try {
    socket = new WebSocket(WS_URL);
  } catch (err) {
    console.error("Failed to create WebSocket:", err);
    status.textContent = "WebSocket creation failed: " + err.message;
    setTimeout(connectAI, 2000);
    return;
  }

  socket.onopen = () => {
    console.log("WebSocket connected!");
    connStatus.textContent = "Connected";
    connStatus.className = "connection connected";
    status.textContent = "Connected to Neural AI";
    send({ type: "hello" });
  };

  socket.onmessage = e => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch (err) {
      console.error("Invalid JSON from server:", err);
      return;
    }
    
    if (msg.type === "state") {
      applyState(msg.payload);
    } else if (msg.type === "model_loaded") {
      const modelType = msg.model_type || "unknown";
      const elo = msg.elo ? `ELO: ${msg.elo}` : "";
      const step = msg.step ? `Step: ${msg.step}` : "";
      modelStatus.textContent = `${msg.filename} (${modelType})`;
      setModelStatusState("is-ok");
      modelStatus.title = `${msg.filename}\nType: ${modelType}\n${elo}\n${step}`.trim();
      status.textContent = `Model loaded: ${msg.filename} ${elo}`;
    } else if (msg.type === "model_error") {
      modelStatus.textContent = "Error";
      setModelStatusState("is-error");
      alert("Model load error: " + msg.error);
    } else if (msg.type === "error") {
      console.error("Server error:", msg.error);
      status.textContent = "Error: " + msg.error;
    }
  };

  socket.onclose = (event) => {
    console.log("WebSocket closed:", event.code, event.reason);
    connStatus.textContent = "Disconnected";
    connStatus.className = "connection disconnected";
    status.textContent = "AI Disconnected - Reconnecting...";
    // Server is gone; any in-flight action or cube prompt is dead. The fresh
    // session after reconnect will send its own state.
    actionPending = false;
    awaitingCube = false;
    document.getElementById("cubeOverlay").style.display = "none";
    updateButtons();
    setTimeout(connectAI, 2000);
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
    connStatus.textContent = "Error";
    connStatus.className = "connection disconnected";
  };
}

function send(data) {
  if (socket && socket.readyState === 1) {
    socket.send(JSON.stringify(data));
    return true;
  }
  return false;
}

/* ============================
   RENDERING
============================ */
function makePoint(idx, down) {
  const p = document.createElement("div");
  p.className = "point " + (down ? "down" : "");
  p.dataset.index = idx;

  const tri = document.createElement("div");
  tri.className = "triangle " + (idx % 2 === 0 ? "light" : "dark");

  const stack = document.createElement("div");
  stack.className = "stack";

  const val = boardState[idx];
  const count = Math.abs(val);
  const maxShow = 5;
  
  for (let i = 0; i < Math.min(count, maxShow); i++) {
    const c = document.createElement("div");
    c.className = "checker " + (val > 0 ? "white" : "black");
    stack.appendChild(c);
  }
  
  if (count > maxShow) {
    const countEl = document.createElement("div");
    countEl.className = "checker-count";
    countEl.textContent = count;
    stack.appendChild(countEl);
  }

  p.appendChild(tri);
  p.appendChild(stack);

  // Highlight logic
  if (selectedPoint !== null) {
    const isTarget = legalMoves.some(m => m[0] === selectedPoint && m[1] === idx);
    if (isTarget) p.classList.add("highlight");
    if (selectedPoint === idx) p.classList.add("selected");
  } else {
    const canSelect = legalMoves.some(m => m[0] === idx);
    if (canSelect && isHumanTurn()) p.classList.add("highlight");
  }

  p.onclick = () => handlePointClick(idx);
  return p;
}

function renderBoard() {
  topLeft.innerHTML = "";
  topRight.innerHTML = "";
  bottomLeft.innerHTML = "";
  bottomRight.innerHTML = "";

  // Top row: points 13-18 (left), 19-24 (right)
  for (let i = 12; i < 18; i++) topLeft.appendChild(makePoint(i, false));
  for (let i = 18; i < 24; i++) topRight.appendChild(makePoint(i, false));
  
  // Bottom row: points 12-7 (left), 6-1 (right)
  for (let i = 11; i >= 6; i--) bottomLeft.appendChild(makePoint(i, true));
  for (let i = 5; i >= 0; i--) bottomRight.appendChild(makePoint(i, true));

  // Point numbers (1-indexed display)
  const midL = document.getElementById("midLeft");
  const midR = document.getElementById("midRight");
  midL.innerHTML = "";
  midR.innerHTML = "";
  for (let i = 12; i < 18; i++) {
    const s = document.createElement("span");
    s.textContent = i + 1;
    midL.appendChild(s);
  }
  for (let i = 18; i < 24; i++) {
    const s = document.createElement("span");
    s.textContent = i + 1;
    midR.appendChild(s);
  }

  renderBar();
  renderBearoff();
  updateButtons();
}

function renderBar() {
  // White bar (player 1)
  barWhite.innerHTML = "";
  for (let i = 0; i < Math.min(bar[0], 5); i++) {
    const c = document.createElement("div");
    c.className = "bar-checker white";
    barWhite.appendChild(c);
  }
  if (bar[0] > 0) {
    const count = document.createElement("div");
    count.className = "bar-count";
    count.textContent = bar[0];
    barWhite.appendChild(count);
  }

  // Black bar (player -1)
  barBlack.innerHTML = "";
  for (let i = 0; i < Math.min(bar[1], 5); i++) {
    const c = document.createElement("div");
    c.className = "bar-checker black";
    barBlack.appendChild(c);
  }
  if (bar[1] > 0) {
    const count = document.createElement("div");
    count.className = "bar-count";
    count.textContent = bar[1];
    barBlack.appendChild(count);
  }

  // Highlight bar if can move from it
  barWhite.classList.remove("highlight", "selected");
  barBlack.classList.remove("highlight", "selected");
  
  if (selectedPoint === "bar") {
    if (turn === 1) barWhite.classList.add("selected");
    else barBlack.classList.add("selected");
  } else if (isHumanTurn()) {
    const canMoveFromBar = legalMoves.some(m => m[0] === "bar");
    if (canMoveFromBar) {
      if (turn === 1) barWhite.classList.add("highlight");
      else barBlack.classList.add("highlight");
    }
  }
}

function renderBearoff() {
  // White bearoff
  bearoffWhiteStack.innerHTML = "";
  for (let i = 0; i < Math.min(off[0], 10); i++) {
    const c = document.createElement("div");
    c.className = "bearoff-checker white";
    bearoffWhiteStack.appendChild(c);
  }
  document.getElementById("bearoffWhiteCount").textContent = off[0];

  // Black bearoff
  bearoffBlackStack.innerHTML = "";
  for (let i = 0; i < Math.min(off[1], 10); i++) {
    const c = document.createElement("div");
    c.className = "bearoff-checker black";
    bearoffBlackStack.appendChild(c);
  }
  document.getElementById("bearoffBlackCount").textContent = off[1];

  // Off display in info panel
  offWhiteEl.textContent = off[0];
  offBlackEl.textContent = off[1];

  // Highlight bearoff if it's a valid target
  bearoffWhite.classList.remove("highlight");
  bearoffBlack.classList.remove("highlight");
  
  if (selectedPoint !== null && isHumanTurn()) {
    const canBearoff = legalMoves.some(m => m[0] === selectedPoint && m[1] === "off");
    if (canBearoff) {
      if (turn === 1) bearoffWhite.classList.add("highlight");
      else bearoffBlack.classList.add("highlight");
    }
  }
}

function renderDice() {
  const diceEls = [die1, die2, die3, die4];
  
  // Hide all first
  diceEls.forEach(d => {
    d.style.display = "none";
    d.classList.remove("used");
  });

  if (!dice || dice.length === 0) {
    die1.style.display = "flex";
    die2.style.display = "flex";
    die1.textContent = "-";
    die2.textContent = "-";
    return;
  }

  // Show dice
  for (let i = 0; i < dice.length; i++) {
    diceEls[i].style.display = "flex";
    diceEls[i].textContent = dice[i];
  }
}

/* ============================
   STATE SYNC
============================ */
function applyState(s) {
  const wasGameOver = gameOver;
  const oldCubeValue = cubeValue;
  const isWaitingForCube = s.waiting_for_cube || false;
  const cubeOverlay = document.getElementById("cubeOverlay");
  const cubePrompt = document.getElementById("cubePrompt");
  
  boardState = s.board;
  bar = s.bar;
  off = s.off || [0, 0];
  turn = s.turn;
  dice = s.dice;
  cubeValue = s.cube_value;
  cubeOwner = s.cube_owner;
  canDouble = s.can_double || false;
  legalMoves = s.legal_moves || [];
  gameOver = s.game_over || false;
  gameMode = s.mode || gameMode;
  hasRolledThisTurn = s.has_rolled || false;
  awaitingCube = isWaitingForCube;
  actionPending = false;  // the server answered; UI actions are unlocked again
  
  // Match info
  matchTarget = s.match_target || 7;
  matchScores = s.match_scores || { white: 0, black: 0 };
  crawford = s.crawford || false;
  
  // Check if match is won - stop autoplay
  const matchWon = matchScores.white >= matchTarget || matchScores.black >= matchTarget;
  if (matchWon && autoplayEnabled) {
    stopAutoplay();
  }

  // Clear selection when state changes
  selectedPoint = null;
  if (isWaitingForCube) {
    if (isHumanTurn()) {
      cubeOverlay.style.display = "flex";
      cubePrompt.textContent = `${turn === 1 ? "White" : "Black"}: Take or Pass?`;
    } else {
      cubeOverlay.style.display = "none";
    }
    // Roll / Double / End are disabled via updateButtons() (awaitingCube).
  } else {
    cubeOverlay.style.display = "none";
  }

  // Update cube display
  cubeValEl.textContent = cubeValue;
  cubeOwnerEl.textContent = cubeOwner === 0 ? "Center" :
    cubeOwner === 1 ? "White" : "Black";
  
  // Animate cube if doubled
  if (cubeValue > oldCubeValue) {
    cubeBox.classList.add('doubled');
    setTimeout(() => cubeBox.classList.remove('doubled'), 500);
  }

  if (s.pips) {
    pipsWhite.textContent = s.pips.white;
    pipsBlack.textContent = s.pips.black;
  }
  
  // Update match score display
  scoreWhite.textContent = matchScores.white;
  scoreBlack.textContent = matchScores.black;
  cardWhite.classList.toggle("is-turn", !gameOver && turn === 1);
  cardBlack.classList.toggle("is-turn", !gameOver && turn === -1);
  matchTargetSelect.value = matchTarget;
  
  // Crawford indicator
  crawfordIndicator.style.display = crawford ? "block" : "none";

  renderDice();
  status.textContent = s.status || `${turn === 1 ? "White" : "Black"}'s turn`;
  
  // Update mode selector
  aiModeSelect.value = gameMode;
  
  // Update model status
  if (s.model_loaded && s.model_filename) {
    const modelType = s.model_type || "unknown";
    modelStatus.textContent = `${s.model_filename} (${modelType})`;
    setModelStatusState("is-ok");
    modelStatus.title = `${s.model_filename}\nType: ${modelType}`;
  } else if (!s.model_loaded) {
    const modelType = s.model_type || "unknown";
    modelStatus.textContent = `No model (${modelType})`;
    setModelStatusState("is-error");
    modelStatus.title = `No model loaded\nExpected type: ${modelType}`;
  }
  
  renderBoard();
}

/* ============================
   GAME LOGIC
============================ */
function isHumanTurn() {
  if (gameOver) return false;
  if (gameMode === "human_vs_ai") return turn === 1;
  if (gameMode === "ai_vs_human") return turn === -1;
  return false; // ai_vs_ai
}

function handlePointClick(idx) {
  if (!isHumanTurn() || dice.length === 0 || awaitingCube || actionPending) return;

  // If clicking on a highlighted target
  if (selectedPoint !== null) {
    const isValidTarget = legalMoves.some(m => m[0] === selectedPoint && m[1] === idx);
    if (isValidTarget) {
      const from = selectedPoint;
      selectedPoint = null;
      if (send({ type: "move", from: from, to: idx })) actionPending = true;
      renderBoard();
      return;
    }
  }

  // Check if this point can be selected as source
  const canSelect = legalMoves.some(m => m[0] === idx);
  if (canSelect) {
    selectedPoint = idx;
    renderBoard();
  } else {
    selectedPoint = null;
    renderBoard();
  }
}

function handleBarClick(player) {
  if (!isHumanTurn() || dice.length === 0 || awaitingCube || actionPending) return;
  if (player !== turn) return;

  const canMoveFromBar = legalMoves.some(m => m[0] === "bar");
  if (canMoveFromBar) {
    selectedPoint = "bar";
    renderBoard();
  }
}

function handleBearoffClick(player) {
  if (!isHumanTurn() || selectedPoint === null || awaitingCube || actionPending) return;
  if (player !== turn) return;
  
  const canBearoff = legalMoves.some(m => m[0] === selectedPoint && m[1] === "off");
  if (canBearoff) {
    const from = selectedPoint;
    selectedPoint = null;
    if (send({ type: "move", from: from, to: "off" })) actionPending = true;
    renderBoard();
  }
}

function updateButtons() {
  const human = isHumanTurn();
  const hasDice = dice && dice.length > 0;
  const hasLegalMoves = legalMoves && legalMoves.length > 0;

  // While a double is pending (or a reply from the server is in flight) no
  // other action may be taken, even if the overlay only covers part of the page.
  const locked = gameOver || awaitingCube || actionPending;

  // Roll: enabled when human's turn, no dice yet, haven't rolled this turn
  btnRoll.disabled = locked || !human || hasRolledThisTurn;
  
  // Double: enabled before rolling, when can_double is true (respects Crawford)
  btnDouble.disabled = locked || !human || !canDouble;
  
  // End turn: enabled after rolling, when either:
  // - No dice left (all used), OR
  // - No legal moves available (blocked)
  const canEndTurn = human && !locked && hasRolledThisTurn && (!hasDice || !hasLegalMoves);
  btnEnd.disabled = !canEndTurn;
  
  // Show AI buttons only for ai_vs_ai mode
  const isAiVsAi = gameMode === "ai_vs_ai";
  btnAI.style.display = isAiVsAi ? "inline-block" : "none";
  btnAutoplay.style.display = isAiVsAi ? "inline-block" : "none";
  autoplaySpeed.style.display = isAiVsAi ? "inline-block" : "none";
  
  // Update autoplay button text
  btnAutoplay.textContent = autoplayEnabled ? "⏸ Stop" : "▶ Autoplay";
  btnAutoplay.classList.toggle("btn-new", autoplayEnabled);
  btnAutoplay.classList.toggle("btn-roll", !autoplayEnabled);
}

/* ============================
   EVENTS
============================ */
btnNew.onclick = () => {
  stopAutoplay();
  send({ type: "new_game" });
};

btnNewMatch.onclick = () => {
  stopAutoplay();
  const target = parseInt(matchTargetSelect.value);
  if (confirm(`Start a new match to ${target} points? This will reset all scores.`)) {
    send({ type: "new_match", target: target });
  }
};

matchTargetSelect.onchange = () => {
  // When match target changes, offer to start new match
  const target = parseInt(matchTargetSelect.value);
  if (confirm(`Change match target to ${target} points? This will start a new match.`)) {
    stopAutoplay();
    send({ type: "new_match", target: target });
  } else {
    // Revert to current value
    matchTargetSelect.value = matchTarget;
  }
};
btnRoll.onclick = () => {
  if (actionPending || awaitingCube) return;
  if (send({ type: "roll" })) actionPending = true;
};
btnEnd.onclick = () => {
  if (actionPending || awaitingCube) return;
  if (send({ type: "end_turn" })) actionPending = true;
};
// btnDouble.onclick is set above with confirmation dialog
btnAI.onclick = () => {
  if (actionPending || awaitingCube) return;
  if (send({ type: "ai_play" })) actionPending = true;
};

// Autoplay functions
function startAutoplay() {
  if (autoplayEnabled) return;
  autoplayEnabled = true;
  updateButtons();
  
  const speed = parseFloat(autoplaySpeed.value) * 1000;
  autoplayInterval = setInterval(() => {
    if (gameMode !== "ai_vs_ai") {
      stopAutoplay();
      return;
    }
    
    // Check if match is won
    const matchWon = matchScores.white >= matchTarget || matchScores.black >= matchTarget;
    if (matchWon) {
      stopAutoplay();
      return;
    }
    
    // If game over but match not won, start new game
    if (gameOver) {
      send({ type: "new_game" });
      return;
    }
    
    send({ type: "ai_play" });
  }, speed);
  
  // Trigger first move immediately
  if (!gameOver && gameMode === "ai_vs_ai") {
    send({ type: "ai_play" });
  }
}

function stopAutoplay() {
  autoplayEnabled = false;
  if (autoplayInterval) {
    clearInterval(autoplayInterval);
    autoplayInterval = null;
  }
  updateButtons();
}

btnAutoplay.onclick = () => {
  if (autoplayEnabled) {
    stopAutoplay();
  } else {
    startAutoplay();
  }
};

autoplaySpeed.onchange = () => {
  if (autoplayEnabled) {
    stopAutoplay();
    startAutoplay();
  }
};

aiModeSelect.onchange = () => {
  stopAutoplay();
  send({ type: "set_mode", mode: aiModeSelect.value });
};

// Model loading

btnLoadModel.onclick = () => modelFile.click();

modelFile.onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  modelStatus.textContent = "Loading...";
  setModelStatusState("is-loading");
  
  // Read file as base64
  const reader = new FileReader();
  reader.onload = () => {
    const base64 = reader.result.split(',')[1];
    send({ type: "load_model", filename: file.name, data: base64 });
  };
  reader.onerror = () => {
    modelStatus.textContent = "Read error";
    setModelStatusState("is-error");
  };
  reader.readAsDataURL(file);
};

barWhite.onclick = () => handleBarClick(1);
barBlack.onclick = () => handleBarClick(-1);
bearoffWhite.onclick = () => handleBearoffClick(1);
bearoffBlack.onclick = () => handleBearoffClick(-1);

// Keyboard shortcuts
document.addEventListener("keydown", e => {
  if (e.key === "r" || e.key === "R") btnRoll.click();
  if (e.key === "e" || e.key === "E") btnEnd.click();
  if (e.key === "n" || e.key === "N") btnNew.click();
  if (e.key === " " && gameMode === "ai_vs_ai") {
    e.preventDefault();
    btnAutoplay.click();
  }
  if (e.key === "s" || e.key === "S") {
    if (gameMode === "ai_vs_ai") btnAI.click();
  }
  if (e.key === "Escape") {
    stopAutoplay();
    selectedPoint = null;
    renderBoard();
  }
});

/* ============================
   START
============================ */
connectAI();
