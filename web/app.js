/* Geometry Figure Copilot — chat-first SPA */

const state = {
  currentTikz: "",
  pngUrl: null,
  pending: null,
  busy: false,
  attachment: null,
  selectedExample: null,
  emptyBoardHtml: "",
  examples: [],
};

const $ = (sel) => document.querySelector(sel);
const chatEl = $("#chat");
const messageEl = $("#message");
const badgeEl = $("#badge");
const figureStage = $("#figure-stage");
const boardHost = $("#board-host");
const tikzCode = $("#tikz-code");
const sendBtn = $("#btn-send");

function authHeaders() {
  // Browser Basic auth is sent automatically for same-origin XHR/fetch
  // after the native login dialog. No extra headers needed.
  return {};
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    credentials: "same-origin",
    ...opts,
    headers: { ...(opts.headers || {}), ...authHeaders() },
  });
  if (res.status === 401) {
    throw new Error("Not authenticated — refresh and sign in.");
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      detail = j.detail || j.message || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

function showEmptyChat() {
  chatEl.innerHTML = `
    <div class="empty-chat">
      <strong>Draw a geometry figure</strong>
      Describe a scene, attach a screenshot or PDF, or paste TikZ.
      Then edit by chat or drag points on the Interactive tab.
    </div>`;
}

function appendMsg(role, text, { thinking = false } = {}) {
  const empty = chatEl.querySelector(".empty-chat");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}` + (thinking ? " thinking" : "");
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function finishThinking(content) {
  const last = chatEl.querySelector(".msg.assistant.thinking:last-of-type");
  if (last) {
    last.classList.remove("thinking");
    last.textContent = content;
  } else {
    appendMsg("assistant", content);
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setBadge(text) {
  const clean = (text || "No figure yet").replace(/^\*|\*$/g, "").replace(/`/g, "");
  badgeEl.textContent = clean || "No figure yet";
  badgeEl.title = text || "";
}

function setFigure(pngUrl) {
  state.pngUrl = pngUrl || null;
  if (!pngUrl) {
    figureStage.innerHTML = `<p class="empty-hint">Your figure will appear here.</p>`;
    return;
  }
  figureStage.innerHTML = "";
  const img = document.createElement("img");
  img.src = pngUrl;
  img.alt = "Generated geometry figure";
  figureStage.appendChild(img);
}

function setBoard(html) {
  if (html) {
    boardHost.innerHTML = html;
  } else if (!state.currentTikz) {
    boardHost.innerHTML = state.emptyBoardHtml || `<div style="padding:14px;color:#666">Generate a figure to edit interactively.</div>`;
  }
  // If html is null but we have a figure, leave the live board alone (apply-board case).
}

function setTikz(tikz) {
  state.currentTikz = tikz || "";
  tikzCode.value = state.currentTikz;
}

function applyResult(data, { updateBoard = true } = {}) {
  if (data.badge) setBadge(data.badge);
  else if (data.message) setBadge(data.message);
  if (data.tikz !== undefined && data.tikz !== null && !(data.kept && !data.ok)) {
    if (data.ok || data.tikz) setTikz(data.tikz);
  }
  if (data.png_url) setFigure(data.png_url);
  if (updateBoard && data.board_html) setBoard(data.board_html);
  state.pending = data.pending || null;
}

function setBusy(busy) {
  state.busy = busy;
  sendBtn.disabled = busy;
}

function autosize() {
  messageEl.style.height = "auto";
  messageEl.style.height = Math.min(messageEl.scrollHeight, 140) + "px";
}

function clearAttachment() {
  state.attachment = null;
  $("#file-input").value = "";
  $("#attach-chip").classList.add("hidden");
  $("#attach-name").textContent = "";
}

function setAttachment(file) {
  if (!file) return clearAttachment();
  state.attachment = file;
  $("#attach-name").textContent = file.name;
  $("#attach-chip").classList.remove("hidden");
}

async function sendMessage({ text = null, file = null, userLabel = null } = {}) {
  if (state.busy) return;
  const message = (text !== null ? text : messageEl.value).trim();
  const attachment = file !== null ? file : state.attachment;
  if (!message && !attachment) return;

  let userTurn = message;
  if (attachment) {
    const isPdf = /\.pdf$/i.test(attachment.name || "");
    userTurn = (isPdf ? "📄 (PDF upload)" : "🖼️ (screenshot)") + (message ? ` — ${message}` : "");
  }
  if (userLabel) userTurn = userLabel;

  appendMsg("user", userTurn);
  appendMsg("assistant", "✏️ …drawing…", { thinking: true });
  messageEl.value = "";
  autosize();
  const heldFile = attachment;
  clearAttachment();
  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("message", message);
    fd.append("current_tikz", state.currentTikz || "");
    fd.append("use_specialist", $("#use-specialist").checked ? "true" : "false");
    fd.append("frontier_model", $("#frontier-model").value || "");
    fd.append("pending_json", state.pending ? JSON.stringify(state.pending) : "");
    if (heldFile) fd.append("file", heldFile, heldFile.name);

    const data = await api("/api/chat", { method: "POST", body: fd });
    finishThinking(data.message || "Done.");
    applyResult(data);
  } catch (err) {
    finishThinking(err.message || "Something went wrong — try again.");
  } finally {
    setBusy(false);
  }
}

async function pasteTikz() {
  if (state.busy) return;
  const tikz = $("#paste-tikz").value.trim();
  if (!tikz) return;
  appendMsg("user", "📋 (pasted TikZ)");
  appendMsg("assistant", "✏️ …drawing…", { thinking: true });
  setBusy(true);
  try {
    const data = await api("/api/paste", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tikz,
        frontier_model: $("#frontier-model").value || "",
      }),
    });
    finishThinking(data.message || "Done.");
    applyResult(data);
  } catch (err) {
    finishThinking(err.message || "Something went wrong — try again.");
  } finally {
    setBusy(false);
  }
}

function requestBoardTikz(timeoutMs = 3000) {
  return new Promise((resolve) => {
    const iframe = boardHost.querySelector('iframe[title="interactive geometry editor"]');
    if (!iframe || !iframe.contentWindow) {
      resolve("");
      return;
    }
    let done = false;
    const finish = (t) => {
      if (done) return;
      done = true;
      window.removeEventListener("message", handler);
      resolve(t || "");
    };
    const handler = (e) => {
      if (!e.data || e.data.type !== "geotikz-tikz") return;
      finish(e.data.tikz || "");
    };
    window.addEventListener("message", handler);
    try {
      iframe.contentWindow.postMessage({ type: "geotikz-request-tikz" }, "*");
    } catch (_) {
      finish("");
      return;
    }
    setTimeout(() => finish(""), timeoutMs);
  });
}

async function applyBoard() {
  if (state.busy) return;
  setBusy(true);
  appendMsg("user", "🖐 (apply board edits)");
  appendMsg("assistant", "✏️ …applying…", { thinking: true });
  try {
    const boardTikz = await requestBoardTikz();
    const data = await api("/api/apply-board", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        board_tikz: boardTikz,
        current_tikz: state.currentTikz || "",
      }),
    });
    finishThinking(data.message || "Done.");
    applyResult(data, { updateBoard: false });
  } catch (err) {
    finishThinking(err.message || "Something went wrong — try again.");
  } finally {
    setBusy(false);
  }
}

function resetAll() {
  state.currentTikz = "";
  state.pngUrl = null;
  state.pending = null;
  state.selectedExample = null;
  chatEl.innerHTML = "";
  showEmptyChat();
  setBadge("Started a new figure.");
  setFigure(null);
  setTikz("");
  setBoard(null);
  clearAttachment();
}

function renderExamples(examples) {
  state.examples = examples || [];
  const list = $("#examples-list");
  list.innerHTML = "";
  state.examples.forEach((ex, i) => {
    const li = document.createElement("li");
    li.textContent = ex.label;
    li.dataset.index = String(i);
    if (state.selectedExample && state.selectedExample.prompt === ex.prompt) {
      li.classList.add("selected");
    }
    li.addEventListener("click", () => {
      state.selectedExample = ex;
      [...list.children].forEach((c) => c.classList.remove("selected"));
      li.classList.add("selected");
      $("#btn-remove-example").disabled = !ex.saved;
      closeMenu("examples");
      sendMessage({ text: ex.prompt });
    });
    list.appendChild(li);
  });
}

function openMenu(which) {
  $(`#${which}-backdrop`).classList.remove("hidden");
  $(`#${which}-menu`).classList.remove("hidden");
}
function closeMenu(which) {
  $(`#${which}-backdrop`).classList.add("hidden");
  $(`#${which}-menu`).classList.add("hidden");
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => {
        const on = p.id === `panel-${name}`;
        p.classList.toggle("active", on);
        p.hidden = !on;
      });
    });
  });
}

async function init() {
  showEmptyChat();
  setupTabs();
  setBoard(null);

  try {
    const cfg = await api("/api/config");
    state.emptyBoardHtml = cfg.empty_board_html || "";
    setBoard(null);
    $("#specialist-label").textContent = cfg.specialist_toggle_label || "Use specialist first";
    $("#use-specialist").checked = !!cfg.specialist_default;
    if (!cfg.specialist_available) {
      $("#specialist-field").classList.add("hidden");
    }
    const sel = $("#frontier-model");
    sel.innerHTML = "";
    (cfg.frontier_models || []).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m === cfg.default_frontier_model) opt.selected = true;
      sel.appendChild(opt);
    });
    renderExamples(cfg.examples || []);
  } catch (err) {
    appendMsg("assistant", `Could not load config: ${err.message}`);
  }

  sendBtn.addEventListener("click", () => sendMessage());
  messageEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  messageEl.addEventListener("input", autosize);

  $("#btn-attach").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    setAttachment(f || null);
  });
  $("#btn-clear-attach").addEventListener("click", clearAttachment);

  $("#btn-paste-toggle").addEventListener("click", () => {
    $("#paste-panel").classList.toggle("hidden");
  });
  $("#btn-paste-render").addEventListener("click", pasteTikz);

  $("#btn-new").addEventListener("click", resetAll);
  $("#btn-apply-board").addEventListener("click", applyBoard);
  $("#btn-copy-tikz").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(tikzCode.value || "");
      $("#btn-copy-tikz").textContent = "Copied";
      setTimeout(() => { $("#btn-copy-tikz").textContent = "Copy TikZ"; }, 1200);
    } catch (_) { /* ignore */ }
  });

  $("#btn-examples").addEventListener("click", () => openMenu("examples"));
  $("#btn-settings").addEventListener("click", () => openMenu("settings"));
  $("#btn-examples-close").addEventListener("click", () => closeMenu("examples"));
  $("#btn-settings-close").addEventListener("click", () => closeMenu("settings"));
  $("#examples-backdrop").addEventListener("click", () => closeMenu("examples"));
  $("#settings-backdrop").addEventListener("click", () => closeMenu("settings"));

  $("#btn-save-example").addEventListener("click", async () => {
    const prompt = messageEl.value.trim() || state.selectedExample?.prompt || "";
    // Prefer last user text message from chat if composer empty
    let toSave = prompt;
    if (!toSave) {
      const users = [...chatEl.querySelectorAll(".msg.user")].reverse();
      for (const u of users) {
        const t = u.textContent || "";
        if (t && !t.startsWith("🖼️") && !t.startsWith("📄") && !t.startsWith("📋") && !t.startsWith("🖐")) {
          toSave = t;
          break;
        }
      }
    }
    if (!toSave) {
      appendMsg("assistant", "Type or send a text prompt first, then Save.");
      return;
    }
    try {
      const data = await api("/api/examples", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: toSave }),
      });
      renderExamples(data.examples || []);
      appendMsg("assistant", data.status || "Saved.");
    } catch (err) {
      appendMsg("assistant", err.message);
    }
  });

  $("#btn-remove-example").addEventListener("click", async () => {
    const ex = state.selectedExample;
    if (!ex || !ex.saved) return;
    try {
      const data = await api("/api/examples", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: ex.prompt }),
      });
      state.selectedExample = null;
      $("#btn-remove-example").disabled = true;
      renderExamples(data.examples || []);
      appendMsg("assistant", data.status || "Removed.");
    } catch (err) {
      appendMsg("assistant", err.message);
    }
  });

  // Board iframe "Apply" button posts to parent — mirror Gradio bridge.
  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "geotikz-click-apply") applyBoard();
  });
}

init();
