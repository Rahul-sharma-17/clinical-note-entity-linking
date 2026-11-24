/* ══════════════════════════════════════════════════════════════════════════════
   Clinical Note Review — Interactive Review Logic
   No backend changes — only UI rendering and interaction handling
   ══════════════════════════════════════════════════════════════════════════════ */

const selection = document.querySelector("#note-select");
const form = document.querySelector("#custom-form");
const customText = document.querySelector("#custom-text");
const status = document.querySelector("#request-status");
const noteText = document.querySelector("#note-text");
const reviewRows = document.querySelector("#review-rows");
const customToggle = document.querySelector("#custom-toggle");
const counts = Object.fromEntries(["total", "approved", "rejected", "pending"].map(
  (name) => [name, document.querySelector("#" + name + "-count")],
));
const empty = document.querySelector("#review-empty");
const csvButton = document.querySelector("#download-csv");
const jsonButton = document.querySelector("#download-json");
const confidenceFilter = document.querySelector("#confidence-filter");
const metricsDialog = document.querySelector("#metrics-dialog");
const metricsStatus = document.querySelector("#metrics-status");
const metricsTable = document.querySelector("#metrics-table");
const metricsRows = document.querySelector("#metrics-rows");
const notes = new Map();
let lastValidation = "";
let active = null;
let inFlight = null;
let mentionViews = [];

document.addEventListener("pointerdown", () => { document.body.dataset.input = "pointer"; }, true);
document.addEventListener("keydown", () => { document.body.dataset.input = "keyboard"; }, true);

async function readJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error("Request failed (" + response.status + ").");
  return response.json();
}

function showStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}

function updateReviewCount() {
  const total = active?.mentions.length ?? 0;
  const reviewed = active?.decisions.size ?? 0;
  const decisions = active ? [...active.decisions.values()] : [];
  const approved = decisions.filter((decision) => decision === "approved").length;
  document.querySelector("#review-progress").textContent = reviewed + " / " + total + " Reviewed";
  document.querySelector("#all-total").textContent = total;
  document.querySelector(".source-label").textContent = active?.note_id ? "Source Text · Case " + active.note_id : "Source Text";

  counts.total.textContent = total;
  counts.approved.textContent = approved;
  counts.rejected.textContent = reviewed - approved;
  counts.pending.textContent = total - reviewed;

  const meter = document.querySelector("#completion-meter");
  if (meter) {
    meter.max = total || 1;
    meter.value = reviewed;
  }
  csvButton.disabled = jsonButton.disabled = reviewed === 0;
}

const search = document.querySelector("#mention-search");
search.addEventListener("input", applyConfidenceFilter);
const allFilter = document.querySelector("#all-filter");
const pendingFilter = document.querySelector("#pending-filter");
function setFilter(mode) {
  pendingFilter.setAttribute("aria-pressed", String(mode === "pending"));
  confidenceFilter.checked = mode === "low";
  allFilter.setAttribute("aria-pressed", String(mode === "all"));
  applyConfidenceFilter();
}
allFilter.onclick = () => setFilter("all");
pendingFilter.onclick = () => setFilter("pending");
function applyConfidenceFilter() {
  const query = search.value.trim().toLowerCase();
  let visible = 0;
  for (const [index, {hit, row, mark, noteLink}] of mentionViews.entries()) {
    const low = hit.confidence != null && hit.confidence < 0.80;
    const matches = [hit.text, hit.concept_name, hit.concept_id].some(value => String(value ?? "").toLowerCase().includes(query));
    row.hidden = (pendingFilter.getAttribute("aria-pressed") === "true" && active.decisions.has(index)) || !matches || (confidenceFilter.checked && !low);
    noteLink.disabled = row.hidden;
    mark.classList.toggle("is-filtered", row.hidden);
    if (row.hidden) {
      mark.classList.remove("is-current");
      row.classList.remove("is-current");
    } else {
      visible++;
    }
  }
  empty.hidden = visible > 0;
  if (mentionViews.length) empty.textContent = query || !confidenceFilter.checked ? "No matching mentions." : "No mentions below 0.80 confidence.";
}

confidenceFilter.onchange = () => setFilter(confidenceFilter.checked ? "low" : "all");

function render(note) {
  closeInfo();
  active = note;
  mentionViews = [];
  noteText.replaceChildren();
  reviewRows.replaceChildren();
  noteText.scrollTop = 0;
  reviewRows.closest(".table-wrap").scrollTop = 0;
  noteText.classList.toggle("empty", !note?.text);
  empty.hidden = Boolean(note?.mentions.length);
  empty.textContent = note ? "No mentions detected." : "Mention predictions will appear here.";
  updateReviewCount();

  if (!note?.text) {
    noteText.textContent = note ? "This note is empty." : "Select or paste a note to begin.";
    return;
  }

  // Python offsets count Unicode code points. JavaScript string slices use UTF-16.
  const characters = Array.from(note.text);
  let cursor = 0;
  let currentPair;
  for (const [index, hit] of note.mentions.entries()) {
    noteText.append(characters.slice(cursor, hit.start).join(""));
    const mark = document.createElement("mark");
    const noteLink = document.createElement("button");
    noteLink.type = "button";
    noteLink.textContent = hit.text;
    noteLink.setAttribute("aria-label", "Show review row for " + hit.text);
    noteLink.dataset.number = index + 1;
    mark.append(noteLink);
    noteText.append(mark);
    cursor = hit.end;

    const row = reviewRows.insertRow();
    row.setAttribute("role", "row");

    const rowLink = document.createElement("button");
    rowLink.dataset.number = index + 1;
    rowLink.type = "button";
    rowLink.className = "mention-link";
    rowLink.textContent = hit.text;
    rowLink.setAttribute("aria-label", "Show note mention for " + hit.text);
    row.insertCell().append(rowLink);
    const conceptId = row.insertCell();
    const termCell = row.insertCell();
    const confidence = row.insertCell();
    mentionViews.push({hit, row, mark, noteLink});

    function updateConceptCells() {
      const badge = document.createElement("span");
      badge.className = "code-tag";
      badge.textContent = hit.concept_id ?? "—";
      conceptId.replaceChildren(badge);
      noteLink.dataset.code = hit.concept_id ?? "";
      confidence.style.setProperty("--score", (100 * (hit.confidence ?? 0)) + "%");
      confidence.textContent = hit.confidence == null ? "—" : Math.round(hit.confidence * 100) + "%";
      const low = hit.confidence != null && hit.confidence < 0.70;
      confidence.classList.toggle("confidence-low", low);
      confidence.title = low ? "Low confidence" : "";
    }

    if (hit.candidates?.length) {
      const candidateSelect = document.createElement("select");
      candidateSelect.className = "candidate-select";
      candidateSelect.setAttribute("aria-label", "Preferred term for " + hit.text + " at character " + hit.start);
      for (const candidate of hit.candidates) {
        const label = candidate.preferred_term + " (" + candidate.confidence.toFixed(2) + ")";
        candidateSelect.add(new Option(label, candidate.concept_id));
      }
      candidateSelect.value = hit.concept_id;
      candidateSelect.title = candidateSelect.selectedOptions[0].text;
      candidateSelect.addEventListener("change", () => {
        const candidate = hit.candidates[candidateSelect.selectedIndex];
        hit.concept_id = candidate.concept_id;
        hit.concept_name = candidate.preferred_term;
        hit.confidence = candidate.confidence;
        candidateSelect.title = candidateSelect.selectedOptions[0].text;
        updateConceptCells();
        applyConfidenceFilter();
        if (row.hidden) confidenceFilter.focus();
      });
      termCell.append(candidateSelect);
    } else {
      termCell.textContent = hit.concept_name ?? "Unlinked";
    }
    updateConceptCells();

    function reveal(target, focusTarget) {
      currentPair?.forEach((element) => element.classList.remove("is-current"));
      currentPair = [mark, row];
      currentPair.forEach((element) => element.classList.add("is-current"));
      focusTarget.focus({preventScroll: true});
      target.scrollIntoView({
        behavior: "instant",
        block: "nearest",
        inline: "nearest",
      });
    }

    function inspect() { showInfo(noteLink, hit, row); }
    noteLink.addEventListener("pointerenter", event => {
      if (event.pointerType === "mouse") inspect();
    });
    noteLink.addEventListener("pointerleave", scheduleInfoClose);
    noteLink.addEventListener("focus", inspect);
    noteLink.addEventListener("focusout", dismissOnFocusExit);
    noteLink.addEventListener("click", event => {
      reveal(row, rowLink);
      if (event.detail > 0) inspect();
    });
    row.addEventListener("click", (event) => {
      if (!event.target.closest(".choices, select")) reveal(mark, noteLink);
    });
    const controls = document.createElement("div");
    controls.className = "choices";
    controls.setAttribute("role", "group");
    controls.setAttribute("aria-label", "Review " + hit.text + " at character " + hit.start);
    row.insertCell().append(controls);

    const buttons = new Map();
    function reflectDecision() {
      const decision = note.decisions.get(index);
      mark.classList.toggle("approved", decision === "approved");
      mark.classList.toggle("rejected", decision === "rejected");
      for (const [value, button] of buttons) {
        button.setAttribute("aria-pressed", String(value === decision));
      }
    }
    for (const [label, decision] of [["Approve", "approved"], ["Reject", "rejected"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = decision;
      button.textContent = label;
      button.addEventListener("click", () => {
        if (note.decisions.get(index) === decision) note.decisions.delete(index);
        else note.decisions.set(index, decision);
        reflectDecision();
        updateReviewCount();
        applyConfidenceFilter();
        closeInfo();
      });
      buttons.set(decision, button);
      controls.append(button);
    }
    reflectDecision();
    const labels = ["Mention", "SNOMED CT ID", "Preferred Term", "Conf.", "Decision"];
    for (const [column, cell] of [...row.cells].entries()) {
      cell.dataset.label = labels[column];
      cell.setAttribute("role", "cell");
    }
  }
  noteText.append(characters.slice(cursor).join(""));
  applyConfidenceFilter();
}

const info = document.querySelector("#mention-info");
let infoRow = null;
let infoAnchor = null;
let infoCloseTimer;

function scheduleInfoClose(event) {
  if (event.pointerType !== "mouse" || !infoAnchor) return;
  if (event.currentTarget !== info && event.currentTarget !== infoAnchor) return;
  clearTimeout(infoCloseTimer);
  // Allow the pointer to cross the small gap between the mention and its actions.
  infoCloseTimer = setTimeout(() => {
    if (!info.contains(document.activeElement)) closeInfo();
  }, 160);
}

function dismissOnFocusExit(event) {
  if (event.currentTarget !== infoAnchor && event.currentTarget !== info) return;
  if (event.relatedTarget !== infoAnchor && !info.contains(event.relatedTarget)) closeInfo();
}

info.addEventListener("pointerenter", () => clearTimeout(infoCloseTimer));
info.addEventListener("pointerleave", scheduleInfoClose);
info.addEventListener("focusout", dismissOnFocusExit);
function closeInfo() {
  clearTimeout(infoCloseTimer);
  infoAnchor = null;
  infoRow = null;
  info.hidePopover();
}
function showInfo(anchor, hit, row) {
  clearTimeout(infoCloseTimer);
  infoAnchor = anchor;
  infoRow = row;
  document.querySelector("#info-code").textContent = "SNOMED CT " + (hit.concept_id ?? "—") + " · " + (hit.confidence == null ? "Unscored" : Math.round(hit.confidence * 100) + "% match");
  document.querySelector("#info-term").textContent = hit.concept_name ?? "Unlinked";
  info.showPopover();
  const rect = anchor.getBoundingClientRect();
  info.style.left = Math.max(8, Math.min(rect.left, innerWidth - info.offsetWidth - 8)) + "px";
  info.style.top = (rect.bottom + info.offsetHeight + 6 < innerHeight ? rect.bottom + 4 : Math.max(4,rect.top - info.offsetHeight - 4)) + "px";
}
info.onclick = event => {
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action) infoRow?.querySelector(".choices ." + action)?.click();
};
document.addEventListener("pointerdown", event => {
  if (!event.target.closest("#mention-info, mark")) closeInfo();
});
document.addEventListener("scroll", closeInfo, true);
window.addEventListener("resize", closeInfo);
document.querySelector("#note-size").onchange = event => noteText.style.setProperty("--reading-scale", event.target.value);
document.querySelector("#show-highlights").onchange = event => noteText.classList.toggle("plain-note", !event.target.checked);
document.addEventListener("keydown", event => {
  if (event.key === "Escape") { closeInfo(); return; }
  if (event.ctrlKey || event.altKey || event.metaKey || event.target.closest("input,textarea,select,dialog,details") || !event.target.closest(".workspace,#mention-info")) return;
  const visible = mentionViews.filter(view => !view.row.hidden);
  const current = infoRow ?? event.target.closest("tr") ?? reviewRows.querySelector(".is-current");
  const index = visible.findIndex(view => view.row === current);
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    closeInfo();
    const next = visible[Math.max(0, Math.min(visible.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)))];
    next?.row.querySelector(".mention-link").focus();
  } else if (current && !event.repeat && ["a","r"].includes(event.key.toLowerCase())) {
    current.querySelector(event.key.toLowerCase() === "a" ? ".approved" : ".rejected").click();
  }
});

async function requestNote(key, url, options) {
  inFlight?.abort();
  const controller = new AbortController();
  inFlight = controller;
  render(null);
  showStatus("Analyzing note…");
  try {
    const note = await readJson(url, {...options, signal: controller.signal});
    if (inFlight !== controller) return;
    note.decisions = new Map();
    notes.set(key, note);
    render(note);
    showStatus("");
  } catch (error) {
    if (!controller.signal.aborted) showStatus("Could not analyze the note. " + error.message, true);
  } finally {
    if (inFlight === controller) inFlight = null;
  }
}

function switchNote() {
  inFlight?.abort();
  inFlight = null;
  const key = selection.value;
  form.hidden = key !== "custom";
  customToggle.setAttribute("aria-expanded", String(!form.hidden));
  customToggle.disabled = key === "custom" && !lastValidation;
  if (key !== "custom") lastValidation = key;
  showStatus("");
  if (key === "custom" || notes.has(key)) {
    render(notes.get(key) ?? null);
    return;
  }
  requestNote(key, "/api/note?" + new URLSearchParams({note_id: key.slice(5)}));
}

selection.addEventListener("change", switchNote);
customToggle.addEventListener("click", () => {
  selection.value = selection.value === "custom" ? lastValidation : "custom";
  switchNote();
  if (!form.hidden) customText.focus();
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!customText.value.trim()) {
    showStatus("Enter note text to analyze.", true);
    customText.focus();
    return;
  }
  const existing = notes.get("custom");
  if (existing?.text === customText.value) {
    inFlight?.abort();
    inFlight = null;
    render(existing);
    showStatus("");
    return;
  }
  requestNote("custom", "/api/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: customText.value}),
  });
});

function csvCell(value) {
  let text = String(value ?? "");
  if (typeof value === "string" && (/^\s*[=+\-@]/.test(value) || /^[\t\r\n]/.test(value))) {
    text = "'" + text;
  }
  return '"' + text.replaceAll('"', '""') + '"';
}

function download(extension) {
  if (!active?.decisions.size) return;
  const fields = ["note_id", "start", "end", "text", "concept_id", "concept_name", "confidence", "decision"];
  const rows = active.mentions.flatMap((hit, index) => {
    const decision = active.decisions.get(index);
    if (!decision) return [];
    const row = {...hit, note_id: active.note_id, decision};
    return [Object.fromEntries(fields.map((key) => [key, row[key]]))];
  });
  const csv = extension === "csv";
  const content = csv
    ? "\uFEFF" + [fields.join(","), ...rows.map((row) => fields.map((key) => csvCell(row[key])).join(","))].join("\r\n") + "\r\n"
    : JSON.stringify(rows, null, 2) + "\n";
  const blob = new Blob([content], {type: csv ? "text/csv;charset=utf-8" : "application/json"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "reviewed-mentions." + extension;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

csvButton.addEventListener("click", () => download("csv"));
jsonButton.addEventListener("click", () => download("json"));

document.querySelector("#open-metrics").addEventListener("click", async () => {
  metricsDialog.showModal();
  metricsTable.hidden = true;
  metricsStatus.textContent = "Loading saved benchmark…";
  try {
    const metrics = await readJson("/api/metrics");
    metricsRows.replaceChildren();
    for (const [label, key] of [
      ["Mention F1", "mention_f1"],
      ["Linking Accuracy", "linking_accuracy"],
      ["End-to-End F1", "end_to_end_f1"],
    ]) {
      const row = metricsRows.insertRow();
      const heading = document.createElement("th");
      heading.scope = "row";
      heading.textContent = label;
      row.append(heading);
      row.insertCell().textContent = metrics.baseline[key].toFixed(6);
      row.insertCell().textContent = metrics.system[key].toFixed(6);
    }
    metricsStatus.textContent = metrics.validation_notes.toLocaleString() + " validation notes";
    metricsTable.hidden = false;
  } catch {
    metricsStatus.textContent = "Saved benchmark unavailable. Run python -m src.pipeline to generate it.";
  }
});

try {
  const ids = await readJson("/api/notes");
  const group = document.querySelector("#validation-options");
  for (const id of ids) group.append(new Option(id, "note:" + id));
  selection.value = ids.length ? "note:" + ids[0] : "custom";
  selection.disabled = false;
  switchNote();
} catch (error) {
  selection.disabled = false;
  selection.value = "custom";
  switchNote();
  showStatus("Could not load validation notes. " + error.message, true);
}


const exportToggle = document.querySelector("#export-toggle");
const exportOptions = document.querySelector("#export-options");
function showExports(open) {
  exportOptions.hidden = !open;
  exportToggle.setAttribute("aria-expanded", String(open));
}
exportToggle.addEventListener("click", () => showExports(exportOptions.hidden));
document.addEventListener("click", event => {
  if (!exportToggle.contains(event.target)) showExports(false);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !exportOptions.hidden) {
    showExports(false);
    exportToggle.focus();
  }
});
