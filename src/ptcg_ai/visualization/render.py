from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_visualizer_html(
    snapshots: list[dict],
    output_path: Path,
    title: str = "PTCG replay",
    metadata: dict[str, Any] | None = None,
    image_dir: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title,
        "metadata": metadata or {},
        "snapshots": snapshots,
        "images": _card_image_paths(image_dir, output_path),
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    output_path.write_text(_HTML.replace("__PTCG_REPLAY_DATA__", encoded), encoding="utf-8")


def _card_image_paths(image_dir: Path | None, output_path: Path) -> dict[str, str]:
    if image_dir is None or not image_dir.exists():
        return {}

    images = {}
    for path in image_dir.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if not path.stem.isdecimal():
            continue
        relative_path = os.path.relpath(path, output_path.parent)
        images[path.stem] = relative_path.replace(os.sep, "/")
    return images


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PTCG Replay</title>
  <style>
    :root {
      color-scheme: light;
      --mat: #89c779;
      --mat-dark: #6fb060;
      --line: rgba(255, 255, 255, 0.88);
      --panel: #ffffff;
      --ink: #14202a;
      --muted: #526474;
      --border: #303a42;
      --you: #2388ff;
      --opponent: #ff5ac7;
      --accent: #1c73ef;
      --danger: #c4474f;
      --card: #f9fbfd;
      --energy: #2b82c6;
      --trainer: #8e6bc7;
      --pokemon: #cf654f;
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      height: 100%;
    }

    body {
      margin: 0;
      overflow: hidden;
      background: var(--mat);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    button,
    input,
    select {
      font: inherit;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: clamp(220px, 12.5vw, 250px) minmax(640px, 1fr) clamp(330px, 21vw, 420px);
      gap: 0;
      background: var(--mat);
    }

    .left-rail,
    .right-rail {
      height: 100vh;
      min-height: 0;
      display: grid;
      gap: 8px;
      padding: 8px 0;
      z-index: 3;
    }

    .left-rail {
      grid-template-rows: minmax(0, 1fr) 90px 96px;
    }

    .right-rail {
      padding-right: 8px;
    }

    .panel {
      min-height: 0;
      display: grid;
      grid-template-rows: 22px minmax(0, 1fr);
      background: var(--panel);
      border: 1px solid var(--border);
      overflow: hidden;
    }

    .panel-title {
      display: grid;
      place-items: center;
      min-width: 0;
      background: var(--mat);
      color: #285443;
      font-weight: 800;
      line-height: 1;
      text-align: center;
    }

    .panel-body {
      min-height: 0;
      overflow: auto;
      padding: 4px;
      background: var(--panel);
    }

    .log-line {
      line-height: 1.28;
      overflow-wrap: anywhere;
      margin-bottom: 2px;
    }

    .who-you {
      color: var(--you);
      font-weight: 800;
    }

    .who-opp {
      color: var(--opponent);
      font-weight: 800;
    }

    .log-card {
      color: var(--danger);
      font-weight: 800;
    }

    .pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.18;
    }

    .observation {
      height: 100%;
      margin: 0;
      padding: 8px 10px;
      overflow: auto;
      background: #fff;
      border: 1px solid var(--border);
      white-space: pre;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.16;
    }

    .stage {
      position: relative;
      height: 100vh;
      min-width: 0;
      overflow: hidden;
      background: var(--mat);
    }

    .hand {
      position: absolute;
      left: 50%;
      display: flex;
      gap: 8px;
      transform: translateX(-50%);
      z-index: 2;
      max-width: min(54vw, 760px);
      overflow: visible;
      padding: 2px 4px;
    }

    .opponent-hand {
      top: 3px;
    }

    .you-hand {
      bottom: 55px;
    }

    .battle-table {
      position: absolute;
      left: 50%;
      top: 50%;
      width: min(78vw, 900px);
      height: min(calc(100vh - 160px), 650px);
      transform: translate(-50%, -50%);
    }

    .outer-lines {
      position: absolute;
      inset: 8px 46px 8px 46px;
      border: 4px solid var(--line);
      border-left-width: 3px;
      border-right-width: 3px;
    }

    .outer-lines::before,
    .outer-lines::after {
      content: "";
      position: absolute;
      left: 21%;
      right: 21%;
      height: 5px;
      background:
        linear-gradient(90deg, var(--line) 0 18%, transparent 18% 22%, var(--line) 22% 40%, transparent 40% 44%, var(--line) 44% 62%, transparent 62% 66%, var(--line) 66% 84%, transparent 84% 88%, var(--line) 88% 100%);
      border-radius: 3px;
    }

    .outer-lines::before {
      top: 20%;
    }

    .outer-lines::after {
      bottom: 20%;
    }

    .center-ring {
      position: absolute;
      left: 50%;
      top: 50%;
      width: min(26vh, 170px);
      aspect-ratio: 1;
      transform: translate(-50%, -50%);
      border: 9px solid var(--line);
      border-radius: 50%;
    }

    .center-ring::before {
      content: "";
      position: absolute;
      left: -8px;
      right: -8px;
      top: 50%;
      height: 8px;
      transform: translateY(-50%);
      background: var(--line);
    }

    .center-ring::after {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      width: 38px;
      aspect-ratio: 1;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      background: var(--mat);
      border: 8px solid var(--line);
    }

    .slots {
      position: absolute;
      display: grid;
      grid-template-rows: repeat(3, 1fr);
      gap: 8px;
      width: 72px;
    }

    .slots-left {
      left: 44px;
      bottom: 28px;
    }

    .slots-right {
      right: 44px;
      top: 28px;
    }

    .slot-outline {
      height: 94px;
      border: 4px solid var(--line);
      border-radius: 5px;
    }

    .stack-label {
      color: #111;
      font-size: 15px;
      font-weight: 620;
      line-height: 1.45;
      position: absolute;
      width: 170px;
    }

    .opponent-stats {
      left: 82px;
      top: 34%;
    }

    .you-stats {
      right: 76px;
      bottom: 27%;
    }

    .turn-word {
      color: #ffdf21;
      font-weight: 900;
      text-shadow: 1px 1px 0 #234;
    }

    .active-card-pos {
      position: absolute;
      left: 50%;
      transform: translateX(-50%);
      z-index: 2;
    }

    .opponent-active {
      top: 34%;
    }

    .you-active {
      bottom: 34%;
    }

    .bench {
      position: absolute;
      left: 50%;
      display: flex;
      gap: 8px;
      transform: translateX(-50%);
      max-width: min(48vw, 650px);
      overflow: visible;
      z-index: 2;
    }

    .opponent-bench {
      top: 19%;
    }

    .you-bench {
      bottom: 20%;
    }

    .game-result {
      position: absolute;
      left: 50%;
      top: 43%;
      transform: translateX(-50%);
      color: #ff5b6b;
      text-shadow: 1px 1px 0 #fff;
      font-weight: 900;
      white-space: nowrap;
      z-index: 3;
    }

    .damage-number {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      color: #fff;
      font-size: 27px;
      font-weight: 900;
      text-shadow: 2px 2px 0 #3c3650;
      z-index: 4;
      pointer-events: none;
    }

    .card {
      position: relative;
      width: 68px;
      aspect-ratio: 63 / 88;
      border: 1px solid rgba(25, 34, 44, 0.45);
      border-radius: 4px;
      overflow: hidden;
      background: var(--card);
      box-shadow: 0 2px 5px rgba(0, 0, 0, 0.25);
      flex: 0 0 auto;
      transform-origin: center center;
      transition: transform 120ms ease, box-shadow 120ms ease;
      z-index: 1;
    }

    .card:hover {
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.36);
      transform: scale(2.18);
      z-index: 30;
    }

    .opponent-hand .card:hover {
      transform: translateY(36px) scale(2.18);
    }

    .you-hand .card:hover {
      transform: translateY(-38px) scale(2.18);
    }

    .opponent-bench .card:hover {
      transform: translateY(26px) scale(2.05);
    }

    .you-bench .card:hover {
      transform: translateY(-26px) scale(2.05);
    }

    .active-card-pos .card:hover {
      transform: scale(2.05);
    }

    .card.active {
      width: 74px;
    }

    .card.back {
      background:
        linear-gradient(90deg, transparent 0 42%, rgba(255, 255, 255, 0.78) 42% 58%, transparent 58%),
        linear-gradient(0deg, #d4484e 0 48%, #24467e 48% 52%, #f3f6fb 52% 100%);
    }

    .card.back::after {
      content: "";
      position: absolute;
      width: 25px;
      aspect-ratio: 1;
      border: 4px solid #24467e;
      border-radius: 50%;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      background: #f3f6fb;
    }

    .card-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .fake-card {
      height: 100%;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    .fake-name {
      padding: 3px 4px 2px;
      min-height: 20px;
      font-size: 8.5px;
      line-height: 1.05;
      font-weight: 800;
      background: rgba(255, 255, 255, 0.82);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }

    .fake-art {
      display: grid;
      place-items: center;
      background: #e9f3fb;
    }

    .fake-card.energy .fake-art {
      background: #dceefa;
    }

    .fake-card.trainer .fake-art {
      background: #eee6fb;
    }

    .fake-card.pokemon .fake-art {
      background: #f9e7e2;
    }

    .fake-mark {
      width: 24px;
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--pokemon);
      color: #fff;
      font-size: 12px;
      font-weight: 900;
    }

    .fake-card.energy .fake-mark {
      background: var(--energy);
    }

    .fake-card.trainer .fake-mark {
      background: var(--trainer);
    }

    .fake-foot {
      min-height: 18px;
      padding: 2px 3px;
      font-size: 8px;
      line-height: 1.1;
      background: rgba(255, 255, 255, 0.85);
      overflow: hidden;
    }

    .controls {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 8px;
      height: 40px;
      display: grid;
      grid-template-columns: 240px 42px minmax(260px, 1fr) 66px;
      gap: 8px;
      align-items: center;
      padding: 0 4px;
      z-index: 5;
    }

    .speed-grid {
      display: grid;
      grid-template-columns: repeat(3, auto);
      gap: 4px 10px;
      font-size: 13px;
      color: #526064;
    }

    .speed-grid label {
      display: inline-flex;
      gap: 3px;
      align-items: center;
      white-space: nowrap;
    }

    .play-button {
      width: 40px;
      height: 34px;
      border: 1px solid #777;
      border-radius: 3px;
      background: #eee;
      color: #111;
      cursor: pointer;
      font-size: 18px;
    }

    .step-slider {
      width: 100%;
      accent-color: var(--accent);
    }

    .step-box {
      width: 64px;
      height: 28px;
      border: 1px solid #999;
      background: #fff;
      padding: 2px 5px;
      font-size: 13px;
    }

    @media (max-width: 1280px) {
      .app {
        grid-template-columns: 210px minmax(560px, 1fr) 330px;
      }

      .battle-table {
        width: min(74vw, 760px);
      }

      .card {
        width: 58px;
      }

      .card.active {
        width: 65px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="left-rail">
      <section class="panel">
        <div class="panel-title">Log</div>
        <div id="logs" class="panel-body"></div>
      </section>
      <section class="panel">
        <div class="panel-title">Select</div>
        <pre id="selectText" class="panel-body pre"></pre>
      </section>
      <section class="panel">
        <div class="panel-title">Selected Action</div>
        <pre id="selectedText" class="panel-body pre"></pre>
      </section>
    </aside>

    <main class="stage">
      <div id="field"></div>
      <div class="controls">
        <div id="speeds" class="speed-grid">
          <label><input type="radio" name="speed" value="1800">x0.3</label>
          <label><input type="radio" name="speed" value="1200">x0.5</label>
          <label><input type="radio" name="speed" value="700" checked>x1</label>
          <label><input type="radio" name="speed" value="450">x1.5</label>
          <label><input type="radio" name="speed" value="300">x2</label>
          <label><input type="radio" name="speed" value="180">x3</label>
        </div>
        <button id="playPause" class="play-button" title="Play" aria-label="Play">&#9654;</button>
        <input id="step" class="step-slider" type="range" min="0" value="0">
        <input id="stepBox" class="step-box" type="text" readonly>
      </div>
    </main>

    <aside class="right-rail">
      <section class="panel">
        <div class="panel-title">Observation</div>
        <pre id="observation" class="observation"></pre>
      </section>
    </aside>
  </div>

  <script id="replay-data" type="application/json">__PTCG_REPLAY_DATA__</script>
  <script>
    const replay = JSON.parse(document.getElementById("replay-data").textContent);
    const snapshots = replay.snapshots || [];
    const images = replay.images || {};
    const field = document.getElementById("field");
    const logs = document.getElementById("logs");
    const selectText = document.getElementById("selectText");
    const selectedText = document.getElementById("selectedText");
    const observation = document.getElementById("observation");
    const stepInput = document.getElementById("step");
    const stepBox = document.getElementById("stepBox");
    const playPause = document.getElementById("playPause");
    let playbackTimer = null;

    stepInput.max = Math.max(0, snapshots.length - 1);
    stepInput.value = 0;

    stepInput.addEventListener("input", () => setStep(Number(stepInput.value)));
    playPause.addEventListener("click", togglePlayback);
    document.getElementById("speeds").addEventListener("change", () => {
      if (playbackTimer !== null) {
        pausePlayback();
        startPlayback();
      }
    });
    document.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") setStep(Number(stepInput.value) - 1);
      if (event.key === "ArrowRight") setStep(Number(stepInput.value) + 1);
      if (event.key === " ") {
        event.preventDefault();
        togglePlayback();
      }
    });

    function setStep(index) {
      const next = Math.max(0, Math.min(snapshots.length - 1, index));
      stepInput.value = next;
      render(next);
    }

    function togglePlayback() {
      if (playbackTimer === null) {
        startPlayback();
      } else {
        pausePlayback();
      }
    }

    function startPlayback() {
      if (snapshots.length <= 1) return;
      if (Number(stepInput.value) >= snapshots.length - 1) {
        setStep(0);
      }
      playbackTimer = window.setInterval(advancePlayback, currentSpeed());
      playPause.innerHTML = "&#10074;&#10074;";
      playPause.title = "Pause";
      playPause.setAttribute("aria-label", "Pause");
    }

    function pausePlayback() {
      if (playbackTimer !== null) {
        window.clearInterval(playbackTimer);
        playbackTimer = null;
      }
      playPause.innerHTML = "&#9654;";
      playPause.title = "Play";
      playPause.setAttribute("aria-label", "Play");
    }

    function advancePlayback() {
      const current = Number(stepInput.value);
      if (current >= snapshots.length - 1) {
        pausePlayback();
        return;
      }
      setStep(current + 1);
    }

    function currentSpeed() {
      const selected = document.querySelector('input[name="speed"]:checked');
      return Number(selected ? selected.value : 700);
    }

    function render(index) {
      const snap = snapshots[index];
      stepBox.value = `${index}/${Math.max(0, snapshots.length - 1)}`;
      if (!snap) {
        field.innerHTML = "";
        return;
      }

      const current = snap.current || {};
      const players = current.players || [{}, {}];
      field.innerHTML = renderField(current, players[0] || {}, players[1] || {});
      logs.innerHTML = renderLogs(index);
      selectText.textContent = renderSelectText(snap.select || {});
      selectedText.textContent = renderSelectedText(snap);
      observation.textContent = JSON.stringify(current, null, 2);
    }

    function renderField(current, you, opponent) {
      return `
        <div class="hand opponent-hand">${renderHand(opponent.hand || [], opponent.handCount, "opponent")}</div>
        <div class="battle-table">
          <div class="outer-lines"></div>
          <div class="center-ring"></div>
          <div class="slots slots-left"><div class="slot-outline"></div><div class="slot-outline"></div><div class="slot-outline"></div></div>
          <div class="slots slots-right"><div class="slot-outline"></div><div class="slot-outline"></div><div class="slot-outline"></div></div>
          <div class="stack-label opponent-stats">${turnLabel(current, 1)}<br>Time&nbsp; 600<br>Discard ${count(opponent.discard)}<br>Deck ${value(opponent.deckCount)}</div>
          <div class="stack-label you-stats">${turnLabel(current, 0)}<br>Time&nbsp; 600<br>Discard ${count(you.discard)}<br>Deck ${value(you.deckCount)}</div>
          <div class="bench opponent-bench">${renderBench(opponent.bench || [])}</div>
          <div class="bench you-bench">${renderBench(you.bench || [])}</div>
          <div class="active-card-pos opponent-active">${renderCard(activeOf(opponent), "active")}</div>
          <div class="active-card-pos you-active">${renderCard(activeOf(you), "active")}</div>
          ${damageText(you, opponent)}
          ${resultText(current)}
        </div>
        <div class="hand you-hand">${renderHand(you.hand || [], you.handCount, "you")}</div>
      `;
    }

    function turnLabel(current, playerIndex) {
      const first = current.firstPlayer === playerIndex ? "First" : "Second";
      return `<span class="turn-word">${first}</span>`;
    }

    function activeOf(player) {
      return (player.active || [])[0] || null;
    }

    function damageText(you, opponent) {
      const cards = [activeOf(you), activeOf(opponent)].filter(Boolean);
      const damage = cards.reduce((total, card) => {
        const maxHp = Number(card.maxHp || 0);
        const hp = Number(card.hp || 0);
        return total + Math.max(0, maxHp - hp);
      }, 0);
      return damage > 0 ? `<div class="damage-number">${damage}</div>` : "";
    }

    function resultText(current) {
      if (current.result === -1 || current.result === undefined) return "";
      return `<div class="game-result">[Win] Player ${current.result}</div>`;
    }

    function renderBench(cards) {
      const visible = cards.slice(0, 5).map(card => renderCard(card, "bench"));
      return visible.join("");
    }

    function renderHand(cards, handCount, owner) {
      if (cards && cards.length) {
        return cards.slice(0, 8).map(card => renderCard(card, "hand")).join("");
      }
      const hiddenCount = Math.min(Number(handCount || 0), 8);
      if (hiddenCount <= 0) return "";
      return Array.from({ length: hiddenCount }, () => hiddenCard()).join("");
    }

    function renderCard(card, role) {
      if (!card) return "";
      const imageUrl = images[String(card.id || "")];
      if (imageUrl) {
        return `<div class="card ${role || ""}"><img class="card-img" src="${escapeHtml(imageUrl)}" alt="${escapeHtml(card.name || "")}"></div>`;
      }

      const kind = cardKind(card);
      const hp = card.maxHp ? `HP ${value(card.hp)}/${value(card.maxHp)}` : `#${value(card.id)}`;
      const energyCount = count(card.energyCards);
      const toolCount = count(card.tools);
      return `
        <div class="card ${role || ""}">
          <div class="fake-card ${kind.className}">
            <div class="fake-name">${escapeHtml(card.name || `Card ${card.id}`)}</div>
            <div class="fake-art"><div class="fake-mark">${escapeHtml(kind.mark)}</div></div>
            <div class="fake-foot">${escapeHtml(hp)}${energyCount ? `<br>E ${energyCount}` : ""}${toolCount ? ` T ${toolCount}` : ""}</div>
          </div>
        </div>
      `;
    }

    function hiddenCard() {
      return `<div class="card back"></div>`;
    }

    function renderLogs(index) {
      const rows = [];
      for (let step = 0; step <= index; step += 1) {
        for (const log of snapshots[step].logs || []) {
          rows.push(logLine(log));
        }
      }
      return rows.slice(-42).join("") || `<div class="log-line">No logs yet.</div>`;
    }

    function logLine(log) {
      const who = log.playerIndex === 0 ? "YOU" : "OPPONENT";
      const whoClass = log.playerIndex === 0 ? "who-you" : "who-opp";
      const card = log.cardName || log.name || cardNameFromLog(log);
      const detail = logDetail(log);
      return `<div class="log-line"><span class="${whoClass}">${who}: </span><span class="log-card">${escapeHtml(card)}</span> ${escapeHtml(detail)}</div>`;
    }

    function cardNameFromLog(log) {
      if (log.cardId) return `Card ${log.cardId}`;
      if (log.cardIdActive) return `Card ${log.cardIdActive}`;
      if (log.cardIdBefore) return `Card ${log.cardIdBefore}`;
      return log.type || "Event";
    }

    function logDetail(log) {
      if (log.type === "MoveCard") return `moved from ${areaName(log.fromArea)} to ${areaName(log.toArea)}.`;
      if (log.type === "Draw") return "drawn.";
      if (log.type === "Play") return "played.";
      if (log.type === "Attach") return `attached to Card ${log.cardIdTarget}.`;
      if (log.type === "Attack") return `used attack ${log.attackId}.`;
      if (log.type === "Result") return `result ${log.result}.`;
      return JSON.stringify(log);
    }

    function renderSelectText(select) {
      return [
        `${select.playerIndex === 1 ? "Opponent" : ""} select`.trim(),
        `type : ${value(select.type)}`,
        `context : ${value(select.context)}`,
        `minCount : ${value(select.minCount)}`,
        `maxCount : ${value(select.maxCount)}`,
      ].join("\\n");
    }

    function renderSelectedText(snap) {
      const selected = snap.selected || [];
      if (!selected.length) return "";
      return selected.map(index => {
        const option = (snap.select && snap.select.option || [])[index] || {};
        return `[${index}]\\n${optionSummary(option)}`;
      }).join("\\n\\n");
    }

    function optionSummary(option) {
      if (option.type === "Card") return `Card : ${option.name || option.cardId || option.id || option.index}`;
      if (option.type === "Play") return `Play : hand ${option.index}`;
      if (option.type === "Attack") return `Attack : ${option.attackName || option.attackId}`;
      if (option.type === "Yes" || option.type === "No") return option.type;
      return JSON.stringify(option);
    }

    function cardKind(card) {
      const name = String(card.name || "");
      const id = Number(card.id || 0);
      if (name.includes("Energy")) return { className: "energy", mark: "E" };
      if ([1145, 1158, 1205, 1227, 1235].includes(id)) return { className: "trainer", mark: "T" };
      return { className: "pokemon", mark: name.includes("Mega") ? "M" : "P" };
    }

    function areaName(value) {
      const areas = {
        1: "Deck",
        2: "Hand",
        3: "Discard",
        4: "Active Spot",
        5: "Bench",
        6: "Prize Cards",
        7: "Stadium",
        8: "Energy",
        9: "Tool",
        12: "Looking",
      };
      return areas[value] || `Area ${value}`;
    }

    function count(items) {
      return items ? items.length : 0;
    }

    function value(input) {
      return input === undefined || input === null ? "-" : input;
    }

    function escapeHtml(input) {
      return String(input)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    render(0);
  </script>
</body>
</html>
"""
