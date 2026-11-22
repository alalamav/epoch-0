// Boot and wiring. The logic lives in src/ui and src/app-state.ts; this file only connects it to
// the weights worker, the audio graph and the views.

import { createAudioGraph } from "./audio/graph";
import { parsePlate, randomPlate } from "./plate/seed";
import { createDishView } from "./view/dish";
import { computeRowLayout } from "./view/layout";
import { createPlateScene } from "./view/scene";

import { createReadiness, createSoundingTracker, statusLine, wellState } from "./app-state";
import type {
  AudioGraph,
  DishView,
  EngineReport,
  FromWeightsWorker,
  LayerWeights,
  PlateScene,
  RowLayout,
  RowMatrices,
  ToEngine,
  ToWeightsWorker,
} from "./contracts";
import { MAX_LAYERS, ROWS, WELLS, wellRow } from "./plate/constants";
import { createAbout } from "./ui/about";
import { createControls } from "./ui/controls";
import { createDebugPalette } from "./ui/debug-palette";
import { createDishPanel } from "./ui/dish-panel";
import { createPlateGrid } from "./ui/grid";
import { NARROW_QUERY } from "./ui/grid-nav";
import { createSpecimenLabel } from "./ui/label";
import { needsRewrite, plateFromSearch, plateHref, resolvePlate } from "./ui/plate-url";
import { createReadout, magnifyTarget } from "./ui/readout";
import { createPressLogic } from "./ui/press";
import { readDocumentTheme } from "./ui/theme";

const byId = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`index.html has no #${id}`);
  return node;
};

const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** Stands in for the scene if WebGL is missing: the buttons then draw the wells (see styles.css). */
const noScene: PlateScene = {
  resize() {},
  setRowLayout() {},
  clear() {},
  setPressed() {},
  applyReport() {},
  setTheme() {},
  setReducedMotion() {},
  render() {},
  dispose() {},
};

function boot() {
  const statusEl = byId("status");
  const now = () => performance.now();

  // ── state ──────────────────────────────────────────────────
  let plate = 0;
  let lastWell: number | null = null;
  let worker: Worker | null = null;
  let graph: AudioGraph | null = null;
  let error: string | null = null;
  /** A well has been pressed at least once: from then on a context that is not running is worth a hint. */
  let audioWanted = false;
  let resuming = false;
  const readiness = createReadiness();
  const tracker = createSoundingTracker();
  const matrices: RowMatrices[] = [];
  const layouts: (RowLayout | null)[] = [];
  /** Weights that arrived before the audio graph existed. */
  let queued: LayerWeights[] = [];

  const isReady = (well: number) => graph !== null && readiness.isReady(well);

  // ── press logic ────────────────────────────────────────────
  const press = createPressLogic({
    isReady,
    isSounding: (well) => tracker.isSounding(well),
    onTouch(well) {
      lastWell = well;
      label.set(plate, well);
      readout.set(well);
    },
    onPress(well) {
      audioWanted = true;
      wake(); // inside the gesture; the press message waits in the port until the context runs
      tracker.press(well, now());
      post({ type: "press", well });
      scene.setPressed(well, true);
      refreshWell(well);
      refreshRinse();
    },
    onRelease(well) {
      post({ type: "release", well });
      scene.setPressed(well, false);
      refreshWell(well);
    },
    onStop(well) {
      tracker.stop(well, now());
      post({ type: "stop", well });
      refreshWell(well);
      refreshRinse();
    },
  });

  // ── interface ──────────────────────────────────────────────
  const grid = createPlateGrid(byId("plate"), { press });
  const label = createSpecimenLabel(byId("label"));
  const about = createAbout(document.body);
  const readout = createReadout(byId("readout"), { onOpen: magnify });
  const aboutButton = byId("about-btn") as HTMLButtonElement;
  aboutButton.addEventListener("click", () => {
    palette.host(about.dialog);
    about.open();
  });
  const controls = createControls(byId("controls"), {
    onRinse: rinse,
    onNewPlate: () => goToPlate(randomPlate(), "push"),
    onMagnify: magnify,
    getLink: () => plateHref(location.href, plate),
  });
  const dishPanel = createDishPanel(document.body, {
    press,
    onClose() {
      dish?.close();
      palette.host(document.body);
    },
    onResize: () => dish?.resize(),
    canOpen: canMagnify,
    onNavigate: (well) => openDish(well),
  });
  about.dialog.addEventListener("close", () => palette.host(document.body));

  // ── views ──────────────────────────────────────────────────
  let theme = readDocumentTheme();
  let scene: PlateScene = noScene;
  try {
    scene = createPlateScene(grid.canvas, theme);
  } catch (e) {
    console.error(e);
    grid.setSceneAvailable(false);
  }
  let dish: DishView | null = null;
  try {
    dish = createDishView(dishPanel.container, theme);
  } catch (e) {
    console.error(e);
  }

  const narrow = matchMedia(NARROW_QUERY);
  grid.setTransposed(narrow.matches);
  narrow.addEventListener("change", () => grid.setTransposed(narrow.matches));
  grid.observeResize((width, height, dpr, wells) => {
    if (width > 0 && height > 0) scene.resize(width, height, dpr, wells);
  });

  const dark = matchMedia("(prefers-color-scheme: dark)");
  function refreshTheme() {
    theme = readDocumentTheme();
    scene.setTheme(theme);
    dish?.setTheme(theme);
  }
  // Debug: stain colour and light/dark switch. Shown with ?debug or Shift+D; applies a stored choice.
  const palette = createDebugPalette(document.body, { systemDark: () => dark.matches, onChange: refreshTheme });
  dark.addEventListener("change", () => palette.apply());
  document.addEventListener("keydown", (e) => {
    if (e.key !== "D" || !e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    const target = e.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) return;
    palette.setVisible(!palette.visible);
  });

  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const applyReducedMotion = () => {
    scene.setReducedMotion(reduced.matches);
    dish?.setReducedMotion(reduced.matches);
  };
  applyReducedMotion();
  reduced.addEventListener("change", applyReducedMotion);

  // ── helpers ────────────────────────────────────────────────
  function post(msg: ToEngine, transfer?: Transferable[]) {
    graph?.post(msg, transfer);
  }

  function refreshWell(well: number) {
    const state = wellState(isReady(well), press.isHeld(well), tracker.isSounding(well));
    grid.setWellState(well, state);
    dishPanel.setMapState(well, state);
    if (dishPanel.well === well) dishPanel.setWellState(state);
  }

  function refreshAllWells() {
    for (let well = 0; well < WELLS; well++) refreshWell(well);
    refreshDishButton();
  }

  function canMagnify(well: number): boolean {
    return dish !== null && readiness.isReady(well) && !!layouts[wellRow(well)];
  }

  function refreshDishButton() {
    let any = false;
    for (let well = 0; well < WELLS && !any; well++) any = canMagnify(well);
    controls.setMagnifyEnabled(any);
    readout.element.disabled = !any;
  }

  function refreshRinse() {
    let any = false;
    for (let well = 0; well < WELLS && !any; well++) any = tracker.isSounding(well);
    controls.setSounding(any);
  }

  /** Open the dish on the last well touched, or on a random well if none was (or it is not loaded). */
  function magnify() {
    const well = magnifyTarget(lastWell, WELLS, canMagnify);
    if (well !== null) openDish(well);
  }

  function refreshStatus() {
    const audioPaused = audioWanted && !resuming && graph !== null && graph.context.state !== "running";
    const status = statusLine({ error, audioPaused, loaded: readiness.count(), total: ROWS * MAX_LAYERS });
    if (statusEl.textContent !== status.text) statusEl.textContent = status.text;
    statusEl.dataset.kind = status.kind;
  }

  function fail(text: string) {
    error = text;
    refreshStatus();
  }

  /** Start or restart the audio context. Must be called from inside a user gesture. */
  function wake() {
    if (!graph || !audioWanted || resuming || graph.context.state === "running") return;
    resuming = true;
    const settle = () => {
      resuming = false;
      clearTimeout(timer);
      refreshStatus();
    };
    // If the browser neither resolves nor rejects (no activation yet), show the hint after a moment.
    const timer = setTimeout(settle, 1000);
    graph.resume().then(settle, settle);
  }

  function rinse() {
    tracker.stopAll(now());
    post({ type: "stopAll" });
    refreshAllWells();
    refreshRinse();
  }

  /** Open the dish on `well` (default: the last well touched), or move an open dish to it. */
  function openDish(well: number | null = lastWell) {
    if (!dish || well === null || !readiness.isReady(well)) return;
    const row = wellRow(well);
    const layout = layouts[row];
    if (!layout) return;
    lastWell = well;
    label.set(plate, well);
    readout.set(well);
    dishPanel.open(plate, well); // first, so the container has a size
    palette.host(dishPanel.dialog);
    dish.setAnalyser(graph ? graph.analyser : null);
    dish.open(well, layout, matrices[row]);
    dish.resize();
    refreshWell(well);
  }

  // ── weights ────────────────────────────────────────────────
  function sendWeights(weights: LayerWeights) {
    const transfer: Transferable[] = [weights.R.buffer, weights.K.buffer];
    if (weights.C) transfer.push(weights.C.buffer);
    if (weights.W) transfer.push(weights.W.buffer);
    post({ type: "weights", weights }, transfer);
  }

  function onLayer(weights: LayerWeights) {
    const { row, layer } = weights;
    // Keep copies of the small matrices for drawing before the buffers go to the engine.
    const m = matrices[row];
    m.R[layer] = weights.R.slice();
    m.C[layer] = weights.C ? weights.C.slice() : null;
    const newlyReady = readiness.addLayer(row, layer);

    // Lay out the layers that have arrived without a gap.
    const layout = computeRowLayout(m.C.slice(0, readiness.layers(row)));
    layouts[row] = layout;
    scene.setRowLayout(row, layout);

    if (graph) sendWeights(weights);
    else queued.push(weights);

    for (const well of newlyReady) refreshWell(well);
    refreshDishButton();
    refreshStatus();
  }

  function startWorker(forPlate: number) {
    let w: Worker;
    try {
      w = new Worker(new URL("./plate/weights.worker.ts", import.meta.url), { type: "module" });
    } catch (e) {
      fail(`weights failed: ${message(e)}`);
      return;
    }
    worker = w;
    w.onmessage = (event: MessageEvent<FromWeightsWorker>) => {
      const msg = event.data;
      // A message for a plate that has since been replaced.
      if (w !== worker || msg.plate !== plate) return;
      if (msg.type === "layer") onLayer(msg.weights);
      else if (msg.type === "error") fail(`weights failed: ${msg.message}`);
      else refreshStatus();
    };
    w.onerror = (event) => {
      if (w !== worker) return;
      event.preventDefault?.();
      fail(`weights failed: ${event.message || "worker error"}`);
    };
    const generate: ToWeightsWorker = { type: "generate", plate: forPlate };
    w.postMessage(generate);
  }

  // ── plates ─────────────────────────────────────────────────
  function loadPlate(next: number) {
    worker?.terminate();
    worker = null;
    plate = next;
    lastWell = null;
    if (error?.startsWith("weights")) error = null;

    dishPanel.close();
    for (const well of press.heldWells()) scene.setPressed(well, false);
    press.reset();
    tracker.clear();
    readiness.reset();
    queued = [];
    for (let row = 0; row < ROWS; row++) {
      matrices[row] = { R: [], C: [] };
      layouts[row] = null;
    }
    post({ type: "clear" });
    scene.clear();

    grid.setPlate(plate);
    controls.setPlate(plate);
    label.set(plate, null);
    readout.set(null);
    refreshRinse();
    refreshAllWells();
    refreshStatus();
    startWorker(plate);
  }

  function goToPlate(next: number, history_: "push" | "replace" | "none") {
    const href = plateHref(location.href, next);
    if (history_ === "push" && next !== plate) history.pushState(null, "", href);
    else if (history_ !== "none" && needsRewrite(location.search, next)) history.replaceState(null, "", href);
    if (next !== plate || worker === null) loadPlate(next);
  }

  addEventListener("popstate", () => {
    const linked = plateFromSearch(location.search, parsePlate);
    if (linked === null) goToPlate(randomPlate(), "replace");
    else goToPlate(linked, "none");
  });

  // ── audio ──────────────────────────────────────────────────
  function onReport(report: EngineReport) {
    scene.applyReport(report);
    dish?.applyReport(report);
    tracker.report(
      report.wells.map((w) => w.well),
      now(),
    );
    for (let well = 0; well < WELLS; well++) refreshWell(well);
    refreshRinse();
  }

  createAudioGraph().then(
    (g) => {
      graph = g;
      g.onReport = onReport;
      g.context.addEventListener("statechange", refreshStatus);
      // Weights that arrived first. A plate change in between has already emptied the queue.
      for (const weights of queued) sendWeights(weights);
      queued = [];
      dish?.setAnalyser(g.analyser);
      refreshAllWells();
      refreshStatus();
    },
    (e) => {
      console.error(e);
      fail(`audio unavailable: ${message(e)}`);
    },
  );

  // Any later gesture restarts a context that the browser suspended or interrupted (iOS), and
  // completes a first resume that a touch pointerdown was not allowed to make.
  for (const type of ["pointerup", "touchend", "keydown", "click"] as const) {
    document.addEventListener(type, wake, { capture: true, passive: true });
  }

  // Wells keep sounding in the background, but no button stays down unseen.
  addEventListener("blur", () => press.releaseAll());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) press.releaseAll();
    else refreshStatus();
  });

  // ── frames ─────────────────────────────────────────────────
  function frame(t: number) {
    scene.render(t);
    dish?.render(t);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // ── first plate ────────────────────────────────────────────
  const initial = resolvePlate(location.search, parsePlate, randomPlate);
  goToPlate(initial.plate, "replace");
}

try {
  boot();
} catch (e) {
  console.error(e);
  const status = document.getElementById("status");
  if (status) {
    status.dataset.kind = "error";
    status.textContent = `could not start: ${message(e)}`;
  }
}
