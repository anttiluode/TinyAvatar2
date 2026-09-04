(() => {
  "use strict";

  const MODEL_SIZE = 96;
  const PACKETS = 256;
  const PARAMS_PER_PACKET = 11;
  // Browser manifold edits use a frozen per-anchor Jacobian, so keep them
  // inside a deliberately local trust region around that immutable anchor.
  // This blocks the old pointer-event integrator from walking arbitrarily
  // far off distribution while still leaving several latent units to pull.
  const LATENT_TRUST_RADIUS = 3.0;

  const ui = {
    canvas: document.querySelector("#faceCanvas"),
    canvasWrap: document.querySelector("#canvasWrap"),
    loading: document.querySelector("#loadingState"),
    engine: document.querySelector("#engineReadout"),
    waveSlider: document.querySelector("#waveSlider"),
    wavePercent: document.querySelector("#wavePercent"),
    waveMeter: document.querySelector("#waveMeter"),
    pullMeter: document.querySelector("#pullMeter"),
    faceNumber: document.querySelector("#faceNumber"),
    newFace: document.querySelector("#newFace"),
    resetFace: document.querySelector("#resetFace"),
    share: document.querySelector("#shareResult"),
    modeButtons: [...document.querySelectorAll(".mode")],
    presets: [...document.querySelectorAll("[data-waves]")],
  };

  const state = {
    session: null,
    anchors: [],
    anchorIndex: 0,
    z: new Float32Array(128),
    latentOffset: new Float32Array(128),
    shifts: new Float32Array(PACKETS * 2),
    mask: new Float32Array(PACKETS).fill(1),
    params: null,
    rankedPackets: Array.from({ length: PACKETS }, (_, index) => index),
    activeWaves: PACKETS,
    mode: "manifold",
    drag: null,
    keyboardGrab: null,
    maxPull: 0,
    running: false,
    dirty: false,
    ready: false,
    inferenceTimes: [],
  };

  const displayContext = ui.canvas.getContext("2d", { alpha: false });
  const fieldCanvas = document.createElement("canvas");
  fieldCanvas.width = MODEL_SIZE;
  fieldCanvas.height = MODEL_SIZE;
  const fieldContext = fieldCanvas.getContext("2d", { alpha: false });

  function parseControls(buffer) {
    const bytes = new Uint8Array(buffer);
    const magic = String.fromCharCode(...bytes.subarray(0, 4));
    if (magic !== "TAV1") throw new Error("Unknown control-map format");

    const view = new DataView(buffer);
    const version = view.getUint32(4, true);
    const anchorCount = view.getUint32(8, true);
    const packetCount = view.getUint32(12, true);
    const latentSize = view.getUint32(16, true);
    if (version !== 1 || packetCount !== PACKETS || latentSize !== 128) {
      throw new Error("Control map does not match the model");
    }

    let offset = 20;
    const takeFloats = (count) => {
      const end = offset + count * 4;
      const values = new Float32Array(buffer.slice(offset, end));
      offset = end;
      return values;
    };

    const anchors = [];
    for (let index = 0; index < anchorCount; index += 1) {
      const seed = view.getInt32(offset, true);
      offset += 4;
      anchors.push({
        seed,
        z: takeFloats(latentSize),
        jx: takeFloats(packetCount * latentSize),
        jy: takeFloats(packetCount * latentSize),
      });
    }
    return anchors;
  }

  function copyArray(values) {
    return new Float32Array(values);
  }

  function setAnchor(index) {
    state.anchorIndex = (index + state.anchors.length) % state.anchors.length;
    state.z = copyArray(state.anchors[state.anchorIndex].z);
    state.latentOffset.fill(0);
    state.shifts.fill(0);
    state.mask.fill(1);
    state.activeWaves = PACKETS;
    state.maxPull = 0;
    state.keyboardGrab = null;
    ui.waveSlider.value = String(PACKETS);
    ui.faceNumber.textContent = `FACE ${String(state.anchorIndex + 1).padStart(2, "0")}`;
    updateMeters();
    requestInference();
  }

  function resetDeformation() {
    state.z = copyArray(state.anchors[state.anchorIndex].z);
    state.latentOffset.fill(0);
    state.shifts.fill(0);
    state.maxPull = 0;
    state.keyboardGrab = null;
    updateMeters();
    requestInference();
  }

  function setMode(mode) {
    if (mode === state.mode) return;
    state.mode = mode;
    ui.modeButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    resetDeformation();
  }

  function updateMeters() {
    ui.pullMeter.textContent = `${String(Math.round(state.maxPull)).padStart(2, "0")} PX`;
    ui.waveMeter.textContent = String(state.activeWaves);
    ui.wavePercent.value = `${Math.round(100 * state.activeWaves / PACKETS)}%`;
  }

  function rankPackets() {
    if (!state.params) return;
    const scores = [];
    for (let packet = 0; packet < PACKETS; packet += 1) {
      const base = packet * PARAMS_PER_PACKET;
      let energy = 0;
      for (let coefficient = 5; coefficient < 11; coefficient += 1) {
        energy += state.params[base + coefficient] ** 2;
      }
      const rms = Math.sqrt(energy / 6);
      const sigma = state.params[base + 2];
      scores.push([packet, rms * sigma]);
    }
    scores.sort((left, right) => right[1] - left[1]);
    state.rankedPackets = scores.map(([packet]) => packet);
  }

  function applyWaveCount(count) {
    state.activeWaves = Math.max(16, Math.min(PACKETS, Math.round(count)));
    state.mask.fill(0);
    for (let index = 0; index < state.activeWaves; index += 1) {
      state.mask[state.rankedPackets[index]] = 1;
    }
    ui.waveSlider.value = String(state.activeWaves);
    updateMeters();
    requestInference();
  }

  function tensor(data, dimensions) {
    return new ort.Tensor("float32", copyArray(data), dimensions);
  }

  async function infer() {
    const started = performance.now();
    const outputs = await state.session.run({
      z_latent: tensor(state.z, [1, 128]),
      packet_mask: tensor(state.mask, [1, PACKETS]),
      packet_shift: tensor(state.shifts, [1, PACKETS, 2]),
    });
    const elapsed = performance.now() - started;
    state.inferenceTimes.push(elapsed);
    if (state.inferenceTimes.length > 8) state.inferenceTimes.shift();

    state.params = copyArray(outputs.packet_params.data);
    paintField(outputs.rendered_image.data);
    drawDisplay();
    if (!state.ready) {
      state.ready = true;
      rankPackets();
      document.body.classList.add("ready");
      document.querySelectorAll("button:disabled, input:disabled").forEach((control) => {
        control.disabled = false;
      });
    }

    const average = state.inferenceTimes.reduce((sum, value) => sum + value, 0) / state.inferenceTimes.length;
    ui.engine.textContent = `${average.toFixed(0)} MS · LOCAL WAVE MODEL`;
  }

  function requestInference() {
    if (!state.session) return;
    state.dirty = true;
    if (state.running) return;
    state.running = true;
    (async () => {
      try {
        while (state.dirty) {
          state.dirty = false;
          await infer();
          await new Promise((resolve) => requestAnimationFrame(resolve));
        }
      } catch (error) {
        fail(error);
      } finally {
        state.running = false;
      }
    })();
  }

  function paintField(values) {
    const pixels = fieldContext.createImageData(MODEL_SIZE, MODEL_SIZE);
    const plane = MODEL_SIZE * MODEL_SIZE;
    for (let y = 0; y < MODEL_SIZE; y += 1) {
      for (let x = 0; x < MODEL_SIZE; x += 1) {
        const source = y * MODEL_SIZE + x;
        const target = source * 4;
        pixels.data[target] = Math.round(255 * clamp01(values[source]));
        pixels.data[target + 1] = Math.round(255 * clamp01(values[plane + source]));
        pixels.data[target + 2] = Math.round(255 * clamp01(values[2 * plane + source]));
        pixels.data[target + 3] = 255;
      }
    }
    fieldContext.putImageData(pixels, 0, 0);
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function syncCanvasSize() {
    const rect = ui.canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (ui.canvas.width !== width || ui.canvas.height !== height) {
      ui.canvas.width = width;
      ui.canvas.height = height;
    }
    return { width: rect.width, height: rect.height, ratio };
  }

  function drawDisplay() {
    const size = syncCanvasSize();
    displayContext.setTransform(size.ratio, 0, 0, size.ratio, 0, 0);
    displayContext.clearRect(0, 0, size.width, size.height);
    displayContext.imageSmoothingEnabled = true;
    displayContext.imageSmoothingQuality = "high";
    displayContext.drawImage(fieldCanvas, 0, 0, size.width, size.height);
    drawPackets(size.width, size.height);
  }

  function drawPackets(width, height) {
    if (!state.params) return;
    displayContext.save();
    displayContext.globalCompositeOperation = "screen";

    for (let packet = 0; packet < PACKETS; packet += 1) {
      if (state.mask[packet] < 0.5) continue;
      const base = packet * PARAMS_PER_PACKET;
      const x = state.params[base] * width;
      const y = state.params[base + 1] * height;
      const theta = state.params[base + 3];
      const frequency = state.params[base + 4];
      const selected = state.drag?.weights?.[packet] || 0;
      const length = Math.max(3.2, Math.min(8, 2.8 + frequency * .22));
      const alpha = selected > .02 ? .28 + .68 * selected : .17;
      const hue = 158 + Math.min(92, frequency * 5.5);

      displayContext.beginPath();
      displayContext.moveTo(x - Math.cos(theta) * length, y - Math.sin(theta) * length);
      displayContext.lineTo(x + Math.cos(theta) * length, y + Math.sin(theta) * length);
      displayContext.strokeStyle = selected > .02
        ? `rgba(200,255,50,${alpha})`
        : `hsla(${hue},90%,67%,${alpha})`;
      displayContext.lineWidth = selected > .02 ? 1.8 : .75;
      displayContext.stroke();
    }

    if (state.drag) {
      const x = state.drag.pointer[0] * width;
      const y = state.drag.pointer[1] * height;
      displayContext.globalCompositeOperation = "source-over";
      displayContext.strokeStyle = "#c8ff32";
      displayContext.lineWidth = 1.5;
      displayContext.beginPath();
      displayContext.arc(x, y, 15, 0, Math.PI * 2);
      displayContext.moveTo(x - 22, y);
      displayContext.lineTo(x + 22, y);
      displayContext.moveTo(x, y - 22);
      displayContext.lineTo(x, y + 22);
      displayContext.stroke();
    }
    displayContext.restore();
  }

  function pointerPosition(event) {
    const rect = ui.canvas.getBoundingClientRect();
    return [
      clamp01((event.clientX - rect.left) / rect.width),
      clamp01((event.clientY - rect.top) / rect.height),
    ];
  }

  function grabWeights(point) {
    const weights = new Float32Array(PACKETS);
    let maximum = 0;
    let maxAmplitude = 0;
    const amplitudes = new Float32Array(PACKETS);

    for (let packet = 0; packet < PACKETS; packet += 1) {
      const base = packet * PARAMS_PER_PACKET;
      let sum = 0;
      for (let coefficient = 5; coefficient < 11; coefficient += 1) {
        sum += state.params[base + coefficient] ** 2;
      }
      amplitudes[packet] = Math.sqrt(sum / 6);
      maxAmplitude = Math.max(maxAmplitude, amplitudes[packet]);
    }

    for (let packet = 0; packet < PACKETS; packet += 1) {
      if (state.mask[packet] < .5) continue;
      const base = packet * PARAMS_PER_PACKET;
      const dx = state.params[base] - point[0];
      const dy = state.params[base + 1] - point[1];
      const distance2 = dx * dx + dy * dy;
      const sigma = Math.max(.004, state.params[base + 2]);
      const envelope = Math.exp(-distance2 / (2 * sigma * sigma));
      const radius = Math.exp(-distance2 / (2 * .10 * .10));
      const amplitude = amplitudes[packet] / Math.max(1e-8, maxAmplitude);
      weights[packet] = amplitude * envelope * radius;
      maximum = Math.max(maximum, weights[packet]);
    }

    if (maximum > 1e-9) {
      for (let packet = 0; packet < PACKETS; packet += 1) {
        const normalized = weights[packet] / maximum;
        weights[packet] = normalized < .02 ? 0 : normalized;
      }
    }
    return weights;
  }

  function buildPin(weights) {
    const anchor = state.anchors[state.anchorIndex];
    const rowX = new Float32Array(128);
    const rowY = new Float32Array(128);
    let weightSum = 0;

    for (let packet = 0; packet < PACKETS; packet += 1) {
      const weight = weights[packet];
      if (weight === 0) continue;
      weightSum += weight;
      const base = packet * 128;
      for (let latent = 0; latent < 128; latent += 1) {
        rowX[latent] += weight * anchor.jx[base + latent];
        rowY[latent] += weight * anchor.jy[base + latent];
      }
    }

    const inverseWeight = 1 / Math.max(1e-9, weightSum);
    // Browser WASM uses one frozen per-anchor linearization rather than the
    // native iterative Jacobian. Damping handles near-singular pins; the
    // target solve below is additionally bounded by LATENT_TRUST_RADIUS.
    let xx = .005 ** 2;
    let xy = 0;
    let yy = .005 ** 2;
    for (let latent = 0; latent < 128; latent += 1) {
      rowX[latent] *= inverseWeight;
      rowY[latent] *= inverseWeight;
      xx += rowX[latent] * rowX[latent];
      xy += rowX[latent] * rowY[latent];
      yy += rowY[latent] * rowY[latent];
    }
    const determinant = Math.max(1e-12, xx * yy - xy * xy);
    return {
      rowX,
      rowY,
      inv00: yy / determinant,
      inv01: -xy / determinant,
      inv10: -xy / determinant,
      inv11: xx / determinant,
    };
  }

  function solvePinDelta(pin, dx, dy) {
    const solveX = pin.inv00 * dx + pin.inv01 * dy;
    const solveY = pin.inv10 * dx + pin.inv11 * dy;
    const delta = new Float32Array(128);
    for (let latent = 0; latent < 128; latent += 1) {
      delta[latent] = pin.rowX[latent] * solveX + pin.rowY[latent] * solveY;
    }
    return delta;
  }

  function setLatentOffset(candidate) {
    let norm2 = 0;
    for (let latent = 0; latent < 128; latent += 1) {
      norm2 += candidate[latent] * candidate[latent];
    }
    const norm = Math.sqrt(norm2);
    const scale = norm > LATENT_TRUST_RADIUS ? LATENT_TRUST_RADIUS / norm : 1;
    const anchorZ = state.anchors[state.anchorIndex].z;
    for (let latent = 0; latent < 128; latent += 1) {
      const offset = candidate[latent] * scale;
      state.latentOffset[latent] = offset;
      state.z[latent] = anchorZ[latent] + offset;
    }
  }

  function latentTarget(pin, baseOffset, dx, dy) {
    const delta = solvePinDelta(pin, dx, dy);
    const candidate = new Float32Array(128);
    for (let latent = 0; latent < 128; latent += 1) {
      candidate[latent] = baseOffset[latent] + delta[latent];
    }
    setLatentOffset(candidate);
  }

  function localStep(weights, dx, dy) {
    for (let packet = 0; packet < PACKETS; packet += 1) {
      const weight = weights[packet];
      state.shifts[packet * 2] += dx * weight;
      state.shifts[packet * 2 + 1] += dy * weight;
    }
  }

  function beginDrag(event) {
    if (!state.ready || !state.params || (event.button !== undefined && event.button !== 0)) return;
    event.preventDefault();
    document.body.classList.add("interacted");
    ui.canvasWrap.classList.add("dragging");
    ui.canvas.setPointerCapture?.(event.pointerId);
    const point = pointerPosition(event);
    const weights = grabWeights(point);
    state.drag = {
      pointerId: event.pointerId,
      start: point,
      last: point,
      pointer: point,
      weights,
      pin: buildPin(weights),
      baseOffset: copyArray(state.latentOffset),
    };
    drawDisplay();
  }

  function moveDrag(event) {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const point = pointerPosition(event);
    const dx = point[0] - state.drag.last[0];
    const dy = point[1] - state.drag.last[1];
    state.drag.last = point;
    state.drag.pointer = point;

    if (state.mode === "manifold") {
      // Target-seeking, not an integrator: the same pointer location gives
      // the same latent target regardless of browser event rate.
      const targetX = point[0] - state.drag.start[0];
      const targetY = point[1] - state.drag.start[1];
      latentTarget(state.drag.pin, state.drag.baseOffset, targetX, targetY);
    } else {
      localStep(state.drag.weights, dx, dy);
    }

    const totalX = point[0] - state.drag.start[0];
    const totalY = point[1] - state.drag.start[1];
    state.maxPull = Math.max(state.maxPull, MODEL_SIZE * Math.hypot(totalX, totalY));
    updateMeters();
    drawDisplay();
    requestInference();
  }

  function endDrag(event) {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    ui.canvas.releasePointerCapture?.(event.pointerId);
    ui.canvasWrap.classList.remove("dragging");
    state.drag = null;
    rankPackets();
    drawDisplay();
  }

  function keyboardNudge(dx, dy) {
    if (!state.ready || !state.params) return;
    document.body.classList.add("interacted");
    if (!state.keyboardGrab) {
      const weights = grabWeights([.5, .43]);
      state.keyboardGrab = { weights, pin: buildPin(weights) };
    }
    if (state.mode === "manifold") {
      latentTarget(state.keyboardGrab.pin, copyArray(state.latentOffset), dx, dy);
    } else {
      localStep(state.keyboardGrab.weights, dx, dy);
    }
    state.maxPull += MODEL_SIZE * Math.hypot(dx, dy);
    updateMeters();
    requestInference();
  }

  async function shareResult() {
    const text = `I pulled a face ${Math.round(state.maxPull)} px with no skeleton, then left it with ${state.activeWaves} interference waves.`;
    const url = window.location.href.split("#")[0];
    try {
      if (navigator.share) {
        await navigator.share({ title: "This face has no skeleton", text, url });
      } else {
        await navigator.clipboard.writeText(`${text} ${url}`);
        flashShare("COPIED. SEND IT.");
      }
    } catch (error) {
      if (error?.name !== "AbortError") {
        fallbackCopy(`${text} ${url}`);
        flashShare("COPIED. SEND IT.");
      }
    }
  }

  function fallbackCopy(text) {
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.append(field);
    field.select();
    document.execCommand("copy");
    field.remove();
  }

  function flashShare(label) {
    const original = "SHARE THE DAMAGE";
    ui.share.textContent = label;
    window.setTimeout(() => { ui.share.textContent = original; }, 1600);
  }

  function bindEvents() {
    ui.canvas.addEventListener("pointerdown", beginDrag);
    ui.canvas.addEventListener("pointermove", moveDrag);
    ui.canvas.addEventListener("pointerup", endDrag);
    ui.canvas.addEventListener("pointercancel", endDrag);

    ui.modeButtons.forEach((button) => {
      button.addEventListener("click", () => setMode(button.dataset.mode));
    });
    ui.waveSlider.addEventListener("input", () => applyWaveCount(Number(ui.waveSlider.value)));
    ui.presets.forEach((button) => {
      button.addEventListener("click", () => applyWaveCount(Number(button.dataset.waves)));
    });
    ui.newFace.addEventListener("click", () => setAnchor(state.anchorIndex + 1));
    ui.resetFace.addEventListener("click", resetDeformation);
    ui.share.addEventListener("click", shareResult);

    ui.canvas.addEventListener("keydown", (event) => {
      const distance = event.shiftKey ? .035 : .014;
      const motion = {
        ArrowLeft: [-distance, 0],
        ArrowRight: [distance, 0],
        ArrowUp: [0, -distance],
        ArrowDown: [0, distance],
      }[event.key];
      if (motion) {
        event.preventDefault();
        keyboardNudge(...motion);
      }
    });

    window.addEventListener("keydown", (event) => {
      if (!state.ready) return;
      if (event.target instanceof HTMLInputElement) return;
      if (event.key.toLowerCase() === "r") resetDeformation();
      if (event.key.toLowerCase() === "n") setAnchor(state.anchorIndex + 1);
    });
    window.addEventListener("resize", drawDisplay, { passive: true });
  }

  function fail(error) {
    console.error(error);
    document.body.classList.remove("ready");
    document.body.classList.add("failed");
    document.querySelectorAll("button, input").forEach((control) => {
      control.disabled = true;
    });
    ui.engine.textContent = "MODEL DID NOT START";
    ui.loading.innerHTML = "<b>THE FACE DIDN'T ASSEMBLE</b><span>Refresh once, or try a current Chrome, Edge, Firefox, or Safari.</span>";
  }

  async function init() {
    bindEvents();
    try {
      if (!window.ort) throw new Error("ONNX Runtime did not load");
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
      ort.env.wasm.numThreads = 1;
      const [controlResponse] = await Promise.all([
        fetch("controls.bin"),
      ]);
      if (!controlResponse.ok) throw new Error("Control map could not be loaded");
      state.anchors = parseControls(await controlResponse.arrayBuffer());
      state.z = copyArray(state.anchors[0].z);
      state.session = await ort.InferenceSession.create("tinyavatar.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      requestInference();
    } catch (error) {
      fail(error);
    }
  }

  init();
})();