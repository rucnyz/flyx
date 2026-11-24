// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Model page, in two views behind tabs, since a model once loaded is mostly watched and tuned, seldom changed.
// "Current model": what the server serves (telemetry's "load") and, while it serves one, the settings a load takes
// (context, K/V, drafting) beside the command line that would load the same.  "Model library": where models come from, as a list and the one
// thing chosen in it: Hugging Face (GET /v1/hub/search, a repository's files from GET /v1/hub/files) or this machine
// (the cache and the model directories, GET /v1/hub/cached; any folder, GET /v1/hub/dir); one box loads the file
// chosen (POST /v1/load).  Fields an older server may not send are read defensively.
import { html, useState, useEffect, useRef } from "../lib/html.js";
import { saved, getJson, call } from "../lib/api.js";
import { t, tErr } from "../lib/i18n.js";
import { fmt, gib, gb, secs, shortDevice, baseName, psQuote, shQuote, compact, ago, contextLimit } from "../lib/format.js";
import { Icon, CodeSnippet, CopyButton, confirmBox, defaultShell, ShellPicker } from "../lib/ui.js";
import { live, poll, busyLoad } from "../lib/telemetry.js";
import { unloadServed, reloadServed, stopFetch } from "../lib/actions.js";
import { VramApps, isVramError, parseVramNeeded } from "../components/vram_apps.js";

// "org/name[:quant][@rev]", not a path: a relative path of two parts reads the same, but for its file's extension.
const hubName = (v) => /^(https?:\/\/(hf\.co|huggingface\.co)\/)?[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?(@[A-Za-z0-9._-]+)?$/.test(v.trim()) &&
                      !/\.(gguf|flyx)$/i.test(v.trim());
const repoOf = (v) => v.trim().replace(/^https?:\/\/(hf\.co|huggingface\.co)\//, "").split("@")[0].split(":")[0];
const quantOf = (v) => { const m = /:([^@]+)/.exec(v); return m ? m[1] : null; };

function describe(m) {
  const repo = m.repo || (m.source === "dir" ? null : repoOf(m.name));
  const quant = m.quant || (m.source !== "dir" ? quantOf(m.name) : null);
  return { repo, quant, file: m.file || (m.path ? baseName(m.path) : m.source === "dir" ? baseName(m.name) : null) };
}

// Drafters the server found or was given, from the status (or, from an older server, its notes).
function draftersOf(L) {
  if (Array.isArray(L.drafters)) return L.drafters;
  const out = [];
  for (const n of L.notes || []) { const m = /^the MTP layer from (\S+)/.exec(n); if (m) out.push({ kind: "mtp", from: m[1] }); }
  return out;
}

// The memory a new model may use: the GPU's, less what other programs hold (flyx's own model goes when it switches).
let gpuTotal = Number(saved.get("gpu_total")) || 0;
function deviceReserve(total) {
  const base = Math.max(384 * 1048576, total / 40);
  return /Windows/i.test(navigator.userAgent) ? Math.max(base, 832 * 1048576) : base;
}
function freeForModel() {
  const d = live.last && live.last.device, m = live.last && live.last.memory;
  if (!d || !d.mem_total) return gpuTotal || 0;
  // When a model is loaded, ledger_budget is the engine's real VRAM ceiling (already excluding other programs and the
  // device reserve, without miscounting flyx's own CUDA context / driver pool as other programs).
  if (m && m.ledger_budget) return Math.min(d.mem_total, m.ledger_budget + deviceReserve(d.mem_total));
  return Math.max(0, d.mem_total - (d.mem_used || 0));
}
function FitTag({ size }) {
  const d = live.last && live.last.device, m = live.last && live.last.memory;
  const room = freeForModel();
  if (!room || !size) return html`<span class="tag2">?</span>`;
  // Untied embeddings (~4% on 27B) stay in host RAM; ~0.85 GB covers workspace and one minimal session beyond reserve.
  const budget = m && m.ledger_budget ? m.ledger_budget : Math.max(0, room - deviceReserve(d && d.mem_total ? d.mem_total : room));
  const fits = size * 0.96 + 0.85e9 < budget;
  const title = "Estimate: " + gb(size) + " GB of weights + 1.5 GB against " + gib(room) + " GiB the GPU can give a model " +
                "(its memory, less other programs'). Long contexts need more; the engine's bill shows once it loads.";
  return html`<span class=${"tag2 " + (fits ? "ok" : "bad")} title=${t(title)}>${t(fits ? "fits" : "too big")}</span>`;
}

// What is served and its state, its facts, a fetch's or a load's progress, its drafters, an error.  With nothing
// served it is the view's whole content: `again` loads the model the server remembers, `browse` shows the library.
function Serving({ tm, again, retry, browse }) {
  const L = tm.load || { state: "none" }, model = tm.model || {}, p = L.progress || {};
  const loadedWant = model.id && L.loaded ? L.loaded : null, busy = busyLoad(L);
  const [msg, setMsg] = useState("");
  const [vramOpen, setVramOpen] = useState(false);
  const facts = [];
  if (model.loaded && !busy) {
    if (model.device) facts.push(["GPU", shortDevice(model.device)]);
    facts.push(["Context", fmt(model.context)]);
    if (loadedWant) facts.push(["KV cache", loadedWant.kv === "16" ? "16-bit" : loadedWant.kv === "fp8" ? "fp8" : "auto"]);
    facts.push(["Sessions", fmt(model.max_sessions)]);
    facts.push(["Drafting", model.drafter ? model.drafter + ", up to " + model.max_width : "off"]);
    facts.push(["Vision", model.vision ? t(normVisionDev(model.vision_device)) + " · " + normVisionQuant(model.vision_quant) : "off"]);
    if (model.load_s) facts.push(["Load time", secs(model.load_s)]);
    if (tm.memory && tm.memory.ledger_used) facts.push(["Memory", gib(tm.memory.ledger_used) + " GiB"]);
  }
  let sub = "";
  if (L.state === "fetching" && !p.total) sub = t("asking Hugging Face which files it needs");
  else if (L.state === "fetching") {
    const left = p.bytes_per_s > 0 ? (p.total - p.done) / p.bytes_per_s : null;
    sub = (p.file || "") + "  " + gb(p.done || 0) + " of " + gb(p.total || 0) + " GB" +
      (p.bytes_per_s ? " · " + fmt(p.bytes_per_s / 1e6) + " MB/s" : "") + (left !== null && !p.checking ? " · " + secs(left) + " left" : "") +
      (p.checking ? " · checking its sha256" : "") + (model.id ? " · " + model.id + " keeps serving meanwhile" : "");
  } else if (L.state === "loading") sub = t("loading onto the GPU · " + secs(L.elapsed_s));
  const drafters = model.loaded && !busy ? draftersOf(L).slice() : [];
  if (model.loaded && !busy && model.drafter && /mtp/.test(model.drafter) && !drafters.some((d) => d.kind === "mtp"))
    drafters.unshift({ kind: "mtp", from: "model file" });
  const limit = model.loaded && !busy ? contextLimit(model) : null;  // memory cut the context: why, and what gives more
  const ctxLimit = model.context_limit;
  const ctxMemKind = !limit || !ctxLimit ? ""
    : ctxLimit.by === "host" ? "ram"
    : ctxLimit.by === "gpu" && ctxLimit.closable ? "vram" : "";
  const vramErr = !busy && isVramError(L.error);
  const showMemPanel = vramErr || (!!ctxMemKind && vramOpen);
  const panelKind = vramErr ? "vram" : (ctxMemKind || "vram");
  const stateName = { none: "Nothing loaded", fetching: "Fetching", loading: "Loading", ready: "Serving", failed: "Failed" }[L.state] || L.state;
  const tone = L.state === "ready" ? "ok" : L.state === "failed" ? "bad" : busy ? "warn" : "";
  const idle = !model.id && !busy;
  return html`
    <div class=${"srv" + (idle ? " idle" : "")}>
      <div class=${"srv-lamp " + tone}><${Icon} name="model"/></div>
      <div class="srv-what">
        <span class=${"badge " + tone} id="ld-state"><span class="dot"></span>${t(stateName)}</span>
        <div class="srv-name" id="ld-now">${busy ? L.target : model.id ? model.id : t("No model loaded")}</div>
        <div class=${"srv-src" + (idle ? " hint" : "")} id="ld-src">${busy ? "" : loadedWant && loadedWant.model !== model.id ? loadedWant.model :
          model.id ? "" : t("Pick one in the model library: from Hugging Face, or one on this machine.")}</div>
      </div>
      <div class="srv-acts">
        <button class="btn" id="ld-cancel" hidden=${L.state !== "fetching"} onClick=${stopFetch}>${t("Stop fetching")}</button>
        <button class="btn danger" id="ld-unload" hidden=${!model.id || busy}
                onClick=${async () => setMsg((await unloadServed()) || "")}><${Icon} name="eject"/>${t("Unload")}</button>
      </div>
      ${idle && again ? html`<div class="srv-last" id="ld-last">${t("Loaded last")} <b>${again.model}</b></div>` : ""}
      ${idle ? html`<div class="srv-go">
        ${again ? html`<button class="btn primary" id="ld-again" title=${again.model} onClick=${again.load}>
          <${Icon} name="play"/>${t("Load the last model again")}</button>` : ""}
        <button class=${"btn" + (again ? "" : " primary")} id="ld-browse" onClick=${browse}>${t("Explore the model library")}</button></div>` : ""}
      <dl class="stats" id="ld-facts" hidden=${!facts.length}>${facts.map(([k, v]) => html`<div><dt>${t(k)}</dt><dd>${t(v)}</dd></div>`)}</dl>
      <div class=${"banner " + (limit && limit.short ? "warn" : "hint")} id="ld-ctx" hidden=${!limit}>
        <div class="grow2">${limit ? limit.parts.map((s) => t(s)).join(" ").replace(/([。！？]) /g, "$1") : ""}</div>
        ${ctxMemKind ? html`<button class="btn sm" id="ld-vram-toggle" onClick=${() => setVramOpen((v) => !v)}>
          ${t(vramOpen ? "Hide apps" : ctxMemKind === "ram" ? "Manage RAM" : "Manage VRAM")}</button>` : ""}</div>
      <div class=${"bar" + (L.state === "loading" ? " busy" : "")} id="ld-prog" hidden=${!busy}>
        <i style=${{ width: L.state === "loading" ? "" : (p.total ? 100 * p.done / p.total : 0) + "%" }}></i></div>
      <div class="prog" id="ld-sub">${sub}</div>
      <div class="srv-dr" id="ld-drafters">${drafters.map((d) => html`
        <div class="dr"><span class="k">${drafterLabel(d.kind)}</span>
          <span class="f">${(d.from === "model file" ? t("the model file's own layer") : d.from) + (d.size ? " · " + gb(d.size) + " GB" : "")}</span></div>`)}
        ${model.loaded && !busy && model.vision ? html`
        <div class="dr"><span class="k">${t("Vision")}</span>
          <span class="f">${(L.vision_projector && L.vision_projector.from ? L.vision_projector.from : t("model file")) +
            (L.vision_projector && L.vision_projector.size ? " · " + gb(L.vision_projector.size) + " GB" : "")}</span></div>` : ""}</div>
      <div class=${"banner " + (L.fallback ? "warn" : "bad")} id="ld-err" hidden=${!L.error && !msg}>
        ${L.error ? (L.fallback ? [t("The new model did not load."), tErr(L.error), t(L.fallback + " was loaded again.")].join(" ").replace(/([。！？]) /g, "$1") : tErr(L.error)) : t(msg)}</div>
      <${VramApps} open=${showMemPanel} kind=${panelKind} id="ld-vram-apps"
                   needBytes=${vramErr ? parseVramNeeded(L.error) : 0}
                   onRetry=${vramErr ? retry : (model.id && loadedWant ? reloadServed : null)}
                   retryLabel=${vramErr ? "Retry loading" : "Reload model"}
                   onClose=${!vramErr ? () => setVramOpen(false) : null}/>
    </div>`;
}

// A drafter given as a GGUF is an MTP layer; a block drafter's kind by its name as published (org/X-DFlash2,
// org/X-DFlash-b16, org/X-DSpine-b16), else the engine's choice from what the file holds ("": auto).
const kindOf = (d) => /\.gguf$/i.test(d) ? "mtp" : /dspine/i.test(d) ? "dspine" : /dflash2/i.test(d) ? "dflash2" :
                      /dflash/i.test(d) ? "dflash" : "";
const drafterLabel = (k) => ({ mtp: "MTP", dflash2: "DFlash2", dflash: "DFlash", dspine: "DSpine" })[k] || k;
const publishedNote = { dspine: "A DSpine drafter is published for this model.",
                        dflash2: "A DFlash2 drafter is published for this model.",
                        dflash: "A DFlash drafter is published for this model." };
const normVisionDev = (v) => (v === "host" || v === "cpu") ? "host" : v === "gpu" ? "gpu" : v === "auto" ? "auto" : "disk";
const normVisionQuant = (v) => (v === "exact" || v === "bf16") ? "exact" : (v === "nvfp4" || v === "fp4") ? "nvfp4" : "q8";

// The settings of a want as the form shows them, for comparing with what is loaded.
function shown(w) {
  const vOn = w.vision !== false;
  return { ctx: w.ctx || "auto", kv: w.kv === "16" ? "16-bit" : w.kv === "fp8" ? "fp8" : "auto",
           kv_ram: typeof w.kv_ram === "number" ? w.kv_ram + " GB" : "auto", kv_gpu: typeof w.kv_gpu === "number" ? fmt(w.kv_gpu) : "as many as fit",
           name: w.name || "the file's name",
           spec: w.speculative === false ? "off" : "on", drafter: [w.drafter || "default", ...(w.drafters || [])],
           width: w.width === null || w.width === undefined ? "auto" : String(w.width),
           head: w.draft_head === "full" ? "full head" : "1-bit pick", weights: w.draft_weights === "exact" ? "exact" : "compact copy",
           lookup: w.lookup !== false ? "on" : "off", trees: w.draft_trees ? "on" : "off",
           vision: vOn ? "on" : "off", mmproj: vOn ? (w.mmproj || "auto") : "auto",
           vision_device: normVisionDev(w.vision_device), vision_quant: normVisionQuant(w.vision_quant),
           vision_max_tokens: w.vision_max_tokens > 0 ? String(w.vision_max_tokens) : "auto" };
}

function draftingOf(w) {
  const files = w.drafters || [], mode = w.drafter || "", own = files.length === 1 ? files[0] : "";
  // The single custom-file control also selects its kind. Keep configurations it cannot express as a separate choice.
  const custom = w.speculative !== false && files.length === 1 && own === own.trim() && mode === kindOf(own);
  const savedDraft = !custom && (mode || files.length) ? { drafter: mode, drafters: files } : null;
  return { draftSource: savedDraft ? "saved" : custom ? "custom" : "default", savedDraft, drafter: own };
}

// The form's fields from a want ({model, ctx, kv, drafters, ...}).
function formOf(w) {
  return { ctx: w.ctx || "", kv: w.kv === "16" || w.kv === "fp8" ? w.kv : "", kvram: typeof w.kv_ram === "number" ? String(w.kv_ram) : "",
           kvgpu: typeof w.kv_gpu === "number" ? String(w.kv_gpu) : "", name: w.name || "",
           spec: w.speculative !== false, ...draftingOf(w),
           width: w.width === null || w.width === undefined ? "" : String(w.width), trees: !!w.draft_trees, lookup: w.lookup !== false,
           head: w.draft_head === "full" ? "full" : "", weights: w.draft_weights === "exact" ? "exact" : "",
           vision: w.vision !== false, visionSource: w.mmproj ? "custom" : "default", mmproj: w.mmproj || "",
           vision_device: normVisionDev(w.vision_device), vision_quant: normVisionQuant(w.vision_quant),
           vision_max_tokens: w.vision_max_tokens > 0 ? String(w.vision_max_tokens) : "" };
}
const EMPTY_FORM = formOf({});

// What the panel asks of POST /v1/load.
function loadBody(chosen, f) {
  const ctxText = String(f.ctx).trim().toLowerCase(), ctxTokens = parseInt(ctxText, 10);
  const ctx = ctxText === "max" ? "max" : ctxTokens > 0 ? ctxTokens : null;  // "max": the model's own, or the longest that fits
  const kvRam = parseFloat(f.kvram), kvGpu = parseInt(f.kvgpu, 10), name = f.name.trim();
  const own = f.draftSource === "custom" && f.spec ? f.drafter.trim() : "";
  const adv = { width: f.width ? parseInt(f.width, 10) : null, draft_trees: f.trees, lookup: f.lookup,
                draft_head: f.head || null, draft_weights: f.weights || null };
  const draft = f.draftSource === "saved" ? f.savedDraft : { drafter: own ? kindOf(own) : "", drafters: own ? [own] : [] };
  const mmproj = f.vision && f.visionSource === "custom" && f.mmproj.trim() ? f.mmproj.trim() : null;
  const vmaxRaw = String(f.vision_max_tokens).trim();
  const vmax = vmaxRaw ? parseInt(vmaxRaw, 10) : NaN;
  const vis = { vision: f.vision, mmproj, vision_device: normVisionDev(f.vision_device),
                vision_quant: normVisionQuant(f.vision_quant),
                vision_max_tokens: Number.isFinite(vmax) && vmax > 0 ? vmax : null };
  // kv "auto": the engine's default, not the server's --kv (null would keep that)
  return Object.assign({ model: chosen, ctx, kv: f.kv || "auto", kv_ram: kvRam >= 0 ? kvRam : "auto",
                         kv_gpu: kvGpu >= 0 ? kvGpu : null, name: name || null, speculative: f.spec }, draft, adv, vis);
}

// The command line that loads the same: what a user can put in a script.
function commandOf(b, shell) {
  const ps = shell === "powershell";
  // Windows PowerShell 5.1 drops literal double quotes and can consume a trailing backslash
  // when joining native arguments that contain spaces. Pass the native command line in these cases.
  const strings = [b.model, b.name, b.mmproj, ...b.drafters];
  const native = ps && strings.some((s) => s && (s.includes('"') || (/\s/.test(s) && s.endsWith("\\"))));
  const nativeQuote = (s) => '"' + s.replace(/(\\*)"/g, '$1$1\\"').replace(/\\+$/, '$&$&') + '"';
  const q = native ? nativeQuote : ps ? psQuote : shQuote;
  const parts = [b.model ? q(b.model) : "MODEL"];
  if (b.ctx) parts.push("--ctx " + b.ctx);
  if (b.kv !== "auto") parts.push("--kv " + b.kv);
  if (b.kv_ram !== "auto") parts.push("--kv-ram " + b.kv_ram);
  if (b.kv_gpu !== null) parts.push("--kv-gpu " + b.kv_gpu);
  if (!b.speculative) parts.push("--no-spec");
  if (b.drafter) parts.push("--drafter " + b.drafter);
  for (const d of b.drafters) parts.push("--drafter-file " + q(d));
  if (b.width !== null) parts.push("--width " + b.width);
  if (b.draft_trees) parts.push("--draft-trees");
  if (!b.lookup) parts.push("--no-lookup");
  if (b.draft_head) parts.push("--backend-opt draft_head=" + b.draft_head);
  if (b.draft_weights) parts.push("--backend-opt draft_weights=" + b.draft_weights);
  if (b.vision === false) parts.push("--no-vision");
  if (b.mmproj) parts.push("--mmproj " + q(b.mmproj));
  if (b.vision !== false && b.vision_device && b.vision_device !== "disk") parts.push("--vision-device " + b.vision_device);
  if (b.vision !== false && b.vision_quant && b.vision_quant !== "q8") parts.push("--vision-quant " + b.vision_quant);
  if (b.vision !== false && b.vision_max_tokens) parts.push("--vision-max-tokens " + b.vision_max_tokens);
  if (b.name) parts.push("--name " + q(b.name));
  const port = location.port || (location.protocol === "https:" ? "443" : "80");
  if (port !== "8080") parts.push("--port " + port);
  if (native) return "Start-Process flyx -NoNewWindow -Wait -ArgumentList " + psQuote("serve " + parts.join(" "));
  const continuation = ps ? " `\n    " : " \\\n    ";
  let s = "flyx serve " + parts[0];
  for (const p of parts.slice(1)) s += (s.length - s.lastIndexOf("\n") + p.length > 60 ? continuation : " ") + p;
  return s;
}

// A labelled field of the form: `k` names it in the comparison with what is loaded.
const Field = ({ k, label, was, hint, wide, children }) => html`
  <label class=${"field" + (was ? " changed" : "")} data-k=${k} style=${wide ? "grid-column: 1 / -1" : ""}>
    <span>${t(label)}<span class="was">${was ? t("was " + t(was)) : ""}</span></span>${children}${hint ? html`<em>${t(hint)}</em>` : ""}</label>`;

// The orders GET /v1/hub/search lists the Hub's repositories in.
const SORTS = [["trending", "Trending"], ["downloads", "Most downloaded"], ["likes", "Most liked"], ["updated", "Recently updated"],
               ["created", "Newest"]];
// The folders kept in this browser's list of places (a convenience of the viewer's: the server knows only its own
// --models-dir).
function savedFolders() {
  try { const v = JSON.parse(saved.get("folders") || "[]"); return Array.isArray(v) ? v.filter((x) => typeof x === "string") : []; }
  catch (e) { return []; }
}
const folderOf = (m) => (m.path || m.name).replace(/[\\/][^\\/]*$/, "");
const samePath = (a, b) => a.replace(/[\\/]+$/, "").toLowerCase() === b.replace(/[\\/]+$/, "").toLowerCase();
const whole = (g) => !(g.split && g.complete === false);
const sized = (b) => b >= 1e8 ? gb(b) + " GB" : Math.max(1, Math.round(b / 1e6)) + " MB";
// An answer of the server's newer endpoints, or why there is none: an older server does not know them.
async function ask(path, what) {
  const r = await call("GET", path);
  if (r.ok) return { j: r.json };
  return { error: r.status === 404 && !/no such|not a model/.test(r.error || "") ? "This server cannot " + what + " yet: update FLYX." : r.error };
}

export function ModelPage({ active, libraryAsked, currentAsked }) {
  const tm = live.last || {}, L = tm.load || { state: "none" }, model = tm.model || {};
  const loadedWant = model.id && L.loaded ? L.loaded : null, busy = busyLoad(L);
  const [tab, setTabState] = useState(() => saved.get("models_tab") === "local" ? "local" : "hub");
  // Hugging Face: what was searched for, the repositories found, the one shown
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("trending");
  const [found, setFound] = useState(null);
  const [foundMsg, setFoundMsg] = useState("");
  const [repoSel, setRepoSel] = useState("");
  const [hub, setHub] = useState(null);  // {repo, j}: GET /v1/hub/files of the repository shown
  const [hubMsg, setHubMsg] = useState("");
  // this machine: the models the server lists, the folders kept here, the folder shown (`loc`; "" for where to start)
  const [catalog, setCatalog] = useState(null);
  const [catalogErr, setCatalogErr] = useState("");
  const [folders, setFoldersState] = useState(savedFolders);
  const [loc, setLoc] = useState("");
  const [dir, setDir] = useState(null);  // GET /v1/hub/dir of `loc`
  const [dirMsg, setDirMsg] = useState("");
  const [pathText, setPathText] = useState("");
  const [localQ, setLocalQ] = useState("");
  const [picked, setPicked] = useState("");  // the model of the list shown at the right; "": the folder at `loc`
  const [listings, setListings] = useState({});  // a folder the server does not list by itself -> {models} of GET /v1/hub/dir
  // the model to load, and how
  const [target, setTarget] = useState("");
  const [f, setF] = useState(EMPTY_FORM);
  // The view shown: "current" (the model served, the load settings) or "library".  The page opens on the model while
  // one is served or coming or its load failed, else on the library; a load started from the library shows the model
  // again.
  const [view, setView] = useState("current");
  const viewed = useRef(false);
  useEffect(() => { if (libraryAsked) setView("library"); }, [libraryAsked]);  // another page's way here
  useEffect(() => { if (currentAsked) setView("current"); }, [currentAsked]);  // and to the model's card
  const [advOpen, setAdvOpen] = useState(false);
  const [stab, setStab] = useState("general");
  const [shell, setShell] = useState(defaultShell);
  const [msg, setMsg] = useState("");
  const filled = useRef(false), sawState = useRef(""), sawLoaded = useRef("");
  const searchSeq = useRef(0), repoSeq = useRef(0), dirSeq = useRef(0), placed = useRef(false);
  const lastAttempt = useRef(null);
  const set = (k) => (e) => setF(Object.assign({}, f, { [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));
  const setTab = (k) => { setTabState(k); saved.set("models_tab", k); setMsg(""); };
  // `next`: the list, or what makes it of the list as it is by then (an answer that took long to come).
  const setFolders = (next) => setFoldersState((old) => {
    const list = typeof next === "function" ? next(old) : next;
    saved.set("folders", list.length ? JSON.stringify(list) : null);
    return list;
  });
  const [picking, setPicking] = useState(false);  // the desktop's folder dialog is open, on the server's machine

  async function refreshCached() {
    try { setCatalog(await getJson("/v1/hub/cached")); setCatalogErr(""); } catch (e) { setCatalogErr(e.message); }
  }
  // ---- Hugging Face ----
  async function search(text, by) {
    const n = ++searchSeq.current;
    setFoundMsg("Asking Hugging Face ...");
    const word = hubName(text) ? repoOf(text) : text.trim();  // a name with its quantization or revision: its repository
    const a = await ask("/v1/hub/search?limit=40&sort=" + by + "&q=" + encodeURIComponent(word), "search Hugging Face");
    if (n !== searchSeq.current) return;
    if (a.error) { setFound(null); setFoundMsg(a.error); return; }
    setFound(a.j);
    setFoundMsg("");
  }
  // A repository's files and what the Hub says of it.  `quant`: the file to choose once they are known.
  async function showRepo(id, quant) {
    const n = ++repoSeq.current;
    setRepoSel(id);
    setHubMsg("Asking Hugging Face about " + id + " ...");
    setMsg("");
    const a = await ask("/v1/hub/files?repo=" + encodeURIComponent(id), "list a repository");
    if (n !== repoSeq.current) return;
    if (a.error) { setHub(null); setHubMsg(a.error); return; }
    setHub({ repo: id, j: a.j });
    setHubMsg(a.j.from_cache ? "From the local cache: Hugging Face was not reached." : "");
    const files = (a.j.ggufs || []).filter(whole);
    const hit = quant ? files.find((g) => String(g.quant).toLowerCase() === quant.toLowerCase()) : (a.j.ggufs || []).length === 1 ? files[0] : null;
    if (hit) setTarget(hit.name);
  }
  // The search box also takes a repository's whole name (org/name, org/name:QUANT, its address): shown at once.
  function named(text) {
    const v = text.trim();
    if (!hubName(v)) return false;
    showRepo(repoOf(v), quantOf(v.replace(/^https?:\/\/(hf\.co|huggingface\.co)\//, "").split("@")[0]) || "");
    return true;
  }
  useEffect(() => {
    if (!active || view !== "library" || tab !== "hub") return;
    const h = setTimeout(() => search(q, sort), found || foundMsg ? 350 : 0);
    return () => clearTimeout(h);
  }, [active, view, tab, q, sort]);
  // The first repository shown: the served model's, else the first found.
  useEffect(() => {
    if (repoSel || view !== "library" || tab !== "hub" || !found) return;
    const served = loadedWant && hubName(loadedWant.model) ? repoOf(loadedWant.model) : "";
    const first = served || (found.models[0] && found.models[0].id);
    if (first) showRepo(first, "");
  }, [found, tab, view]);

  // ---- this machine ----
  async function openPath(p) {
    const n = ++dirSeq.current;
    setLoc(p);
    setPathText(p);
    setMsg("");
    // The folder Hugging Face downloads to is shown by its repositories, as the server lists them, not by its files as
    // they lie in it.
    if (catalog && p && samePath(p, catalog.cache)) { setDir(null); setDirMsg(""); return; }
    setDirMsg("Looking...");
    const a = await ask("/v1/hub/dir?path=" + encodeURIComponent(p), "browse folders");
    if (n !== dirSeq.current) return;
    if (a.error) { setDir(null); setDirMsg(a.error); return; }
    setDir(a.j);
    setDirMsg("");
    setLoc(a.j.path);
    setPathText(a.j.path);
    if (a.j.kind === "file" && a.j.models.length === 1 && a.j.models[0].kind !== "mmproj") setTarget(a.j.models[0].name);
  }
  const kept = folders.some((x) => samePath(x, loc));
  const keep = () => setFolders(kept ? folders.filter((x) => !samePath(x, loc)) : [...folders, loc]);
  // A folder for the list, chosen in the desktop's own folder dialog.  The server shows it, on its machine: a page
  // cannot learn from the browser's where a folder is.  Where none can be shown (the browser is on another machine,
  // there is no desktop), the folders are browsed in the page instead.
  async function addFolder() {
    if (picking) return;
    setPicking(true);
    setMsg("");
    const r = await call("POST", "/v1/hub/dir", { title: t("Choose a folder of model files"), start: loc });
    setPicking(false);
    if (r.ok) {
      const j = r.json;
      if (!j || !j.path) return;  // the dialog was closed without a choice
      setFolders((old) => [...known.keys(), ...old].some((x) => samePath(x, j.path)) ? old : [...old, j.path]);
      dirSeq.current++;
      setLoc(j.path);
      setPathText(j.path);
      setDir(j);
      setPicked("");
      setListings((old) => Object.assign({}, old, { [j.path]: { models: j.models } }));
      setDirMsg("");
      return;
    }
    setPicked("");  // the folders, at the right, in place of a model
    await openPath("");
    setDirMsg(r.status === 409 ? r.error : "The system's folder dialog cannot be shown from here: type the folder's path, or browse to it.");
  }
  // The first folder shown: the served model's when it is a file, else the one Hugging Face downloads to.
  useEffect(() => {
    if (placed.current || view !== "library" || tab !== "local" || !catalog) return;
    placed.current = true;
    const served = loadedWant && !hubName(loadedWant.model) && /[\\/]/.test(loadedWant.model) ? folderOf({ name: loadedWant.model }) : "";
    openPath(served || catalog.cache);
    const first = loadedWant ? loadedWant.model : (catalog.models.find((m) => m.source === "hub" && m.kind !== "drafter" && m.kind !== "mmproj") || {}).name;
    if (first) { setPicked(first); setTarget(first); }
  }, [catalog, tab, view]);

  useEffect(() => { if (active) refreshCached(); }, [active]);
  useEffect(() => {
    if (tm.device && tm.device.mem_total && tm.device.mem_total !== gpuTotal) {
      gpuTotal = tm.device.mem_total;
      saved.set("gpu_total", String(gpuTotal));
    }
    // Decided on the first answer this draw was made from: `model` is that draw's, and telemetry may have come since.
    if (!viewed.current && "model" in tm) {
      viewed.current = true;
      if (!model.id && !busy && !L.error) setView("library");  // a load that failed shows why first
    }
    // The settings start from what is loaded, else what the server remembers; and follow a model loaded since, from
    // this page or from elsewhere.
    const now = loadedWant ? JSON.stringify(loadedWant) : "";
    if (!filled.current && (L.loaded || L.remembered)) { setF(formOf(L.loaded || L.remembered)); filled.current = true; }
    else if (now && now !== sawLoaded.current && !busy) setF(formOf(loadedWant));
    if (!busy) sawLoaded.current = now;
    if (sawState.current && busyLoad({ state: sawState.current }) && !busy) refreshCached();
    sawState.current = L.state;
  }, [live.last]);

  const entryOf = (name) => catalog && catalog.models.find((m) => m.name === name);
  async function removeCached(m) {
    const d = describe(m);
    const ok = await confirmBox("Delete " + (d.quant || d.file || m.name) + "?",
      [(d.file || m.name) + " (" + gb(m.size) + " GB) is removed from the Hugging Face cache, freeing its space.",
       "It can be downloaded again by loading the same name."], "Delete", true);
    if (!ok) return;
    const r = await call("DELETE", "/v1/hub/cached?name=" + encodeURIComponent(m.name));
    if (!r.ok) { setMsg(r.status === 404 && !r.json ? "This server cannot delete files yet." : r.error); return; }
    if (target === m.name) setTarget("");
    if (picked === m.name) setPicked("");
    refreshCached();
    if (hub) showRepo(hub.repo, "");
  }

  // Every file the cache holds of a repository, deleted: `list`, its models as the server lists them.
  async function removeRepo(name, list) {
    const ok = await confirmBox("Delete " + name + "?",
      [list.length + (list.length === 1 ? " file" : " files") + " (" + sized(sum(list)) + ") are removed from the Hugging Face cache, freeing their space.",
       "It can be downloaded again by loading the same name."], "Delete", true);
    if (!ok) return;
    let failed = "";
    for (const m of list) {
      const r = await call("DELETE", "/v1/hub/cached?name=" + encodeURIComponent(m.name));
      if (!r.ok) failed = r.status === 404 && !r.json ? "This server cannot delete files yet." : r.error;
      else if (target === m.name) setTarget("");
    }
    if (failed) await confirmBox("Not everything was deleted", [failed], "OK");
    refreshCached();
    if (hub) showRepo(hub.repo, "");
  }

  // ---- the settings a load takes: those shown with the model served (a change is compared with what it has); with
  // none served there are none to show, and a load takes the defaults ----
  const use = model.id ? f : EMPTY_FORM;
  const body = (name) => loadBody(name, use);
  const before = loadedWant ? shown(loadedWant) : null, now = shown(body(loadedWant ? loadedWant.model : ""));
  const changed = before ? Object.keys(now).filter((k) => k === "drafter"
    ? before[k].length !== now[k].length || before[k].some((value, i) => value !== now[k][i]) : before[k] !== now[k]) : [];
  const was = (k) => changed.includes(k) ? (k === "drafter" ? before[k].join(" · ") : before[k]) : null;
  const advChanged = ["width", "head", "weights", "lookup", "trees"].filter((k) => was(k) !== null);
  useEffect(() => { if (advChanged.length) setAdvOpen(true); }, [advChanged.length]);

  // Loads `name` with the settings; or `want`, a load the server remembers, as it was.
  async function loadModel(name, want) {
    if (!name) { setMsg("Choose a model first."); return; }
    const same = !!loadedWant && name === loadedWant.model;
    const st = tm.state || {}, served = model.id;
    if (served) {
      const lines = [];
      const n = (st.running || 0) + (st.queued || 0);
      if (n) lines.push(n + (n === 1 ? " request is" : " requests are") + " running or waiting: " + (n === 1 ? "it is" : "they are") + " stopped.");
      lines.push(served + " is unloaded first; its GPU memory comes back before the new one loads.");
      if (hubName(name) && !entryOf(name)) lines.push("The files are fetched first, while " + served + " keeps answering.");
      lines.push("fallback" in L ? "If the new one fails to load, " + served + " is loaded again." :
                 "If the new one fails to load, no model is served until another is loaded.");
      if (!await confirmBox(same ? "Reload " + served + " with these settings?" : "Switch to " + (describe(entryOf(name) || { name }).quant || name) + "?",
                            lines, same ? "Reload" : "Switch")) return;
    }
    setMsg("");
    const reqBody = want || body(name);
    lastAttempt.current = reqBody;
    const r = await call("POST", "/v1/load", reqBody);
    if (!r.ok) { setMsg(r.error); return; }
    setView("current");  // where the fetch's and the load's progress show
    document.getElementById("main").scrollTo({ top: 0 });
    poll();
  }

  // The default drafter, as the settings say it: what the server loaded with the model served, when that drafts with
  // its default; else what a model brings along.
  function defaultDrafter() {
    if (loadedWant && loadedWant.speculative !== false && !loadedWant.drafter && !loadedWant.drafters?.length) {
      const list = draftersOf(L).map((d) => drafterLabel(d.kind) + " · " +
        (d.from === "model file" ? t("the model file's own layer") : t("from " + d.from + (d.size ? " (" + gb(d.size) + " GB)" : ""))));
      if (!list.length && model.drafter) list.push(model.drafter.toUpperCase() + " · " + t("the model file's own layer"));
      return list.length ? { tag: "in use", tone: "ok", lines: list } : { tag: "none", tone: "", lines: [t("This model loaded no drafter: plain decoding.")] };
    }
    return { tag: "", tone: "", lines: [t("The model's own MTP layer, if it has one.")] };
  }
  const dflt = defaultDrafter(), own = f.drafter.trim();
  const ownCached = !!own && !!catalog && catalog.models.some((x) => x.kind === "drafter" && x.name === own);
  const specOff = !f.spec, custom = f.draftSource === "custom";
  const savedDraft = f.draftSource === "saved" ? f.savedDraft : draftingOf(loadedWant || L.remembered || {}).savedDraft;
  const pickDraft = (source) => { if (!specOff) setF(Object.assign({}, f, { draftSource: source, savedDraft })); };
  // A drafter named by the page (one of the cache, one published for a repository's model) goes to the settings.
  const draftWith = (name) => {
    setF((current) => Object.assign({}, current, { spec: true, draftSource: "custom", drafter: name }));
    setView("current");
    setStab("spec");
    setTimeout(() => { const row = document.getElementById("sd-custom"); if (row) row.scrollIntoView({ behavior: "smooth", block: "center" }); }, 60);
  };
  const projectWith = (path) => {
    setF((current) => Object.assign({}, current, { vision: true, visionSource: "custom", mmproj: path }));
    setView("current");
    setStab("vision");
    setTimeout(() => { const row = document.getElementById("vs-custom"); if (row) row.scrollIntoView({ behavior: "smooth", block: "center" }); }, 60);
  };
  const visOff = !f.vision, visCustom = f.visionSource === "custom", mmOwn = f.mmproj.trim();
  const pickVision = (source) => { if (!visOff) setF(Object.assign({}, f, { visionSource: source })); };
  const visDefault = () => {
    if (loadedWant && loadedWant.vision !== false && !loadedWant.mmproj) {
      if (model.vision && L.vision_projector && L.vision_projector.from)
        return { tag: "in use", tone: "ok", line: t("from " + L.vision_projector.from + (L.vision_projector.size ? " (" + gb(L.vision_projector.size) + " GB)" : "")) };
      if (model.vision) return { tag: "in use", tone: "ok", line: t("Loaded from the model file.") };
      return { tag: "none", tone: "", line: t("No vision projector loaded: text-only.") };
    }
    return { tag: "", tone: "", line: t("Auto-loads mmproj from Hugging Face if available.") };
  };
  const vdflt = visDefault();
  const drafting = !use.spec ? "off" : use.draftSource === "custom" && use.drafter.trim() ? baseName(use.drafter.trim()) :
                   use.draftSource === "saved" ? "last loaded" : "default";
  const visionBrief = !use.vision ? "off" : t(normVisionDev(use.vision_device)) + " · " + normVisionQuant(use.vision_quant);
  const brief = [["context", String(now.ctx)], ["KV", now.kv], ["drafting", drafting], ["vision", visionBrief]];
  const genChg = ["ctx", "kv", "kv_ram", "kv_gpu", "name"].filter((k) => changed.includes(k)).length;
  const specChg = ["spec", "drafter", "width", "head", "weights", "lookup", "trees"].filter((k) => changed.includes(k)).length;
  const visChg = ["vision", "mmproj", "vision_device", "vision_quant", "vision_max_tokens"].filter((k) => changed.includes(k)).length;

  function settingsView(cmdModel) {
    const specSum = !f.spec ? t("off") : f.draftSource === "custom" && own ? baseName(own) : f.draftSource === "saved" ? t("last loaded") : t("default");
    const visSum = !f.vision ? t("off") : t(normVisionDev(f.vision_device)) + " · " + normVisionQuant(f.vision_quant);
    return html`
      <div class="pane lset" id="mf-settings">
        <div class="lset-h">
          <div><h2>${t("Load settings")}</h2>
            <p class="lead">${t(loadedWant ? "What the next model loads with. The model being served takes a change when it is reloaded."
                                           : "What the next model loads with.")}</p></div>
          ${changed.length ? html`<span class="was">${t(changed.length + " changed")}</span>
            <button class="linkbtn" id="mf-reset" onClick=${() => { setF(formOf(loadedWant)); setMsg(""); }}>${t("Reset to what is loaded")}</button>
            <button class="btn primary" id="mf-reload" disabled=${busy} onClick=${() => loadModel(loadedWant.model)}>
              ${t("Reload with these settings")}</button>` : ""}
        </div>
        ${msg ? html`<div class="banner bad" id="mf-setmsg" style="margin-bottom: 12px">${t(msg)}</div>` : ""}
        <div class="lset-nav">
          <div class="seg lset-seg" role="tablist" id="lset-tabs">
            <button role="tab" data-stab="general" aria-selected=${String(stab === "general")} onClick=${() => setStab("general")}>
              <span>${t("Configuration")}</span>
              ${genChg ? html`<span class="chg">+${genChg}</span>` : ""}
            </button>
            <button role="tab" data-stab="spec" aria-selected=${String(stab === "spec")} onClick=${() => setStab("spec")}>
              <span class=${"dot" + (f.spec ? " on" : "")}></span>
              <span>${t("Speculative decoding")}</span>
              <span class="sum">${specSum}</span>
              ${specChg ? html`<span class="chg">+${specChg}</span>` : ""}
            </button>
            <button role="tab" data-stab="vision" aria-selected=${String(stab === "vision")} onClick=${() => setStab("vision")}>
              <span class=${"dot" + (f.vision ? " on" : "")}></span>
              <span>${t("Vision encoder")}</span>
              <span class="sum">${visSum}</span>
              ${visChg ? html`<span class="chg">+${visChg}</span>` : ""}
            </button>
          </div>
        </div>
        <div class="lset-panel" id="lset-general" role="tabpanel" hidden=${stab !== "general"}>
          <div class="pane-h"><h2>${t("Configuration")}</h2></div>
          <div class="opts c3">
            <${Field} k="ctx" label="Context length" was=${was("ctx")} hint="Tokens, or empty (auto) for the model's full context, or the longest the GPU and host memory hold.">
              <input type="text" id="mf-ctx" inputmode="numeric" autocomplete="off" spellcheck="false" placeholder=${t("auto")}
                     value=${f.ctx} onInput=${set("ctx")}/><//>
            <${Field} k="kv" label="KV cache" was=${was("kv")} hint="fp8 holds twice the context.">
              <select id="mf-kv" value=${f.kv} onChange=${set("kv")}><option value="">${t("auto")}</option><option value="fp8">fp8</option><option value="16">${t("16-bit")}</option></select><//>
            <${Field} k="kv_ram" label="KV in host RAM" was=${was("kv_ram")} hint="GB, for context past the GPU.">
              <input type="number" id="mf-kvram" min="0" step="1" placeholder=${t("auto")} value=${f.kvram} onInput=${set("kvram")}/><//>
            <${Field} k="kv_gpu" label="KV on the GPU" was=${was("kv_gpu")} hint="Tokens, at most.">
              <input type="number" id="mf-kvgpu" min="0" step="1024" placeholder=${t("as many as fit")} value=${f.kvgpu} onInput=${set("kvgpu")}/><//>
            <${Field} k="name" label="API model id" was=${was("name")}>
              <input type="text" id="mf-name" placeholder=${t("the file's name")} spellcheck="false" autocomplete="off" value=${f.name} onInput=${set("name")}/><//>
          </div>
        </div>
        <div class="lset-panel" id="lset-spec" role="tabpanel" hidden=${stab !== "spec"}>
          <div class="pane-h"><h2>${t("Speculative decoding")}</h2>
            <span class=${"sd-was" + (was("spec") !== null ? " on" : "")}>${was("spec") !== null ? t("was " + t(was("spec"))) : ""}</span>
            <label class="switch sd-switch" title=${t("Speculative decoding")}>
              <input type="checkbox" id="sd-on" aria-label=${t("Speculative decoding")} checked=${f.spec} onChange=${set("spec")}/><i></i></label></div>
          <div class=${"sdopts" + (f.spec ? "" : " off")} data-k="drafter">
            ${savedDraft ? html`<div class=${"sdrow pick" + (f.draftSource === "saved" ? " sel" : "")} id="sd-saved" onClick=${() => pickDraft("saved")}>
              <input type="radio" name="sd-src" aria-label=${t("last loaded")} checked=${f.draftSource === "saved"} disabled=${specOff}/>
              <div class="grow"><div class="dhead"><b>${t("last loaded")}</b><span class="tag2">${savedDraft.drafter || t("default")}</span></div>
                ${savedDraft.drafters.map((file) => html`<span class="ddesc">${file}</span>`)}</div></div>` : ""}
            <div class=${"sdrow pick" + (f.draftSource === "default" ? " sel" : "")} id="sd-default" onClick=${() => pickDraft("default")}>
              <input type="radio" name="sd-src" aria-label=${t("Default")} checked=${f.draftSource === "default"} disabled=${specOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Default")}</b>${dflt.tag ? html`<span class=${"tag2 " + dflt.tone}>${t(dflt.tag)}</span>` : ""}</div>
                ${dflt.lines.map((l) => html`<em class="sd-what">${l}</em>`)}
              </div>
            </div>
            <div class=${"sdrow pick" + (custom ? " sel" : "")} id="sd-custom" onClick=${() => pickDraft("custom")}>
              <input type="radio" name="sd-src" aria-label=${t("Custom drafter")} checked=${custom} disabled=${specOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Custom drafter")}</b>
                  ${custom && own ? html`<span class=${"tag2 " + (ownCached ? "ok" : "")}>${t(ownCached ? "downloaded" : hubName(own) ? "Hugging Face" : "file")}</span>` : ""}
                  <span class="was on">${was("drafter") !== null ? t("was " + t(was("drafter"))) : ""}</span></div>
                <p class="ddesc">${t("A drafter of your own, in place of the default: a Hugging Face name or a path on the server's machine.")}</p>
                <input type="text" id="sd-file" placeholder=${t("org/name, or a path")} spellcheck="false" autocomplete="off" disabled=${specOff}
                       value=${f.drafter} onFocus=${() => !custom && pickDraft("custom")}
                       onInput=${(e) => setF(Object.assign({}, f, { draftSource: "custom", drafter: e.target.value }))}/>
                ${custom ? html`<em>${t(!own ? "Give a Hugging Face name or a path; empty, the default is used." : ownCached ? "On this machine." :
                                         hubName(own) ? "Fetched from Hugging Face when the model loads." : "A file on the server's machine.")}</em>` : ""}
              </div>
            </div>
          </div>
          ${specOff ? html`<em class="sd-note">${t("Plain decoding: one token a step.")}</em>` : ""}
          <details class="adv" id="sd-adv" open=${advOpen} onToggle=${(e) => { e.stopPropagation(); setAdvOpen(e.target.open); }}>
            <summary><${Icon} name="chev"/>${t("Advanced")}<span class="was" id="sd-adv-was">${advChanged.length ? t(advChanged.length + " changed") : ""}</span></summary>
            <div class="opts c3" style="margin-top: 12px">
              <${Field} k="width" label="Drafts a round" was=${was("width")} hint="How many drafts the model checks a step.">
                <select id="sd-width" value=${f.width} onChange=${set("width")} disabled=${specOff}>
                  <option value="">${t("Auto (chosen each round)")}</option>
                  ${["1", "2", "3", "4", "5", "6", "7", "8", "12", "16"].map((n) => html`<option value=${n}>${n}</option>`)}
                </select><//>
              <${Field} k="head" label="Draft head" was=${was("head")} hint="How drafts choose their tokens.">
                <select id="sd-head" value=${f.head} onChange=${set("head")} disabled=${specOff}>
                  <option value="">${t("1-bit pick (default)")}</option><option value="full">${t("Full head")}</option></select><//>
              <${Field} k="weights" label="Drafter weights" was=${was("weights")} hint="Only the drafts can change, never the output.">
                <select id="sd-weights" value=${f.weights} onChange=${set("weights")} disabled=${specOff}>
                  <option value="">${t("Compact copy (default)")}</option><option value="exact">${t("Exact")}</option></select><//>
              <${Field} k="lookup" label="Drafts from repeated text" was=${was("lookup")} hint="Where the reply repeats earlier text, that text is proposed too.">
                <label class="switch"><input type="checkbox" id="sd-lookup" checked=${f.lookup} onChange=${set("lookup")} disabled=${specOff}/><i></i>${t(f.lookup ? "on" : "off")}</label><//>
              <${Field} k="trees" label="Draft trees" was=${was("trees")} hint="The drafter proposes several branches at once, not one line.">
                <label class="switch"><input type="checkbox" id="sd-trees" checked=${f.trees} onChange=${set("trees")} disabled=${specOff}/><i></i>${t(f.trees ? "on" : "off")}</label><//>
            </div>
          </details>
        </div>
        <div class="lset-panel" id="lset-vision" role="tabpanel" hidden=${stab !== "vision"}>
          <div class="pane-h"><h2>${t("Vision encoder")}</h2>
            <span class=${"sd-was" + (was("vision") !== null ? " on" : "")}>${was("vision") !== null ? t("was " + t(was("vision"))) : ""}</span>
            <label class="switch sd-switch" title=${t("Vision encoder")}>
              <input type="checkbox" id="vs-on" aria-label=${t("Vision encoder")} checked=${f.vision} onChange=${set("vision")}/><i></i></label></div>
          <div class=${"sdopts" + (f.vision ? "" : " off")} data-k="mmproj">
            <div class=${"sdrow pick" + (!visCustom ? " sel" : "")} id="vs-default" onClick=${() => pickVision("default")}>
              <input type="radio" name="vs-src" aria-label=${t("Default")} checked=${!visCustom} disabled=${visOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Default")}</b>${vdflt.tag ? html`<span class=${"tag2 " + vdflt.tone}>${t(vdflt.tag)}</span>` : ""}</div>
                <em class="sd-what">${vdflt.line}</em>
              </div>
            </div>
            <div class=${"sdrow pick" + (visCustom ? " sel" : "")} id="vs-custom" onClick=${() => pickVision("custom")}>
              <input type="radio" name="vs-src" aria-label=${t("Custom projector")} checked=${visCustom} disabled=${visOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Custom projector")}</b>
                  ${visCustom && mmOwn ? html`<span class="tag2">${t("file")}</span>` : ""}
                  <span class="was on">${was("mmproj") !== null ? t("was " + t(was("mmproj"))) : ""}</span></div>
                <p class="ddesc">${t("Path to a companion mmproj-*.gguf file on the server's machine.")}</p>
                <input type="text" id="vs-file" placeholder=${t("path to mmproj-*.gguf")} spellcheck="false" autocomplete="off" disabled=${visOff}
                       value=${f.mmproj} onFocus=${() => !visCustom && pickVision("custom")}
                       onInput=${(e) => setF(Object.assign({}, f, { visionSource: "custom", mmproj: e.target.value }))}/>
              </div>
            </div>
          </div>
          ${visOff ? html`<em class="sd-note">${t("Vision disabled: text-only decoding.")}</em>` : ""}
          <div class=${"opts c3" + (visOff ? " off" : "")} style="margin-top: 14px">
            <${Field} k="vision_device" label="Vision storage" was=${was("vision_device")} hint="Disk keeps 0 VRAM and 0 locked RAM so KV capacity is untouched.">
              <select id="vs-device" value=${f.vision_device} onChange=${set("vision_device")} disabled=${visOff}>
                <option value="disk">${t("Disk (default · 0 VRAM, 0 RAM)")}</option>
                <option value="host">${t("Host RAM (0 VRAM)")}</option>
                <option value="gpu">${t("GPU (VRAM)")}</option>
                <option value="auto">${t("Auto (GPU if >= 8 GB free)")}</option>
              </select><//>
            <${Field} k="vision_quant" label="Vision precision" was=${was("vision_quant")} hint="Q8_0 matches BF16 quality (~46% smaller); pre-quantized files stay packed.">
              <select id="vs-quant" value=${f.vision_quant} onChange=${set("vision_quant")} disabled=${visOff}>
                <option value="q8">${t("Q8_0 (default)")}</option>
                <option value="exact">${t("Exact (file precision)")}</option>
                <option value="nvfp4">${t("NVFP4 (compact)")}</option>
              </select><//>
            <${Field} k="vision_max_tokens" label="Max tokens / image" was=${was("vision_max_tokens")} hint="Cap merged visual tokens per image (empty: auto).">
              <input type="number" id="vs-maxtok" min="16" step="1" placeholder=${t("auto")} disabled=${visOff}
                     value=${f.vision_max_tokens} onInput=${set("vision_max_tokens")}/><//>
          </div>
        </div>
        <div class="lset-cmd">
          <div class="pane-h"><h2>${t("Command line")}</h2><span class="aside"><${ShellPicker} shell=${shell} onChange=${setShell}/></span></div>
          <${CodeSnippet} id="mf-cmd" text=${commandOf(body(cmdModel), shell)}/>
        </div>
      </div>`;
  }

  // ---- the rows of what can be loaded, and the box that loads the one chosen ----
  // rows: {name, title, sub, tags: [[text, tone]], size, off (cannot be chosen), del (a cached model, to delete)}
  function rowsView(rows) {
    const serving = loadedWant ? loadedWant.model : null, remembered = L.remembered ? L.remembered.model : null;
    return rows.map((r) => html`
      <div class=${"mrow" + (r.name === target ? " sel" : "") + (r.off ? " off" : "")} data-name=${r.name} role="option" aria-selected=${String(r.name === target)}
           onClick=${r.off ? undefined : () => { setTarget(r.name); setMsg(""); }}>
        <span class="radio"></span>
        <div><div class="q">${r.title}</div>${r.sub ? html`<div class="fn">${r.sub}</div>` : ""}</div>
        <div class="tags">
          ${serving && r.name === serving ? html`<span class="tag2 live">${t("serving")}</span>`
            : remembered && r.name === remembered ? html`<span class="tag2">${t("last loaded")}</span>` : ""}
          ${r.tags.map(([text, tone]) => html`<span class=${"tag2 " + (tone || "")}>${t(text)}</span>`)}
        </div>
        <div class="sz">${r.size ? sized(r.size) : "-"}</div>
        <div class="fit">${r.off ? "" : html`<${FitTag} size=${r.size}/>`}</div>
        ${r.del ? html`<button class="iconbtn del" title=${t("Delete from the Hugging Face cache")} aria-label=${t("Delete")}
                               onClick=${(e) => { e.stopPropagation(); removeCached(r.del); }}><${Icon} name="trash"/></button>` : html`<span></span>`}
      </div>`);
  }
  const localRow = (m) => {
    const d = describe(m), tags = [];
    if (m.kind === "flyx") tags.push([".flyx"]);
    if (m.kind === "mmproj") tags.push(["mmproj", "info"]);
    if (m.mtp === "in-file") tags.push(["MTP inside", "ok"]); else if (m.mtp === "in-repo") tags.push(["+ MTP file", "ok"]);
    if (m.mmproj === "in-repo") tags.push(["+ Vision", "ok"]);
    if (m.current === false) tags.push(["older revision", "info"]);
    if (m.parts > 1) tags.push([m.parts + " parts"]);
    return { name: m.name, title: d.quant || d.file || m.name, sub: d.quant ? (d.file || m.name) : "", tags, size: m.size,
             del: m.source === "hub" ? m : null };
  };
  // `rows`: the ones shown, so a choice made elsewhere is not loaded from here.  `note`: what stands in the way.
  function loadBox(rows, note) {
    const row = rows.find((r) => r.name === target && !r.off), name = row ? row.name : "";
    const same = !!loadedWant && name === loadedWant.model, m = entryOf(name);
    let label = "Load model", disabled = !name || busy || !!note;
    if (busy) label = L.state === "fetching" ? "Fetching…" : "Loading…";
    else if (same) { label = changed.length ? "Reload with these settings" : "Loaded"; disabled = disabled || !changed.length; }
    else if (loadedWant && name) label = "Switch to this model";
    const what = !name ? "" : same ? "This is the model being served." : m || !hubName(name) ? sized(row.size) + " on disk" : sized(row.size) + " to download first";
    // A drafter or vision projector chosen for the model served may not match another.
    const strange = !!name && !same && use.spec && use.draftSource !== "default" && (use.draftSource === "saved" || !!use.drafter.trim());
    const strangeVis = !!name && !same && use.vision && use.visionSource === "custom" && !!use.mmproj.trim();
    return html`
      <div class="loadbox" id="mf-loadbox">
        <div class="what">
          ${name ? html`<div class="tn" id="mf-chosen" title=${name}>${name}</div>
                        <div class="ts" id="mf-chosen-s">${t(what)}<span class="with"> · ${brief.map(([k, v]) => t(k) + " " + t(v)).join(" · ")}${model.id ? " · " : ""}</span>
                          ${model.id ? html`<button class="linkbtn" id="mf-tosettings" onClick=${() => setView("current")}>${t("Change the settings")}</button>` : ""}</div>`
                 : html`<div class="ts" id="mf-chosen-s">${t(note || "Choose one of these to load.")}</div>`}
          ${strange ? html`<div class="ts warn" id="mf-strange">${t("Drafts with the drafter set for the model served.")}
            <button class="linkbtn" onClick=${() => setF(Object.assign({}, f, { draftSource: "default" }))}>${t("Use this model's default")}</button></div>` : ""}
          ${strangeVis ? html`<div class="ts warn" id="mf-strange-vis">${t("Uses the vision projector set for the model served.")}
            <button class="linkbtn" onClick=${() => setF(Object.assign({}, f, { visionSource: "default" }))}>${t("Use default projector")}</button></div>` : ""}
          ${msg ? html`<div class="ts bad" id="mf-msg">${t(msg)}</div>` : ""}
        </div>
        <button class="btn primary" id="mf-load" disabled=${disabled} onClick=${() => loadModel(name)}>${t(label)}</button>
      </div>`;
  }

  // ---- Hugging Face: the repositories found, the one shown ----
  function hubList() {
    const list = found ? found.models : [];
    return html`
      ${found && found.family ? html`<div class="hub-note" id="hf-family">${t("Showing " + found.query + " models, the family this build runs. Search for any other.")}</div>` : ""}
      ${foundMsg ? html`<div class="empty pad" id="hf-msg">${t(foundMsg)}</div>` : ""}
      ${found && !list.length && !foundMsg ? html`<div class="empty pad">${t("Nothing matches.")}</div>` : ""}
      ${list.map((m) => html`
        <button class=${"prow" + (m.id === repoSel ? " sel" : "")} data-repo=${m.id} onClick=${() => showRepo(m.id, "")}>
          <span class="pn">${m.id.split("/")[1]}</span>
          <span class="pm"><${Icon} name="heart"/>${compact(m.likes)}<${Icon} name="dl"/>${compact(m.downloads)}</span>
          <span class="ps">${m.id.split("/")[0]}${m.cached ? html` <span class="tag2 info">${t("downloaded")}</span>` : ""}${m.gated ? html` <span class="tag2">${t("gated")}</span>` : ""}</span>
          <span class="pt">${t(ago(m.updated || m.created))}</span>
        </button>`)}`;
  }
  function hubDetail() {
    if (!repoSel) return html`<div class="hub-scroll"><div class="empty">${t("Pick a repository on the left to see its files.")}</div></div>${loadBox([], "")}`;
    const j = hub && hub.repo === repoSel ? hub.j : null, info = (j && j.info) || {};
    const rows = j ? (j.ggufs || []).map((g) => {
      const tags = [];
      if (g.split && g.parts > 1) tags.push([g.parts + " parts"]);
      if (!whole(g)) tags.push(["split, incomplete"]);
      if (g.cached) tags.push(["downloaded", "info"]);
      const m = entryOf(g.name);
      return { name: g.name, title: g.quant, sub: "", tags, size: g.size, off: !whole(g), del: m && m.source === "hub" ? m : null };
    }) : [];
    const unsupported = j && j.runs === false;
    const note = unsupported ? "This build does not run this architecture." : j && !rows.length ? "Nothing here to load." : "";
    const chips = [];
    if (info.pipeline) chips.push([info.pipeline, ""]);
    if (info.architecture) chips.push([info.architecture, unsupported ? "bad" : j.runs ? "ok" : ""]);
    if (info.parameters) chips.push([compact(info.parameters) + " " + t("parameters"), ""]);
    if (info.context) chips.push([t("context") + " " + fmt(info.context), ""]);
    if (info.license) chips.push([info.license, ""]);
    if (info.gated) chips.push([t("gated"), "warn"]);
    const [org, short] = repoSel.split("/");
    return html`
      <div class="hub-scroll" id="hf-detail">
        <div class="dh">
          <div class="dh-icon"><${Icon} name="model"/></div>
          <div class="dh-t"><h3 id="hf-name">${short}</h3><span>${org}</span></div>
          <${CopyButton} get=${() => repoSel} label="Copy name"/>
          <a class="btn ghost sm" href=${((found && found.endpoint) || "https://huggingface.co") + "/" + repoSel} target="_blank" rel="noopener noreferrer">
            <${Icon} name="ext"/>${t("Open on Hugging Face")}</a>
        </div>
        ${chips.length ? html`<div class="dchips">${chips.map(([text, tone]) => html`<span class=${"tag2 " + tone}>${text}</span>`)}</div>` : ""}
        ${j && (info.updated || info.downloads || info.likes) ? html`
          <dl class="stats small">
            ${info.updated ? html`<div><dt>${t("Updated")}</dt><dd>${t(ago(info.updated))}</dd></div>` : ""}
            <div><dt>${t("Downloads")}</dt><dd>${compact(info.downloads)}</dd></div>
            <div><dt>${t("Likes")}</dt><dd>${compact(info.likes)}</dd></div>
            ${j.base_models && j.base_models.length ? html`<div><dt>${t("Base model")}</dt><dd>${j.base_models[0]}</dd></div>` : ""}
          </dl>` : ""}
        ${hubMsg ? html`<p class="lead" id="mf-hubmsg">${t(hubMsg)}</p>` : ""}
        ${unsupported ? html`<div class="banner warn" id="hf-unsupported">${t("These files are of the " + info.architecture + " architecture, which this build does not run. It runs: " +
                                                                       (j.build_architectures || []).join(", ") + ".")}</div>` : ""}
        ${rows.length ? html`
          <div class="dsec">${t("Files")}<span>${t(rows.length + " quantizations")}</span></div>
          <div class="mrows" id="mf-files" role="listbox">${rowsView(rows)}</div>` : ""}
        ${j && j.mtp ? html`<p class="lead dnote">${t("This repository keeps an MTP layer in " + j.mtp.path + " (" + gb(j.mtp.size) + " GB): it comes along for a file without its own.")}</p>` : ""}
        ${j && j.mmproj ? html`<p class="lead dnote" id="hf-mmproj">${t("This repository includes a vision projector in " + j.mmproj.path + " (" + gb(j.mmproj.size) + " GB): it loads automatically unless vision is turned off.")}</p>` : ""}
        ${j && j.drafter ? html`<div class="banner hint"><div class="grow2">${t(repoSel + " is a drafter (" + gb(j.drafter.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div>
          ${model.id ? html`<button class="btn sm" onClick=${() => draftWith(repoSel)}>${t("Draft with it")}</button>` : ""}</div>` : ""}
        ${j && !j.drafter && !rows.length ? html`<p class="lead">${t("No GGUF files in " + repoSel + ".")}</p>` : ""}
        ${j && (j.known_drafters || []).length ? html`
          <div class="banner hint" id="hf-drafters"><div class="grow2">${t(publishedNote[kindOf(j.known_drafters[0])] || publishedNote.dflash2)} ${t("It often adds speed on top of MTP:")}
            ${" "}<code>${j.known_drafters[0]}</code></div>
            ${model.id ? html`<button class="btn sm" onClick=${() => draftWith(j.known_drafters[0])}>${t("Add to settings")}</button>` : ""}</div>` : ""}
      </div>
      ${loadBox(rows, note)}`;
  }

  // ---- this machine: folders in a list, the models of the one shown ----
  const local = catalog ? catalog.models : [];
  const repos = new Map(), known = new Map();  // repository -> its models; folder -> the server's models in it
  for (const m of local) {
    const into = m.source === "hub" ? repos : known, key = m.source === "hub" ? m.repo || repoOf(m.name) : folderOf(m);
    if (!into.has(key)) into.set(key, []);
    into.get(key).push(m);
  }
  // The folders: the one Hugging Face downloads to, those with the server's other models, those of the model served
  // and the one remembered, those added here.
  const cacheDir = catalog ? catalog.cache : "", atCache = !!cacheDir && !!loc && samePath(loc, cacheDir);
  const places = cacheDir ? [cacheDir] : [];
  const wanted = [loadedWant, L.remembered].filter((w) => w && w.model && !hubName(w.model) && /[\\/]/.test(w.model)).map((w) => folderOf({ name: w.model }));
  for (const x of [...known.keys(), ...wanted, ...folders]) if (!places.some((k) => samePath(k, x))) places.push(x);
  const sum = (list) => list.reduce((n, m) => n + (m.size || 0), 0);
  const ql = localQ.trim().toLowerCase(), hit = (text) => !ql || text.toLowerCase().includes(ql);
  // A folder's models: the server's own list for the download folder and its model directories, else what it found
  // there when asked.
  const modelsOf = (p) => samePath(p, cacheDir) ? local.filter((m) => m.source === "hub") : known.get(p) || (listings[p] && listings[p].models) || [];
  useEffect(() => {
    if (!active || view !== "library" || tab !== "local" || !catalog) return;
    for (const folder of places) {
      if (samePath(folder, cacheDir) || known.has(folder) || listings[folder]) continue;
      setListings((old) => Object.assign({}, old, { [folder]: { models: [], looking: true } }));
      ask("/v1/hub/dir?path=" + encodeURIComponent(folder), "browse folders").then((a) =>
        setListings((old) => Object.assign({}, old, { [folder]: a.error ? { models: [], error: a.error } : { models: a.j.models } })));
    }
  }, [active, view, tab, catalog, places.join("\n")]);
  const kindOfModel = (m) => m.kind === "drafter" ? "drafter" : m.kind === "mmproj" ? "mmproj" : m.kind === "flyx" ? ".flyx" : "GGUF";
  // A model as the list names it: a repository's by its name and quantization, a file by its own name.
  const titleOf = (m) => {
    const d = describe(m);
    return m.source === "hub" && d.repo ? d.repo.split("/")[1] + (d.quant ? " \u00b7 " + d.quant : "") : d.file || baseName(m.name);
  };
  const pickModel = (m) => { setPicked(m.name); setTarget(m.kind === "drafter" || m.kind === "mmproj" ? "" : m.name); setMsg(""); };
  function localList() {
    if (catalogErr) return html`<div class="empty pad">${catalogErr}</div>`;
    if (!catalog) return html`<div class="empty pad">${t("Looking...")}</div>`;
    const act = (go) => (e) => { e.stopPropagation(); go(); };
    const onEnter = (go) => (e) => { if (e.key === "Enter" && e.target === e.currentTarget) go(); };
    const serving = loadedWant ? loadedWant.model : null;
    return html`
      ${places.map((p) => {
        const all = modelsOf(p), state = listings[p] || {};
        const list = all.filter((m) => hit(titleOf(m) + " " + m.name)).sort((x, y) => titleOf(x).localeCompare(titleOf(y)));
        const mine = !samePath(p, cacheDir) && folders.some((x) => samePath(x, p));
        const open = () => { setPicked(""); openPath(p); };
        return html`
          <div class=${"prow act folder" + (!picked && samePath(loc, p) ? " sel" : "")} role="button" tabindex="0" data-place=${"dir:" + p}
               onClick=${open} onKeyDown=${onEnter(open)}>
            <span class="pn path"><${Icon} name="folder"/>${baseName(p.replace(/[\\/]+$/, "")) || p}</span>
            <span class="pm">${all.length ? sized(sum(all)) : ""}</span>
            <span class="ps path">${p}</span>
            <span class="pt">${all.length ? t(all.length + (all.length === 1 ? " file" : " files")) : ""}</span>
            ${mine ? html`<button class="iconbtn del" title=${t("Remove from the list")} aria-label=${t("Remove from the list")}
                                  onClick=${act(() => setFolders((old) => old.filter((x) => !samePath(x, p))))}><${Icon} name="close"/></button>`
                   : html`<span class="del"></span>`}
          </div>
          ${list.map((m) => {
            const d = describe(m), hubModel = m.source === "hub";
            const under = hubModel ? (d.repo || "").split("/")[0] : folderOf(m).length > p.length ? folderOf(m).slice(p.length).replace(/^[\\/]+/, "") : "";
            const go = () => pickModel(m);
            return html`
              <div class=${"prow act model" + (picked === m.name ? " sel" : "")} role="button" tabindex="0" data-model=${m.name} onClick=${go} onKeyDown=${onEnter(go)}>
                <span class="pn">${titleOf(m)}</span>
                <span class="pm">${m.size ? sized(m.size) : ""}</span>
                <span class="ps">${under}${serving && m.name === serving ? html` <span class="tag2 live">${t("serving")}</span>` : ""}</span>
                <span class="pt">${t(kindOfModel(m))}</span>
                ${hubModel ? html`<button class="iconbtn del" title=${t("Delete from the Hugging Face cache")} aria-label=${t("Delete")}
                                          onClick=${act(() => removeCached(m))}><${Icon} name="trash"/></button>` : html`<span class="del"></span>`}
              </div>`;
          })}
          ${list.length ? "" : html`<div class="pnone">${t(state.looking ? "Looking..." : state.error ? state.error : all.length ? "Nothing matches." : "No model files here.")}</div>`}`;
      })}
      <button class="prow add" id="mf-add" disabled=${picking} onClick=${addFolder}>
        <span class="pn path"><${Icon} name="plus"/>${t(picking ? "Choose the folder in the window that opened\u2026" : "Add a folder\u2026")}</span></button>`;
  }
  // One model of the list: what it is, where its file is, and the box that loads it.
  function modelDetail(m) {
    const d = describe(m), hubModel = m.source === "hub", drafter = m.kind === "drafter", mmproj = m.kind === "mmproj";
    const chips = [[t(kindOfModel(m)), ""]];
    if (d.quant) chips.push([d.quant, ""]);
    if (m.mtp === "in-file") chips.push([t("MTP inside"), "ok"]); else if (m.mtp === "in-repo") chips.push([t("+ MTP file"), "ok"]);
    if (m.mmproj === "in-repo") chips.push([t("+ Vision"), "ok"]);
    if (m.parts > 1) chips.push([t(m.parts + " parts"), ""]);
    if (m.current === false) chips.push([t("older revision"), "info"]);
    if (loadedWant && m.name === loadedWant.model) chips.push([t("serving"), "live"]);
    return html`
      <div class="hub-scroll" id="lc-detail">
        <div class="dh">
          <div class="dh-icon"><${Icon} name=${mmproj ? "image" : "model"}/></div>
          <div class="dh-t"><h3 id="lc-name">${titleOf(m)}</h3><span>${hubModel ? (d.repo || "").split("/")[0] : folderOf(m)}</span></div>
          ${hubModel && !drafter && !mmproj ? html`<button class="btn ghost sm" id="lc-quants" onClick=${() => { setTab("hub"); showRepo(d.repo, ""); }}>${t("All its quantizations")}</button>` : ""}
          ${hubModel ? html`<button class="btn ghost sm" id="lc-delete" onClick=${() => removeCached(m)}><${Icon} name="trash"/>${t("Delete")}</button>` : ""}
        </div>
        <div class="dchips">${chips.map(([text, tone]) => html`<span class=${"tag2 " + tone}>${text}</span>`)}${drafter || mmproj ? "" : html`<${FitTag} size=${m.size}/>`}</div>
        <dl class="stats small">
          <div><dt>${t("Size")}</dt><dd>${m.size ? sized(m.size) : "-"}</dd></div>
          ${m.modified ? html`<div><dt>${t("Modified")}</dt><dd>${t(ago(m.modified))}</dd></div>` : ""}
          ${hubModel && d.repo ? html`<div><dt>${t("Repository")}</dt><dd>${d.repo}</dd></div>` : ""}
          ${m.revision ? html`<div><dt>${t("Revision")}</dt><dd>${String(m.revision).slice(0, 8)}</dd></div>` : ""}
        </dl>
        <div class="dsec">${t(m.kind === "drafter" ? "Folder" : "File")}</div>
        <${CodeSnippet} id="lc-path" text=${m.path || m.name}/>
        ${drafter ? html`<div class="banner hint"><div class="grow2">${t(m.name + " is a drafter (" + gb(m.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div>
          ${model.id ? html`<button class="btn sm" onClick=${() => draftWith(m.name)}>${t("Draft with it")}</button>` : ""}</div>` : ""}
        ${mmproj ? html`<div class="banner hint"><div class="grow2">${t(m.file + " is a vision projector (" + gb(m.size) + " GB), not a standalone model: attach it to a compatible VLM.")}</div>
          ${model.id ? html`<button class="btn sm" id="lc-use-mmproj" onClick=${() => projectWith(m.path || m.name)}>${t("Use as projector")}</button>` : ""}</div>` : ""}
      </div>
      ${loadBox(drafter || mmproj ? [] : [localRow(m)], drafter ? "A drafter is not loaded on its own." : mmproj ? "A vision projector is not loaded on its own." : "")}`;
  }
  function localDetail() {
    const j = atCache ? null : dir, start = !!j && j.kind === "start";
    const go = () => openPath(pathText.trim());
    const above = atCache ? loc.replace(/[\\/]+$/, "").replace(/[\\/][^\\/]*$/, "") : j ? j.parent : null;
    const bar = html`
      <div class="pathbar">
        <button class="iconbtn" id="mf-up" title=${t("The folder above")} aria-label=${t("The folder above")} disabled=${!above}
                onClick=${() => openPath(above)}><${Icon} name="up"/></button>
        <input type="text" id="mf-path" placeholder=${t("A folder or a model file on the server's machine")} spellcheck="false" autocomplete="off"
               aria-label=${t("A folder or a model file on the server's machine")} value=${pathText} onInput=${(e) => setPathText(e.target.value)}
               onKeyDown=${(e) => { if (e.key === "Enter") { e.preventDefault(); go(); } }}/>
        <button class="btn" id="mf-open" onClick=${go}>${t("Open")}</button>
        <button class="btn" id="mf-keep" hidden=${!j || j.kind !== "dir" || places.some((x) => samePath(x, loc)) && !kept} aria-pressed=${String(kept)} onClick=${keep}>
          ${t(kept ? "Remove from the list" : "Keep in the list")}</button>
      </div>`;
    if (atCache) {
      // What Hugging Face downloaded, by repository: the files of each to load, the whole of it to delete.
      const groups = [...repos].filter(([name, list]) => hit(name) || list.some((m) => hit(m.name + " " + (m.file || ""))));
      const rows = [];
      const body = groups.map(([name, all]) => {
        const list = all.slice().sort((x, y) => (x.size || 0) - (y.size || 0));
        const mine = list.filter((m) => m.kind !== "drafter" && m.kind !== "mmproj").map(localRow), drafter = list.find((m) => m.kind === "drafter");
        rows.push(...mine);
        return html`
          <div class="rgroup" data-repo=${name}>
            <div class="rgroup-h"><b>${name}</b><span>${sized(sum(list))}</span><span class="grow"></span>
              ${mine.length ? html`<button class="btn ghost sm" onClick=${() => { setTab("hub"); showRepo(name, ""); }}>${t("All its quantizations")}</button>` : ""}
              <button class="iconbtn del" title=${t("Delete from the Hugging Face cache")} aria-label=${t("Delete")}
                      onClick=${() => removeRepo(name, all)}><${Icon} name="trash"/></button></div>
            ${mine.length ? html`<div class="mrows" role="listbox">${rowsView(mine)}</div>` : ""}
            ${drafter ? html`<div class="banner hint"><div class="grow2">${t(name + " is a drafter (" + gb(drafter.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div>
              ${model.id ? html`<button class="btn sm" onClick=${() => draftWith(drafter.name)}>${t("Draft with it")}</button>` : ""}</div>` : ""}
          </div>`;
      });
      return html`
        <div class="hub-scroll" id="lc-detail">
          ${bar}
          <p class="lead dnote" id="mf-cachenote">${t("What is downloaded from Hugging Face is kept here, a repository each.")}</p>
          <div id="mf-cached">${groups.length ? body : html`<div class="empty" style="margin-top: 16px">${t(repos.size ? "Nothing matches." : "Nothing downloaded yet.")}</div>`}</div>
        </div>
        ${loadBox(rows, rows.length ? "" : "Nothing here to load.")}`;
    }
    // Any other folder: the folders in it, the model files in it and under it.
    const allModels = j ? j.models.filter((m) => m.kind !== "mmproj") : [];
    const rows = allModels.map((m) => {
      const r = localRow(m), rel = j.kind === "dir" && m.path.length > j.path.length ? m.path.slice(j.path.length).replace(/^[\\/]+/, "") : m.file;
      return Object.assign(r, { sub: r.title === rel ? "" : rel });
    }).filter((r) => hit(r.title + " " + r.sub));
    const folderMmproj = j ? j.models.filter((m) => m.kind === "mmproj" && hit(m.file + " " + (m.path || ""))) : [];
    const loadNote = start ? "Open a folder to see its model files."
      : j && j.kind === "file" && folderMmproj.length ? "A vision projector is not loaded on its own." : "";
    return html`
      <div class="hub-scroll" id="lc-detail">
        ${bar}
        ${dirMsg ? html`<p class="lead" id="mf-dirmsg">${t(dirMsg)}</p>` : ""}
        ${j && j.dirs.length ? html`
          <div class="dsec">${t(start ? "Start from" : "Folders")}</div>
          <div class="fchips" id="mf-dirs">${j.dirs.map((d) => html`
            <button class="chip" title=${d.path} onClick=${() => openPath(d.path)}><${Icon} name="folder"/>${d.name}</button>`)}</div>` : ""}
        ${j && !start ? html`
          <div class="dsec">${t("Model files")}<span>${t(j.kind === "dir" ? "here and two folders down" : "")}</span></div>
          ${rows.length ? html`<div class="mrows" id="mf-cached" role="listbox">${rowsView(rows)}</div>`
            : folderMmproj.length && j.kind === "file" ? ""
            : html`<div class="empty">${t(allModels.length ? "Nothing matches." : "No .gguf or .flyx model files here.")}</div>`}
          ${folderMmproj.map((mm) => html`<div class="banner hint" style="margin-top: 10px"><div class="grow2">${t(mm.file + " is a vision projector (" + gb(mm.size) + " GB), not a standalone model: attach it to a compatible VLM.")}</div>
            ${model.id ? html`<button class="btn sm" onClick=${() => projectWith(mm.path || mm.name)}>${t("Use as projector")}</button>` : ""}</div>`)}` : ""}
      </div>
      ${loadBox(rows, loadNote)}`;
  }

  const hubTab = tab === "hub";
  const pickedModel = picked ? places.flatMap(modelsOf).find((m) => m.name === picked) : null;
  const shownRows = hubTab ? (hub && hub.repo === repoSel ? (hub.j.ggufs || []).filter(whole).map((g) => g.name) : [])
                  : pickedModel ? [pickedModel.name]
                  : atCache ? local.filter((m) => m.source === "hub" && m.kind !== "drafter" && m.kind !== "mmproj").map((m) => m.name)
                  : dir ? dir.models.filter((m) => m.kind !== "mmproj").map((m) => m.name) : [];
  const cmdModel = shownRows.includes(target) ? target : loadedWant ? loadedWant.model : L.remembered ? L.remembered.model : "";
  const again = L.remembered ? { model: L.remembered.model, load: () => loadModel(L.remembered.model, L.remembered) } : null;
  const retryWant = L.target
    ? (lastAttempt.current && lastAttempt.current.model === L.target ? lastAttempt.current
       : L.remembered && L.remembered.model === L.target ? L.remembered
       : { model: L.target })
    : L.remembered || null;
  const retry = retryWant ? () => loadModel(retryWant.model, retryWant) : null;
  const current = view === "current";
  const stateTone = L.state === "ready" ? "ok" : L.state === "failed" ? "bad" : busy ? "warn" : "";
  const stateWord = L.state === "fetching" ? t("Fetching") + (L.progress && L.progress.total ? " " + fmt(100 * L.progress.done / L.progress.total) + "%" : "") :
                    L.state === "loading" ? t("Loading") : L.state === "failed" ? t("Failed") : model.id ? model.id : "";
  return html`
    <div class="tabs mtabs" role="tablist" id="mv-tabs">
      <button role="tab" data-view="current" aria-selected=${String(current)} onClick=${() => setView("current")}>
        ${t("Current model")}${stateWord ? html`<span class=${"badge " + stateTone} id="mv-state"><span class="dot"></span>${stateWord}</span>` : ""}</button>
      <button role="tab" data-view="library" aria-selected=${String(!current)} onClick=${() => setView("library")}>${t("Model library")}</button>
    </div>
    <div class="mview" id="mv-current" hidden=${!current}>
      <div class="sec"><div class="pane statbar"><${Serving} tm=${tm} again=${again} retry=${retry} browse=${() => setView("library")}/></div></div>
      ${model.id ? html`<div class="sec">${settingsView(cmdModel)}</div>` : ""}
    </div>
    <div class="sec hubsec" id="mv-library" hidden=${current}>
      <div class="pane hub">
        <div class="hub-h">
          <div class="seg" role="tablist" id="src-seg">
            <button role="tab" data-src="hub" aria-selected=${String(hubTab)} onClick=${() => setTab("hub")}><${Icon} name="hf"/>Hugging Face</button>
            <button role="tab" data-src="local" aria-selected=${String(!hubTab)} onClick=${() => setTab("local")}>
              <${Icon} name="machine"/>${t("On this machine")}${local.length ? html`<span class="n" id="mf-count">${local.filter((m) => m.kind !== "drafter" && m.kind !== "mmproj").length}</span>` : ""}</button>
          </div>
          <label class="search"><${Icon} name="search"/>
            ${hubTab ? html`<input type="text" id="hf-q" placeholder=${t("Search Hugging Face, or give a repository: org/name")} spellcheck="false" autocomplete="off"
                                   value=${q} onInput=${(e) => setQ(e.target.value)}
                                   onKeyDown=${(e) => { if (e.key === "Enter") { e.preventDefault(); if (!named(q)) search(q, sort); } }}/>`
                     : html`<input type="text" id="mf-q" placeholder=${t("Filter by name or quantization")} spellcheck="false" autocomplete="off"
                                   value=${localQ} onInput=${(e) => setLocalQ(e.target.value)}/>`}</label>
          ${hubTab ? html`<select id="hf-sort" aria-label=${t("Order")} value=${sort} onChange=${(e) => setSort(e.target.value)}>
            ${SORTS.map(([k, name]) => html`<option value=${k}>${t(name)}</option>`)}</select>` : ""}
        </div>
        <div class="hub-b">
          <div class="hub-list" id=${hubTab ? "hf-list" : "lc-list"}>${hubTab ? hubList() : localList()}</div>
          <div class="hub-detail">${hubTab ? hubDetail() : pickedModel ? modelDetail(pickedModel) : localDetail()}</div>
        </div>
      </div>
    </div>`;
}
