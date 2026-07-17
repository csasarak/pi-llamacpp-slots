/**
 * Llama.cpp Slots Extension
 *
 * Automatically saves llama.cpp slot KV cache to disk at the end of each
 * agent turn and restores it when resuming sessions. Derives the server
 * URL from ctx.model.baseUrl and probes GET /slots to detect capability.
 *
 * Features:
 * - Per-turn slot save (fire-and-forget, non-blocking)
 * - Slot restore on session resume
 * - id_slot injection for deterministic slot routing
 * - KV cache erase on quit (configurable, off by default)
 *
 * Usage:
 * 1. Copy this file to ~/.pi/agent/extensions/llamacpp-slots.ts
 * 2. Ensure your llama.cpp server is started with --slot-save-path /path/to/dir
 *
 * Settings (~/.pi/agent/llama-slots/settings.json):
 * {
 *   "eraseOnQuit": false,        // Erase in-memory KV cache on quit (default: false)
 *   "serverUrl": "http://localhost:4000"  // Override llama.cpp server URL (default: derived from ctx.model.baseUrl)
 * }
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { deriveBinFilename, buildSlotRequest, type SlotState, getModelReloadPending, setModelReloadPending, discoverSlots, getSlotInfo, restoreSlot, eraseSlot } from "./slot-logic";

// ── Settings Interface ───────────────────────────────────────

interface SlotSettings {
	/** Erase in-memory KV cache on session quit. Default: false. */
	eraseOnQuit?: boolean;
	/** Explicit llama.cpp server URL override. When set, bypasses ctx.model.baseUrl derivation. */
	serverUrl?: string;
	/** Explicit model name override for router mode. When set, used instead of ctx.model.id. */
	modelName?: string;
	/** Save slot on agent_end (once per agent loop) instead of turn_end (per tool call). Default: true. */
	saveOnAgentEnd?: boolean;
	/** Save slot on session shutdown (quit). Default: false — per-turn/agent saves are usually sufficient. */
	saveOnShutdown?: boolean;
}

// ── In-Memory State ──────────────────────────────────────────

let slotState: SlotState | null = null;
let slotsActive = false;
let saveController: AbortController | null = null;
let restoringPromise: Promise<boolean> | null = null;  // In-flight restore promise for race prevention
let firstTurn = true;  // Track if this is the first turn of the session
let isNewSession = false;  // True when no persisted slot state was found (brand-new session)
let isRouterMode = false;  // True when server requires model param (router mode)
  // Heuristic: GET /slots without model fails → assume router. Could be server down, but fallback is graceful.

// ── Constants ────────────────────────────────────────────────

const SAVE_TIMEOUT_MS = 3000;
const CUSTOM_ENTRY_TYPE = "llamacpp-slot";
const SETTINGS_DIR = path.join(os.homedir(), ".pi", "agent", "llama-slots");
const SETTINGS_FILE = path.join(SETTINGS_DIR, "settings.json");
const LOG_FILE = path.join(SETTINGS_DIR, "debug.log");

// ── Logging ──────────────────────────────────────────────────

/**
 * Log a message to the debug.log file only.
 * The log file is append-only and lives at ~/.pi/agent/llama-slots/debug.log.
 */
function log(message: string): void {
	const timestamp = new Date().toISOString();
	const line = `[${timestamp}] ${message}\n`;
	try {
		fs.mkdirSync(SETTINGS_DIR, { recursive: true });
		fs.appendFileSync(LOG_FILE, line);
	} catch {
		// Non-critical — silent failure
	}
}

// ── Settings Helpers ─────────────────────────────────────────

/**
 * Load settings from ~/.pi/agent/llama-slots/settings.json.
 * Returns defaults if the file doesn't exist or is invalid.
 */
function loadSettings(): SlotSettings {
	const defaults: SlotSettings = { eraseOnQuit: false, saveOnAgentEnd: true, saveOnShutdown: false };
	try {
		const raw = fs.readFileSync(SETTINGS_FILE, "utf-8");
		const parsed = JSON.parse(raw) as SlotSettings;
		return {
			eraseOnQuit: parsed.eraseOnQuit ?? defaults.eraseOnQuit,
			serverUrl: parsed.serverUrl,
			saveOnAgentEnd: parsed.saveOnAgentEnd ?? defaults.saveOnAgentEnd,
			saveOnShutdown: parsed.saveOnShutdown ?? defaults.saveOnShutdown,
		};
	} catch {
		return defaults;
	}
}

/**
 * Save settings to ~/.pi/agent/llama-slots/settings.json.
 * Creates the directory if it doesn't exist.
 */
function saveSettings(settings: SlotSettings): void {
	try {
		fs.mkdirSync(SETTINGS_DIR, { recursive: true });
		fs.writeFileSync(SETTINGS_FILE, JSON.stringify(settings, null, 2) + "\n");
	} catch (err) {
		log(`[llamacpp-slots] Failed to save settings: ${(err as Error).message}`);
	}
}

// ── Slot API Helpers ─────────────────────────────────────────

/**
 * Discover available slots by probing GET /slots.
 * Returns the first available slot ID, or the first slot ID if none available.
 * Returns null if the server doesn't support slot management.
 */
// ── Save Slot ────────────────────────────────────────────────

/**
 * Save the current slot's KV cache to a .bin file.
 * Fire-and-forget — does not await. Uses a dedicated AbortController
 * with timeout (NOT ctx.signal) to avoid cancellation on user Escape.
 */
function saveSlot(state: SlotState): void {
	// Abort any previous in-flight save to avoid duplicates
	saveController?.abort();

	const controller = new AbortController();
	saveController = controller;
	setTimeout(() => controller.abort(), SAVE_TIMEOUT_MS);

	const { modelParam, body } = buildSlotRequest(state, { filename: state.binFilename });

	fetch(`${state.serverUrl}/slots/${state.slotId}?action=save${modelParam}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
		signal: controller.signal,
	})
		.then(async (res) => {
			if (!res.ok) {
				log(`[llamacpp-slots] Save failed: HTTP ${res.status}`);
			}
		})
		.catch((err) => {
			if (err.name !== "AbortError") {
				log(`[llamacpp-slots] Save error: ${err.message}`);
			}
		});
}

// ── Persistence ──────────────────────────────────────────────

/**
 * Persist current slot state to session entries.
 * Follows plan-mode/index.ts:108-113 pattern.
 */
function persistSlotState(pi: ExtensionAPI): void {
	if (!slotState) return;
	pi.appendEntry<SlotState>(CUSTOM_ENTRY_TYPE, { ...slotState });
}

/**
 * Restore slot state from session branch entries.
 * Follows tools.ts:52-68 pattern — last entry wins.
 */
function restoreFromBranch(ctx: ExtensionContext): SlotState | null {
	const branchEntries = ctx.sessionManager.getBranch();
	let restored: SlotState | null = null;

	for (const entry of branchEntries) {
		if (entry.type === "custom" && entry.customType === CUSTOM_ENTRY_TYPE) {
			const data = entry.data as SlotState | undefined;
			if (data?.slotId != null && data.sessionFile && data.binFilename && data.serverUrl) {
				restored = data;
			}
		}
	}

	return restored;
}

// ── Extension Factory ────────────────────────────────────────

// ── Slash Command: /llama-slots ─────────────────────────────

/** Toggle items for the settings selector */
/**
 * Allocate a slot by discovering available slots with model param.
 * Sets modelReloadPending so the first slot call triggers model load.
 */
async function tryActivateRouterSlot(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	if (slotsActive) return;

	const settings = loadSettings();
	const serverUrl = settings.serverUrl?.replace(/\/+$/, "") ?? ctx.model?.baseUrl?.replace(/\/+$/, "");
	const modelName = settings.modelName ?? ctx.model?.id;
	if (!serverUrl || !modelName) return;

	const slotId = await discoverSlots(serverUrl, modelName);
	if (slotId == null) return;

	const sessionId = ctx.sessionManager.getSessionId();
	const sessionFile = ctx.sessionManager.getSessionFile();
	if (!sessionId || !sessionFile) return;

	slotState = {
		slotId,
		sessionFile,
		binFilename: deriveBinFilename(sessionId, modelName),
		serverUrl,
		modelName,
	};
	slotsActive = true;
	firstTurn = true;
	isNewSession = true;
	setModelReloadPending(true);
	persistSlotState(pi);
	log(`[llamacpp-slots] Router mode: allocated slot ${slotId}`);
}

const TOGGLE_OPTIONS = [
	{ key: "eraseOnQuit" as const, label: "eraseOnQuit", description: "Erase in-memory KV cache on quit" },
	{ key: "saveOnAgentEnd" as const, label: "saveOnAgentEnd", description: "Save once per agent loop instead of per tool call" },
	{ key: "saveOnShutdown" as const, label: "saveOnShutdown", description: "Save slot on session shutdown" },
] as const;

export default function llamacppSlotsExtension(pi: ExtensionAPI): void {
	log("[llamacpp-slots] Extension loaded!");

	// ── Command: /llama-slots ────────────────────────────────

	pi.registerCommand("llama-slots", {
		description: "Configure llama.cpp slots extension settings",
		getArgumentCompletions: (prefix) => {
			const keys = TOGGLE_OPTIONS.map((o) => o.key);
			const filtered = keys.filter((k) => k.startsWith(prefix));
			return filtered.length > 0 ? filtered.map((k) => ({ value: k, label: k })) : null;
		},
		handler: async (args, ctx) => {
			const settings = loadSettings();

			// Direct toggle: /llama-slots saveOnShutdown
			const arg = args?.trim();
			if (arg) {
				const option = TOGGLE_OPTIONS.find((o) => o.key === arg);
				if (!option) {
					log(`[llamacpp-slots] Unknown option "${arg}". Available: ${TOGGLE_OPTIONS.map((o) => o.key).join(", ")}`);
					return;
				}
				const current = settings[option.key] ?? false;
				settings[option.key] = !current;
				saveSettings(settings);
				log(`[llamacpp-slots] ${option.key} toggled to ${!current}`);
				return;
			}

			// Show settings menu
			const items = TOGGLE_OPTIONS.map((opt) => {
				const value = settings[opt.key] ?? false;
				return `${opt.key}: ${value}  (${opt.description})`;
			});
			items.push("---");
			items.push("View settings file");

			const selected = await ctx.ui.select("llama.cpp Slots Settings", items);
			if (!selected) return;

			if (selected === "View settings file") {
				// Read current file contents (or show empty object if not created yet)
				let currentContent = "{}";
				try {
					currentContent = fs.readFileSync(SETTINGS_FILE, "utf-8");
				} catch {
					// File doesn't exist yet — show formatted empty object
					currentContent = JSON.stringify({}, null, 2) + "\n";
				}
				const edited = await ctx.ui.editor(
					`Edit settings (${SETTINGS_FILE})`,
					currentContent,
				);
				if (edited !== undefined) {
					// Validate JSON before saving
					try {
						JSON.parse(edited);
						fs.mkdirSync(SETTINGS_DIR, { recursive: true });
						fs.writeFileSync(SETTINGS_FILE, edited.endsWith("\n") ? edited : edited + "\n");
						log(`[llamacpp-slots] Settings file saved`);
					} catch (err) {
						log(`[llamacpp-slots] Settings save failed: invalid JSON`);
					}
				}
				return;
			}

			// Parse the toggled key from the selected line
			const key = selected.split(":")[0].trim() as "eraseOnQuit" | "saveOnAgentEnd" | "saveOnShutdown";
			const option = TOGGLE_OPTIONS.find((o) => o.key === key);
			if (!option) return;

			const current = settings[key] ?? false;
			settings[key] = !current;
			saveSettings(settings);
			log(`[llamacpp-slots] ${key} toggled to ${!current}`);
		},
	});

	// ── Session Start: Discover + Register Slot ───────────────

	pi.on("session_start", async (_event, ctx) => {
		log(`[llamacpp-slots] session_start fired, reason=${_event.reason}`);
		isRouterMode = false;  // Reset each session
		// Determine server URL: settings override wins, then fall back to ctx.model.baseUrl
		const settings = loadSettings();
		log(`[llamacpp-slots] settings.serverUrl=${settings.serverUrl}, ctx.model.baseUrl=${ctx.model?.baseUrl}`);
		let serverUrl: string | undefined;
		let modelName: string | undefined;

		// ctx.ui.setStatus("llamacpp-slots", "checking...");

		if (settings.serverUrl) {
			serverUrl = settings.serverUrl.replace(/\/+$/, "");
		}
		if (settings.modelName) {
			modelName = settings.modelName;
		} else if (ctx.model?.id) {
			modelName = ctx.model.id;
		}
		if (ctx.model?.baseUrl && !serverUrl) {
			serverUrl = ctx.model.baseUrl.replace(/\/+$/, "");
		}
		if (!serverUrl) {
			log("[llamacpp-slots] No server URL — staying dormant");
			// ctx.ui.setStatus("llamacpp-slots", "no server URL");
			return;
		}
		log(`[llamacpp-slots] session_start: serverUrl=${serverUrl}, modelName=${modelName ?? "<none>"}`);

		// Step 1: Try to restore persisted slot state from branch
		const restored = restoreFromBranch(ctx);

		if (restored) {
			// We have a saved slot state — register it. Actual restore happens
			// in turn_start (more reliable: llama.cpp is definitely ready by then).
			slotState = restored;
			// Always use the current model's URL (in case server was reconfigured)
			slotState.serverUrl = serverUrl;
			slotsActive = true;
			firstTurn = true;

			log(`[llamacpp-slots] Restored slot state from branch: slot=${restored.slotId}, bin=${restored.binFilename}`);
			isNewSession = false;

			// Refresh modelName from current model (in case it changed)
			if (modelName !== undefined) {
				// Model changed since slot was allocated — discover new slot
				if (modelName !== restored.modelName) {
					const newSlotId = await discoverSlots(serverUrl, modelName);
					if (newSlotId != null) {
						slotState.slotId = newSlotId;
						slotState.binFilename = deriveBinFilename(ctx.sessionManager.getSessionId() ?? restored.binFilename, modelName);
						log(`[llamacpp-slots] Model changed (${restored.modelName} -> ${modelName}), new slot ${newSlotId}`);
					}
				}
				slotState.modelName = modelName;
			}

			// Skip probe — sends model param which triggers reload. turn_start checks cold anyway.
			return;
		}

		log("[llamacpp-slots] No persisted slot state in branch — discovering fresh");

		// Step 2: Try discovery without model param first (avoids reload trigger)
		const slotId = await discoverSlots(serverUrl, undefined);
		if (slotId == null) {
			// GET /slots failed — assume router mode (requires model param) and defer.
			// Don't retry with model — that triggers reload. If server is actually down,
			// before_provider_request will fail gracefully later.
			isRouterMode = true;
			slotsActive = false;
			log(`[llamacpp-slots] GET /slots failed — assuming router mode, deferring slot allocation`);
			return;
		}

		// Step 3: Allocate slot for this session
		const sessionFile = ctx.sessionManager.getSessionFile();
		const sessionId = ctx.sessionManager.getSessionId();

		if (!sessionFile || !sessionId) {
			// In-memory session without ID — can't derive filename
			slotsActive = false;
			return;
		}

		slotState = {
			slotId,
			sessionFile,
			binFilename: deriveBinFilename(sessionId, modelName),
			serverUrl,
			modelName,
		};
		slotsActive = true;
		firstTurn = true;
		isNewSession = true;

		// Persist the new slot allocation
		persistSlotState(pi);
		log(`[llamacpp-slots] Allocated slot ${slotState.slotId} for session ${sessionId} (bin=${slotState.binFilename})`);
		// ctx.ui.notify(`llamacpp-slots: allocated slot ${slotState.slotId}`, "info");
		// ctx.ui.setStatus("llamacpp-slots", `slot ${slotState.slotId} allocated`);
	});

	// ── Model Switch: Save old slot, allocate new one for new model ──

	pi.on("model_select", async (event, ctx) => {
		if (!slotState) return;
		const prevId = event.previousModel?.id ?? "<unknown>";
		log(`[llamacpp-slots] model_select: ${prevId} -> ${event.model.id}`);

		// Save current slot state before switching
		log(`[llamacpp-slots] Saving slot ${slotState.slotId} before model switch`);
		saveSlot(slotState);

		// Derive new model name
		const settings = loadSettings();
		const newModelName = settings.modelName ?? event.model.id;

		// Update model name regardless of discovery outcome
		slotState.modelName = newModelName;

		// Discover a slot for the new model
		const newSlotId = await discoverSlots(slotState.serverUrl, newModelName);
		if (newSlotId == null) {
			log("[llamacpp-slots] Model switch: slot discovery failed — keeping old slot ID");
			ctx.ui.notify("llamacpp-slots: slot discovery failed on model switch", "warning");
			setModelReloadPending(true);
			return;
		}

		// Update slot state for new model
		const sessionId = ctx.sessionManager.getSessionId();
		if (!sessionId) {
			log("[llamacpp-slots] Model switch: no session ID — keeping old slot");
			return;
		}

		const oldModelName = slotState.modelName;
		slotState.slotId = newSlotId;
		slotState.modelName = newModelName;
		slotState.binFilename = deriveBinFilename(sessionId, newModelName);
		firstTurn = true;
		// Always try restore on model switch — fails gracefully if .bin doesn't exist
		isNewSession = false;
		// Only trigger model load if model actually changed (avoid reload on same-model switch)
		setModelReloadPending(oldModelName !== newModelName);

		persistSlotState(pi);
		log(`[llamacpp-slots] Model switch complete: slot=${newSlotId}, bin=${slotState.binFilename}`);
	});

	// ── Session Shutdown: Final Save + Conditional Erase ──────

	pi.on("session_shutdown", async (_event, _ctx) => {
		if (!slotState) return;

		const settings = loadSettings();

		// Optional final save before shutdown (disabled by default).
		// Per-turn saves via turn_end/agent_end are usually sufficient.
		// NOTE: Enabling this with router mode *could* trigger a model reload on quit
		// if you quit another session which is using a different model that is not loaded.
		if (settings.saveOnShutdown && _event.reason !== "reload") {
			try {
				const { modelParam, body } = buildSlotRequest(slotState, { filename: slotState.binFilename });

				const response = await fetch(
					`${slotState.serverUrl}/slots/${slotState.slotId}?action=save${modelParam}`,
					{
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify(body),
					},
				);
				if (response.ok) {
					log(`[llamacpp-slots] Final save: ${slotState.binFilename}`);
				}
			} catch (err) {
				// Server may already be shutting down
				log(`[llamacpp-slots] Final save error: ${(err as Error).message}`);
			}
		}

		// Erase in-memory KV cache only on quit AND only if configured
		if (_event.reason === "quit" && settings.eraseOnQuit) {
			await eraseSlot(slotState);
			log("[llamacpp-slots] Erased slot KV cache (eraseOnQuit=true)");
		}

		// Persist final state
		persistSlotState(pi);
		log(`[llamacpp-slots] session_shutdown (reason=${_event.reason}): persisted slot ${slotState.slotId}, bin=${slotState.binFilename}`);
	});

	// ── Turn Start: Restore Slot if Cold (ensures KV cache is loaded before requests) ──

	pi.on("turn_start", async (_event, _ctx) => {
		if (!slotsActive || !slotState) return;
		const state = slotState;  // Capture for TS narrowing across awaits

		// If model reload is pending, wait for slot to be ready before proceeding
		// (prevents sending messages while router is still loading the new model)
		if (getModelReloadPending()) {
			log(`[llamacpp-slots] Waiting for model load to complete (slot ${state.slotId})`);
			let attempts = 0;
			const maxAttempts = 30;  // 30s max wait
			while (attempts < maxAttempts) {
				await new Promise((r) => setTimeout(r, 1000));
				// Poll without model param to avoid triggering router model switch
				const slotInfo = await getSlotInfo(state.serverUrl, state.slotId, state.modelName, false);
				if (slotInfo && !slotInfo.is_processing && (slotInfo.n_prompt_tokens !== undefined)) {
					log(`[llamacpp-slots] Model load complete (slot ${state.slotId})`);
					break;
				}
				attempts++;
				if (attempts % 5 === 0) {
					log(`[llamacpp-slots] Still waiting for model load... (${attempts}s)`);
				}
			}
			if (attempts >= maxAttempts) {
				log(`[llamacpp-slots] Timeout waiting for model load (slot ${state.slotId})`);
			}
			// Model load triggered — flag cleared after wait loop (not in buildSlotRequest)
			// so restoreSlot retries still see modelReloadPending=true
			setModelReloadPending(false);
		}

		// Omit model param on first turn of resumed session to avoid triggering
		// model reload. Slot already allocated for this model — just check state.
		const skipModel = firstTurn && !isNewSession;
		const slotInfo = await getSlotInfo(state.serverUrl, state.slotId, state.modelName, !skipModel);
		// Server can return "idle" string for n_prompt_tokens when no task is assigned
		const nTokensRaw = slotInfo?.n_prompt_tokens as number | string | undefined;

		if (slotInfo?.is_processing) {
			// Slot is actively processing — skip restore to avoid interference.
			log(`[llamacpp-slots] Slot ${state.slotId} is processing — skipping restore`);
		} else if (firstTurn && isNewSession) {
			// First turn of a new session — no .bin file exists yet, skip restore.
			log(`[llamacpp-slots] Slot ${state.slotId} first turn of new session — skipping restore (no .bin yet)`);
		} else if (firstTurn && !isNewSession) {
			// First turn of a resumed session: restore from .bin.
			// n_prompt_tokens is unreliable here — it's only reported when
			// task or task_prev exists, and may be missing even on warm slots.
			log(`[llamacpp-slots] Slot ${state.slotId} first turn — restoring`);
			restoringPromise = restoreSlot(state).then((ok) => {
				if (ok) {
					log(`[llamacpp-slots] Restored slot ${state.slotId} from ${state.binFilename}`);
					// ctx.ui.setStatus("llamacpp-slots", `slot ${state.slotId} restored`);
				} else {
					log(`[llamacpp-slots] Restore failed for slot ${state.slotId} (file may not exist)`);
					// ctx.ui.setStatus("llamacpp-slots", `slot ${state.slotId} restore failed`);
				}
				restoringPromise = null;
				return ok;
			});
			await restoringPromise;
		} else if (nTokensRaw === 0 || nTokensRaw === undefined || nTokensRaw === "idle") {
			// Subsequent turn, slot is cold (0, missing, or idle) — llama.cpp restarted mid-session.
			log(`[llamacpp-slots] Slot ${state.slotId} cold (n_prompt_tokens=${nTokensRaw}) — restoring`);
			// Server restarted, model may need reloading
			setModelReloadPending(true);
			restoringPromise = restoreSlot(state).then((ok) => {
				if (ok) {
					log(`[llamacpp-slots] Restored slot ${state.slotId} from ${state.binFilename}`);
					// ctx.ui.setStatus("llamacpp-slots", `slot ${state.slotId} restored`);
				} else {
					log(`[llamacpp-slots] Restore failed for slot ${state.slotId}`);
					// ctx.ui.setStatus("llamacpp-slots", `slot ${state.slotId} restore failed`);
				}
				restoringPromise = null;
				return ok;
			});
			await restoringPromise;
		} else {
			// nTokens > 0 (warm) — skip restore.
			log(`[llamacpp-slots] Slot ${state.slotId} warm (n_prompt_tokens=${nTokensRaw ?? "idle"}) — skipping restore`);
		}

		restoringPromise = null;
		firstTurn = false;
	});

	// ── Turn End / Agent End: Fire-and-Forget Slot Save ──────

	// Determine save timing from settings on each handler (settings can change between reloads).
	pi.on("turn_end", async (_event, _ctx) => {
		if (!slotsActive || !slotState) return;
		const settings = loadSettings();
		if (settings.saveOnAgentEnd) return;  // Defer to agent_end
		log(`[llamacpp-slots] turn_end: saving slot ${slotState.slotId} to ${slotState.binFilename}`);
		saveSlot(slotState);
	});

	pi.on("agent_end", async (_event, _ctx) => {
		if (!slotsActive || !slotState) return;
		const settings = loadSettings();
		if (!settings.saveOnAgentEnd) return;  // Already saved on turn_end
		log(`[llamacpp-slots] agent_end: saving slot ${slotState.slotId} to ${slotState.binFilename}`);
		saveSlot(slotState);
	});

	// ── Before Provider Request: Inject id_slot + Wait for Restore ──

	pi.on("before_provider_request", async (event, ctx) => {
		// Avoid reload on session_start in router mode; discover when first needed
		if (isRouterMode) await tryActivateRouterSlot(ctx, pi);
		if (!slotsActive || !slotState) return undefined;
		// Safety net: if turn_start triggered a restore that hasn't finished yet,
		// wait for it before sending the request. Prevents the race where
		// before_provider_request fires before restoreSlot completes.
		if (restoringPromise) {
			log("[llamacpp-slots] before_provider_request: waiting for restore to complete");
			await restoringPromise;
		}
		// Inject id_slot into the provider payload for deterministic slot routing.
		// The OpenAI SDK's FallbackEncoder JSON.stringifies the full body —
		// custom fields like id_slot survive to the HTTP request.
		const payload = event.payload as Record<string, unknown>;
		return {
			...payload,
			id_slot: slotState.slotId,
		};
	});
}
