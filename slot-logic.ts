/**
 * Pure logic for llamacpp-slots extension.
 * Extracted from index.ts for testability.
 */

// ── Types ────────────────────────────────────────────────────

export interface SlotState {
	slotId: number;
	sessionFile: string;
	binFilename: string;
	serverUrl: string;
	modelName?: string;
}

// ── Model Reload Pending Flag ────────────────────────────────

let _modelReloadPending = false;

/**
 * Get/set the model reload pending flag.
 * Used by restoreSlot to decide whether to retry on 400/500.
 */
export function getModelReloadPending(): boolean {
	return _modelReloadPending;
}

export function setModelReloadPending(value: boolean): void {
	_modelReloadPending = value;
}

// ── Pure Functions ───────────────────────────────────────────

/**
 * Derive a deterministic .bin filename from the session ID.
 */
export function deriveBinFilename(sessionId: string, modelName?: string): string {
	const cleanId = sessionId.replace(/-/g, "");
	const modelSuffix = modelName ? `_${modelName.replace(/[^a-zA-Z0-9]/g, "_")}` : "";
	return `session_${cleanId}${modelSuffix}.bin`;
}

/**
 * Build query param and body for a slot API call.
 * Always includes `model` param when state.modelName is set so router routes to correct model.
 */
export function buildSlotRequest(state: SlotState, extraBody?: Record<string, string>): { modelParam: string; body: Record<string, string> } {
	const modelParam = state.modelName ? `&model=${encodeURIComponent(state.modelName)}` : "";
	const body: Record<string, string> = { ...extraBody };
	if (state.modelName) {
		body.model = state.modelName;
	}
	return { modelParam, body };
}

// ── Async Functions (HTTP) ───────────────────────────────────

/**
 * Discover an available slot by probing GET /slots.
 * Returns the first available slot ID, or the first slot if none available, or null.
 */
export async function discoverSlots(serverUrl: string, modelName?: string): Promise<number | null> {
	try {
		const modelParam = modelName ? `?model=${encodeURIComponent(modelName)}` : "";
		const response = await fetch(`${serverUrl}/slots${modelParam}`, {
			signal: AbortSignal.timeout(3000),
		});
		if (!response.ok) return null;

		const data = await response.json();
		if (Array.isArray(data)) {
			for (const slot of data) {
				if (slot.state === "available" || slot.state === "loading") {
					return slot.id;
				}
			}
			return data[0]?.id ?? null;
		}
		return null;
	} catch {
		return null;
	}
}

/**
 * Get full slot info from GET /slots.
 * Returns the slot object (with state, n_prompt_tokens, etc.) or null if unavailable.
 */
export async function getSlotInfo(serverUrl: string, slotId: number, modelName?: string, includeModel?: boolean): Promise<{ is_processing?: boolean; n_prompt_tokens?: number | string; model?: string } | null> {
	try {
		const modelParam = (includeModel !== false && modelName) ? `?model=${encodeURIComponent(modelName)}` : "";
		const response = await fetch(`${serverUrl}/slots${modelParam}`, {
			signal: AbortSignal.timeout(3000),
		});
		if (!response.ok) return null;

		const data = await response.json();
		if (Array.isArray(data)) {
			const slot = data.find((s: any) => s.id === slotId);
			return slot ?? null;
		}
		return null;
	} catch {
		return null;
	}
}

/**
 * Restore a slot's KV cache from a .bin file.
 * Retries briefly if model is still loading (modelReloadPending=true).
 */
export async function restoreSlot(state: SlotState): Promise<boolean> {
	const maxRetries = 3;
	const retryDelay = 2000; // ms

	for (let attempt = 0; attempt <= maxRetries; attempt++) {
		try {
			const { modelParam, body } = buildSlotRequest(state, { filename: state.binFilename });

			const response = await fetch(
				`${state.serverUrl}/slots/${state.slotId}?action=restore${modelParam}`,
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(body),
				},
			);
			if (response.ok) return true;

			// Retry on 400/500 if modelReloadPending (model likely still loading)
			if (getModelReloadPending() && [400, 500].includes(response.status) && attempt < maxRetries) {
				await new Promise((r) => setTimeout(r, retryDelay));
				continue;
			}

			return false;
		} catch {
			return false;
		}
	}
	return false;
}

/**
 * Erase a slot's in-memory KV cache.
 * Does not throw on error.
 */
export async function eraseSlot(state: SlotState): Promise<void> {
	try {
		const { modelParam, body } = buildSlotRequest(state);

		const response = await fetch(
			`${state.serverUrl}/slots/${state.slotId}?action=erase${modelParam}`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(body),
			},
		);
		if (!response.ok) {
			// Server may already be shutting down
		}
	} catch {
		// Server may already be shutting down
	}
}
