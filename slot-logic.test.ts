import { describe, it, expect, vi, beforeEach } from "vitest";
import type { SlotState } from "./slot-logic";

// ── Helpers ──────────────────────────────────────────────────

function makeState(overrides?: Partial<SlotState>): SlotState {
	return {
		slotId: 0,
		sessionFile: "/tmp/session_abc",
		binFilename: "session_abc.bin",
		serverUrl: "http://localhost:8080",
		...overrides,
	};
}

// ── discoverSlots ────────────────────────────────────────────

describe("discoverSlots", () => {
	beforeEach(() => vi.restoreAllMocks());

	it("returns first available slot, skips busy", async () => {
		const slots = [
			{ id: 0, state: "processing" },
			{ id: 1, state: "available" },
			{ id: 2, state: "available" },
		];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { discoverSlots } = await import("./slot-logic");
		expect(await discoverSlots("http://localhost:8080")).toBe(1);
	});

	it("returns first slot when none available", async () => {
		const slots = [{ id: 0, state: "processing" }, { id: 1, state: "idle" }];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { discoverSlots } = await import("./slot-logic");
		expect(await discoverSlots("http://localhost:8080")).toBe(0);
	});

	it("returns null on HTTP error", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: false,
			status: 500,
		} as Response);

		const { discoverSlots } = await import("./slot-logic");
		expect(await discoverSlots("http://localhost:8080")).toBeNull();
	});

	it("returns null on network error", async () => {
		vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("ECONNREFUSED"));

		const { discoverSlots } = await import("./slot-logic");
		expect(await discoverSlots("http://localhost:8080")).toBeNull();
	});

	it("includes model param in URL when modelName set", async () => {
		const slots = [{ id: 0, state: "available" }];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { discoverSlots } = await import("./slot-logic");
		await discoverSlots("http://localhost:8080", "org/model:Q4");

		expect(fetch).toHaveBeenCalledWith(
			"http://localhost:8080/slots?model=org%2Fmodel%3AQ4",
			expect.objectContaining({ signal: expect.any(AbortSignal) }),
		);
	});
});

// ── getSlotInfo ──────────────────────────────────────────────

describe("getSlotInfo", () => {
	beforeEach(() => vi.restoreAllMocks());

	it("returns slot by ID", async () => {
		const slots = [
			{ id: 0, state: "available", n_prompt_tokens: 100 },
			{ id: 1, state: "processing" },
		];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { getSlotInfo } = await import("./slot-logic");
		const info = await getSlotInfo("http://localhost:8080", 1);
		expect(info?.id).toBe(1);
		expect(info?.state).toBe("processing");
	});

	it("returns null when slot not found", async () => {
		const slots = [{ id: 0, state: "available" }];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { getSlotInfo } = await import("./slot-logic");
		expect(await getSlotInfo("http://localhost:8080", 99)).toBeNull();
	});

	it("includes model param by default", async () => {
		const slots = [{ id: 0, state: "available" }];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { getSlotInfo } = await import("./slot-logic");
		await getSlotInfo("http://localhost:8080", 0, "model", undefined);

		expect(fetch).toHaveBeenCalledWith(
			"http://localhost:8080/slots?model=model",
			expect.anything(),
		);
	});

	it("omits model param when includeModel=false", async () => {
		const slots = [{ id: 0, state: "available" }];
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(slots),
		} as Response);

		const { getSlotInfo } = await import("./slot-logic");
		await getSlotInfo("http://localhost:8080", 0, "model", false);

		expect(fetch).toHaveBeenCalledWith(
			"http://localhost:8080/slots",
			expect.anything(),
		);
	});
});

// ── restoreSlot ──────────────────────────────────────────────

describe("restoreSlot", () => {
	beforeEach(() => vi.restoreAllMocks());

	it("returns true on success, sends correct request", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
		} as Response);

		const { restoreSlot } = await import("./slot-logic");
		const result = await restoreSlot(makeState());
		expect(result).toBe(true);
		expect(fetch).toHaveBeenCalledTimes(1);
		expect(fetch).toHaveBeenCalledWith(
			"http://localhost:8080/slots/0?action=restore",
			expect.objectContaining({
				method: "POST",
				body: JSON.stringify({ filename: "session_abc.bin" }),
			}),
		);
	});

	it("returns false on persistent failure", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: false,
			status: 500,
		} as Response);

		const { restoreSlot } = await import("./slot-logic");
		const result = await restoreSlot(makeState());
		expect(result).toBe(false);
		expect(fetch).toHaveBeenCalledTimes(1); // no retry without modelReloadPending
	});

	it("retries on 400/500 when modelReloadPending=true", async () => {
		vi.spyOn(globalThis, "fetch")
			.mockResolvedValueOnce({ ok: false, status: 400 } as Response)
			.mockResolvedValueOnce({ ok: true } as Response);

		const { restoreSlot, setModelReloadPending } = await import("./slot-logic");
		setModelReloadPending(true);
		const result = await restoreSlot(makeState({ modelName: "model" }));
		expect(result).toBe(true);
		expect(fetch).toHaveBeenCalledTimes(2);
	});

	it("does NOT retry when modelReloadPending=false", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: false,
			status: 500,
		} as Response);

		const { restoreSlot, setModelReloadPending } = await import("./slot-logic");
		setModelReloadPending(false);
		const result = await restoreSlot(makeState());
		expect(result).toBe(false);
		expect(fetch).toHaveBeenCalledTimes(1);
	});

	it("returns false on network error", async () => {
		vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("ECONNREFUSED"));

		const { restoreSlot } = await import("./slot-logic");
		const result = await restoreSlot(makeState());
		expect(result).toBe(false);
	});

	it("includes model param in body when modelName set", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
		} as Response);

		const { restoreSlot } = await import("./slot-logic");
		await restoreSlot(makeState({ modelName: "org/model:Q4_K_M" }));

		expect(fetch).toHaveBeenCalledWith(
			"http://localhost:8080/slots/0?action=restore&model=org%2Fmodel%3AQ4_K_M",
			expect.objectContaining({
				method: "POST",
				body: JSON.stringify({
					filename: "session_abc.bin",
					model: "org/model:Q4_K_M",
				}),
			}),
		);
	});
});

// ── eraseSlot ────────────────────────────────────────────────

describe("eraseSlot", () => {
	beforeEach(() => vi.restoreAllMocks());

	it("sends POST with model param", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue({
			ok: true,
		} as Response);

		const { eraseSlot } = await import("./slot-logic");
		await eraseSlot(makeState({ modelName: "model" }));

		expect(fetch).toHaveBeenCalledWith(
			"http://localhost:8080/slots/0?action=erase&model=model",
			expect.objectContaining({
				method: "POST",
				body: JSON.stringify({ model: "model" }),
			}),
		);
	});

	it("doesn't throw on error", async () => {
		vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("ECONNREFUSED"));

		const { eraseSlot } = await import("./slot-logic");
		await expect(eraseSlot(makeState())).resolves.toBeUndefined();
	});
});
