import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, generateStream } from "../src/api";
import { applyTransportError, initialGenerationState } from "../src/state";
import type { GenerationRequest, GenerationResult } from "../src/types";

const request: GenerationRequest = {
  task: "explanation",
  profile: "quick",
  instruction: "Explain predicate logic.",
  retrieve: false,
  save_run: false,
};

const completedResult: GenerationResult = {
  request_id: "request-1",
  status: "completed",
  task: "explanation",
  profile: "quick",
  output: "Approved output",
  errors: [],
  grounding: {
    reviewed: false,
    revision_performed: false,
    initial: null,
    final: null,
    initial_decision: null,
    final_decision: null,
  },
  sources: { retrieved: 0, assembled: 0 },
  timing: {
    total_seconds: 1,
    planner_seconds: null,
    writer_seconds: 1,
    initial_review_seconds: null,
    revision_seconds: null,
    final_review_seconds: null,
    selected_contexts: {},
    estimated_input_tokens: {},
  },
  saved_run: null,
};

afterEach(() => vi.unstubAllGlobals());

describe("generateStream", () => {
  it("processes streamed progress followed by a result", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode('event: progress\ndata: {"stage":"writing"}\n\n'));
        controller.enqueue(
          encoder.encode(`event: result\ndata: ${JSON.stringify(completedResult)}\n\n`),
        );
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const onProgress = vi.fn();

    const actual = await generateStream(request, onProgress);

    expect(actual).toEqual(completedResult);
    expect(onProgress).toHaveBeenCalledWith({ stage: "writing" });
  });

  it("turns an HTTP error into a transport failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "Ollama unavailable" }), {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    let failure: unknown;
    try {
      await generateStream(request, () => undefined);
    } catch (error) {
      failure = error;
    }

    expect(failure).toBeInstanceOf(ApiError);
    expect(failure).toMatchObject({ status: 503, message: "Ollama unavailable" });
    const state = applyTransportError(
      initialGenerationState(),
      failure instanceof Error ? failure.message : "Unknown API error",
    );
    expect(state.failureKind).toBe("transport");
    expect(state.result).toBeNull();
  });
});
