import { describe, expect, it } from "vitest";
import {
  applyProgress,
  applyResult,
  applyTransportError,
  healthLabel,
  initialGenerationState,
  startGeneration,
} from "../src/state";
import { SseParser } from "../src/sse";
import type { GenerationResult, ProgressEvent, SseMessage } from "../src/types";

const progress = (stage: ProgressEvent["stage"], message = stage): ProgressEvent => ({
  stage,
  message,
  elapsed_seconds: 4.2,
  generated_tokens: null,
  tokens_per_second: null,
  estimate_seconds_remaining: 12,
  estimate_rate_source: "fallback",
  estimate_is_approximate: true,
});

const result = (
  status: GenerationResult["status"],
  finalDecision: string | null = null,
): GenerationResult => ({
  request_id: "request-123",
  status,
  task: "explanation",
  profile: "standard",
  output: status === "completed" ? "# Approved" : null,
  errors: status === "failed" ? ["No final artifact approved"] : [],
  grounding: {
    reviewed: finalDecision !== null,
    revision_performed: status === "failed",
    initial_decision: "revision_required",
    final_decision: finalDecision,
  },
  sources: { retrieved: 2, assembled: 2 },
  timing: {
    total_seconds: 4,
    planner_seconds: null,
    writer_seconds: 2,
    initial_review_seconds: null,
    revision_seconds: null,
    final_review_seconds: null,
    selected_contexts: {},
    estimated_input_tokens: {},
  },
  saved_run: null,
});

function parseChunks(chunks: string[]): SseMessage[] {
  const parser = new SseParser();
  const messages: SseMessage[] = [];
  for (const chunk of chunks) parser.push(chunk, (message) => messages.push(message));
  parser.finish((message) => messages.push(message));
  return messages;
}

describe("SSE parser", () => {
  it("parses multiple progress and result records from one chunk", () => {
    const messages = parseChunks([
      'event: progress\ndata: {"stage":"writing"}\n\n' +
        'event: result\ndata: {"status":"completed"}\n\n',
    ]);
    expect(messages).toEqual([
      { event: "progress", data: '{"stage":"writing"}' },
      { event: "result", data: '{"status":"completed"}' },
    ]);
  });

  it("handles partial network chunks and data spanning line boundaries", () => {
    const messages = parseChunks([
      "event: res",
      "ult\r\ndata: {\"output\":\r",
      '\ndata: "approved"}\r\n\r',
      "\n",
    ]);
    expect(messages).toEqual([
      { event: "result", data: '{"output":\n"approved"}' },
    ]);
  });
});

describe("generation state", () => {
  it("tracks actual stage changes without inserting unobserved stages", () => {
    const running = applyProgress(
      applyProgress(startGeneration(), progress("preparing")),
      progress("writing"),
    );
    expect(running.stages).toEqual([
      { stage: "preparing", status: "complete" },
      { stage: "writing", status: "active" },
    ]);
    expect(running.elapsedSeconds).toBe(4.2);
    expect(running.estimateIsApproximate).toBe(true);
  });

  it("selects the output tab after success", () => {
    const state = applyResult(startGeneration(), result("completed"));
    expect(state.status).toBe("completed");
    expect(state.activeTab).toBe("output");
    expect(state.result?.output).toBe("# Approved");
  });

  it("classifies grounding rejection as a pipeline result, not transport failure", () => {
    const state = applyResult(
      startGeneration(),
      result("failed", "revision_required"),
    );
    expect(state.failureKind).toBe("grounding");
    expect(state.activeTab).toBe("grounding");
    expect(state.error).toBeNull();
  });

  it("classifies a failed grounding review without a final decision as grounding failure", () => {
    const failedReview = result("failed");
    failedReview.grounding.reviewed = true;
    const state = applyResult(startGeneration(), failedReview);
    expect(state.failureKind).toBe("grounding");
    expect(state.activeTab).toBe("grounding");
  });

  it("keeps transport failures distinct from generation results", () => {
    const state = applyTransportError(initialGenerationState(), "Server unavailable");
    expect(state.failureKind).toBe("transport");
    expect(state.result).toBeNull();
    expect(state.activeTab).toBe("diagnostics");
  });
});

describe("health label", () => {
  it("reports ready and degraded API states", () => {
    expect(
      healthLabel({
        generation_ready: true,
        ollama_reachable: true,
        knowledge_index_available: true,
      }),
    ).toBe("Ready");
    expect(
      healthLabel({
        generation_ready: true,
        ollama_reachable: true,
        knowledge_index_available: false,
      }),
    ).toBe("API Available - Knowledge Index Unavailable");
    expect(
      healthLabel({
        generation_ready: false,
        ollama_reachable: false,
        knowledge_index_available: null,
      }),
    ).toBe("API Available - Model Unavailable");
    expect(healthLabel(null, true)).toBe("Server Unavailable");
  });
});
