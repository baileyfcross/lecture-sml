import type { GenerationResult, GenerationStage, ProgressEvent } from "./types";

export type TabId = "output" | "sources" | "grounding" | "diagnostics";
export type GenerationUiStatus = "idle" | "generating" | "completed" | "failed";
export type StageUiStatus = "active" | "complete" | "failed";

export interface StageState {
  stage: GenerationStage;
  status: StageUiStatus;
}

export interface GenerationState {
  status: GenerationUiStatus;
  requestId: string | null;
  stage: GenerationStage | null;
  elapsedSeconds: number;
  latestMessage: string;
  estimateSecondsRemaining: number | null;
  estimateIsApproximate: boolean;
  stages: StageState[];
  result: GenerationResult | null;
  failureKind: "grounding" | "pipeline" | "transport" | null;
  error: string | null;
  activeTab: TabId;
}

export function initialGenerationState(): GenerationState {
  return {
    status: "idle",
    requestId: null,
    stage: null,
    elapsedSeconds: 0,
    latestMessage: "Ready when you are.",
    estimateSecondsRemaining: null,
    estimateIsApproximate: true,
    stages: [],
    result: null,
    failureKind: null,
    error: null,
    activeTab: "output",
  };
}

export function startGeneration(): GenerationState {
  return {
    ...initialGenerationState(),
    status: "generating",
    latestMessage: "Starting generation…",
  };
}

export function applyProgress(
  state: GenerationState,
  progress: ProgressEvent,
): GenerationState {
  const stages = state.stages.map((item) =>
    item.stage === state.stage && state.stage !== progress.stage && item.status === "active"
      ? { ...item, status: "complete" as const }
      : item,
  );
  const currentIndex = stages.findIndex((item) => item.stage === progress.stage);
  const status: StageUiStatus =
    progress.stage === "failed"
      ? "failed"
      : progress.stage === "complete"
        ? "complete"
        : "active";
  if (currentIndex === -1) {
    stages.push({ stage: progress.stage, status });
  } else {
    stages[currentIndex] = { stage: progress.stage, status };
  }

  return {
    ...state,
    stage: progress.stage,
    elapsedSeconds: progress.elapsed_seconds,
    latestMessage: progress.message,
    estimateSecondsRemaining: progress.estimate_seconds_remaining,
    estimateIsApproximate: progress.estimate_is_approximate,
    stages,
  };
}

export function applyResult(
  state: GenerationState,
  result: GenerationResult,
): GenerationState {
  const groundedFailure = result.status === "failed" && result.grounding.reviewed;
  const completed = result.status === "completed";
  const stages = state.stages.map((item) =>
    item.status === "active" ? { ...item, status: "complete" as const } : item,
  );

  return {
    ...state,
    status: completed ? "completed" : "failed",
    requestId: result.request_id,
    stages,
    result,
    failureKind: completed ? null : groundedFailure ? "grounding" : "pipeline",
    error: null,
    latestMessage: completed
      ? "Generation complete."
      : groundedFailure
        ? "Grounding validation did not pass."
        : "Generation did not complete.",
    activeTab: completed ? "output" : groundedFailure ? "grounding" : "diagnostics",
  };
}

export function applyTransportError(state: GenerationState, error: string): GenerationState {
  return {
    ...state,
    status: "failed",
    failureKind: "transport",
    error,
    latestMessage: error,
    activeTab: "diagnostics",
  };
}

export function selectTab(state: GenerationState, tab: TabId): GenerationState {
  return { ...state, activeTab: tab };
}

export function healthLabel(
  health: {
    generation_ready: boolean;
    ollama_reachable: boolean;
    knowledge_index_available: boolean | null;
  } | null,
  unavailable = false,
): string {
  if (unavailable || health === null) return "Server Unavailable";
  if (!health.generation_ready || !health.ollama_reachable) {
    return "API Available - Model Unavailable";
  }
  if (health.knowledge_index_available === false) {
    return "API Available - Knowledge Index Unavailable";
  }
  return "Ready";
}
