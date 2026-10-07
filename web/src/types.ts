export type TaskType =
  | "lecture"
  | "slides"
  | "lab"
  | "activity"
  | "instructor_guide"
  | "homework"
  | "assessment"
  | "explanation";

export type ProfileName = "quick" | "standard" | "deep";
export type GenerationStatus = "completed" | "failed";
export type GenerationStage =
  | "preparing"
  | "planning"
  | "writing"
  | "reviewing"
  | "revising"
  | "complete"
  | "failed";

export interface HealthResponse {
  status: string;
  service: string;
  generation_ready: boolean;
  ollama_reachable: boolean;
  model_available: boolean;
  model: string;
  knowledge_index_available: boolean | null;
  knowledge_index_error: string | null;
}

export interface TasksResponse {
  tasks: TaskType[];
}

export interface ProfileSummary {
  id: ProfileName;
  purpose: string;
  planner_enabled: boolean;
  grounding_review_for_sourced_requests: boolean;
}

export interface ProfilesResponse {
  profiles: ProfileSummary[];
}

export interface GenerationRequest {
  task: TaskType;
  profile: ProfileName;
  instruction: string;
  retrieve: boolean;
  retrieval_top_k?: number;
  save_run: boolean;
}

export interface ProgressEvent {
  stage: GenerationStage;
  message: string;
  elapsed_seconds: number;
  generated_tokens: number | null;
  tokens_per_second: number | null;
  estimate_seconds_remaining: number | null;
  estimate_rate_source: "fallback" | "observed_previous_stage" | null;
  estimate_is_approximate: boolean;
}

export interface GroundingSummary {
  reviewed: boolean;
  revision_performed: boolean;
  initial: GroundingPhaseSummary | null;
  final: GroundingPhaseSummary | null;
  initial_decision: string | null;
  final_decision: string | null;
}

export interface GroundingPhaseSummary {
  claims_extracted: number;
  direct_supported: number;
  reviewer_supported: number;
  pedagogical: number;
  unsupported: number;
  evidence_validation_failures: number;
  decision: string;
}

export interface SourceSummary {
  retrieved: number;
  assembled: number;
}

export interface SourceCoverage {
  status: "sufficient" | "partial" | "insufficient";
  supported_topics: string[];
  unsupported_requested_topics: string[];
  scope_note: string | null;
}

export interface TimingSummary {
  total_seconds: number;
  planner_seconds: number | null;
  writer_seconds: number | null;
  initial_review_seconds: number | null;
  revision_seconds: number | null;
  final_review_seconds: number | null;
  selected_contexts: Record<string, number>;
  estimated_input_tokens: Record<string, number>;
}

export interface GenerationResult {
  request_id: string;
  status: GenerationStatus;
  task: TaskType;
  profile: ProfileName;
  output: string | null;
  errors: string[];
  source_coverage: SourceCoverage | null;
  grounding: GroundingSummary;
  sources: SourceSummary;
  timing: TimingSummary;
  saved_run: string | null;
}

export interface SseMessage {
  event: string;
  data: string;
}
