import DOMPurify from "dompurify";
import { marked } from "marked";
import {
  ApiError,
  generateStream,
  getHealth,
  getProfiles,
  getTasks,
} from "./api";
import {
  applyProgress,
  applyResult,
  applyTransportError,
  healthLabel,
  initialGenerationState,
  selectTab,
  startGeneration,
  type TabId,
} from "./state";
import type { GenerationRequest, GenerationStage, HealthResponse, ProfileSummary, TaskType } from "./types";
import "./styles.css";

const tabs: TabId[] = ["output", "sources", "grounding", "diagnostics"];
const storageKeys = {
  task: "lecture-slm.task",
  profile: "lecture-slm.profile",
  retrieve: "lecture-slm.retrieve",
  topK: "lecture-slm.top-k",
  saveRun: "lecture-slm.save-run",
};

let generation = initialGenerationState();
let profiles: ProfileSummary[] = [];
let health: HealthResponse | null = null;
let savedRunPath: string | null = null;

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing UI element: ${id}`);
  return element as T;
}

const form = byId<HTMLFormElement>("generation-form");
const taskSelect = byId<HTMLSelectElement>("task");
const profileSelect = byId<HTMLSelectElement>("profile");
const retrieveCheckbox = byId<HTMLInputElement>("retrieve");
const saveRunCheckbox = byId<HTMLInputElement>("save-run");
const topKInput = byId<HTMLInputElement>("top-k");
const instructionInput = byId<HTMLTextAreaElement>("instruction");
const generateButton = byId<HTMLButtonElement>("generate");
const requestError = byId<HTMLParagraphElement>("request-error");
const healthIndicator = byId<HTMLSpanElement>("health-indicator");
const healthLabelElement = byId<HTMLSpanElement>("health-label");
const refreshHealthButton = byId<HTMLButtonElement>("refresh-health");
const profilePurpose = byId<HTMLSpanElement>("profile-purpose");
const stageList = byId<HTMLOListElement>("stage-list");
const outputMarkdown = byId<HTMLElement>("output-markdown");
const outputEmpty = byId<HTMLParagraphElement>("output-empty");
const copyOutputButton = byId<HTMLButtonElement>("copy-output");
const sourcesDetails = byId<HTMLDListElement>("sources-details");
const groundingMessage = byId<HTMLParagraphElement>("grounding-message");
const groundingDetails = byId<HTMLDListElement>("grounding-details");
const diagnosticDetails = byId<HTMLDListElement>("diagnostic-details");
const diagnosticErrors = byId<HTMLDivElement>("diagnostic-errors");
const errorItems = byId<HTMLUListElement>("error-items");
const copyRunPathButton = byId<HTMLButtonElement>("copy-run-path");
const copyMessage = byId<HTMLSpanElement>("copy-message");

function title(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function setHealthIndicator(label: string, status: "ready" | "warning" | "unavailable"): void {
  healthLabelElement.textContent = label;
  healthIndicator.dataset.status = status;
}

function renderHealth(): void {
  const label = healthLabel(health);
  const status = health === null
    ? "unavailable"
    : label === "Ready"
      ? "ready"
      : "warning";
  setHealthIndicator(label, status);
}

async function refreshHealth(): Promise<void> {
  refreshHealthButton.disabled = true;
  try {
    health = await getHealth();
    renderHealth();
  } catch (error) {
    health = null;
    setHealthIndicator(healthLabel(null, true), "unavailable");
    byId<HTMLSpanElement>("health-label").title =
      error instanceof Error ? error.message : String(error);
  } finally {
    refreshHealthButton.disabled = false;
  }
}

function setOptions<T extends string>(
  select: HTMLSelectElement,
  values: T[],
  preferred: string | null,
): void {
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = title(value);
    select.append(option);
  }
  if (preferred && values.includes(preferred as T)) select.value = preferred;
}

function updateProfilePurpose(): void {
  const profile = profiles.find((item) => item.id === profileSelect.value);
  if (!profile) {
    profilePurpose.textContent = "";
    return;
  }
  const workflow = profile.planner_enabled ? "Planning enabled" : "Writer-only";
  const grounding = profile.grounding_review_for_sourced_requests
    ? "Grounding review for sourced requests"
    : "No grounding review";
  profilePurpose.textContent = `${profile.purpose} ${workflow}. ${grounding}.`;
}

function instructionIsValid(): boolean {
  return instructionInput.value.trim().length > 0;
}

function updateFormState(): void {
  const isGenerating = generation.status === "generating";
  const hasOptions = taskSelect.value !== "" && profileSelect.value !== "";
  generateButton.disabled = isGenerating || !hasOptions || !instructionIsValid();
  generateButton.textContent = isGenerating ? "Generating…" : "Generate";
  taskSelect.disabled = isGenerating;
  profileSelect.disabled = isGenerating;
  retrieveCheckbox.disabled = isGenerating;
  saveRunCheckbox.disabled = isGenerating;
  topKInput.disabled = isGenerating || !retrieveCheckbox.checked;
  instructionInput.disabled = isGenerating;
}

function stageLabel(stage: GenerationStage): string {
  const labels: Record<GenerationStage, string> = {
    preparing: "Preparing",
    planning: "Planning",
    writing: "Writing",
    reviewing: "Grounding Review",
    revising: "Revision",
    complete: "Complete",
    failed: "Failed",
  };
  return labels[stage];
}

function renderStages(): void {
  stageList.replaceChildren();
  for (const item of generation.stages) {
    const listItem = document.createElement("li");
    listItem.className = "stage-item";
    listItem.dataset.status = item.status;
    const stageName = document.createElement("span");
    stageName.textContent = stageLabel(item.stage);
    const status = document.createElement("span");
    status.className = "stage-status";
    status.textContent = title(item.status);
    listItem.append(stageName, status);
    stageList.append(listItem);
  }
}

function renderTabs(): void {
  for (const tab of tabs) {
    const selected = generation.activeTab === tab;
    const tabButton = byId<HTMLButtonElement>(`tab-${tab}`);
    tabButton.setAttribute("aria-selected", String(selected));
    tabButton.tabIndex = selected ? 0 : -1;
    byId<HTMLElement>(`panel-${tab}`).hidden = !selected;
  }
}

function addDetail(list: HTMLDListElement, label: string, value: string): void {
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = value;
  list.append(term, description);
}

function addGroundingSummary(
  list: HTMLDListElement,
  label: string,
  summary:
    | {
        claims_extracted: number;
        direct_supported: number;
        reviewer_supported: number;
        pedagogical: number;
        unsupported: number;
        evidence_validation_failures: number;
        decision: string;
      }
    | null,
): void {
  addDetail(list, `${label} claims extracted`, summary === null ? "Unavailable" : String(summary.claims_extracted));
  addDetail(
    list,
    `${label} direct matches`,
    summary === null ? "Unavailable" : String(summary.direct_supported),
  );
  addDetail(
    list,
    `${label} reviewer-supported`,
    summary === null ? "Unavailable" : String(summary.reviewer_supported),
  );
  addDetail(
    list,
    `${label} pedagogical`,
    summary === null ? "Unavailable" : String(summary.pedagogical),
  );
  addDetail(
    list,
    `${label} unsupported`,
    summary === null ? "Unavailable" : String(summary.unsupported),
  );
  addDetail(
    list,
    `${label} evidence validation failures`,
    summary === null ? "Unavailable" : String(summary.evidence_validation_failures),
  );
  addDetail(
    list,
    `${label} decision`,
    summary === null ? "Unavailable" : title(summary.decision),
  );
}

function formatSeconds(value: number | null): string {
  return value === null ? "Unavailable" : `${value.toFixed(1)} s`;
}

function renderResult(): void {
  const result = generation.result;
  outputMarkdown.replaceChildren();
  outputEmpty.hidden = result?.status === "completed" && result.output !== null;
  outputEmpty.textContent =
    result?.status === "failed"
      ? "No approved final artifact is available."
      : "Your approved output will appear here.";
  copyOutputButton.disabled = result?.status !== "completed" || !result.output;
  if (result?.status === "completed" && result.output) {
    const html = marked.parse(result.output, { async: false });
    outputMarkdown.innerHTML = DOMPurify.sanitize(html);
  }

  sourcesDetails.replaceChildren();
  if (result) {
    addDetail(sourcesDetails, "Sources retrieved", String(result.sources.retrieved));
    addDetail(sourcesDetails, "Sources assembled", String(result.sources.assembled));
  }

  groundingDetails.replaceChildren();
  if (result) {
    const rejected = generation.failureKind === "grounding";
    groundingMessage.hidden = !rejected;
    groundingMessage.className = rejected
      ? "message message-warning"
      : "empty-state";
    groundingMessage.textContent = rejected
      ? `Grounding validation did not produce an approved result. No final artifact was approved.${
          result.grounding.final_decision
            ? ` Final review decision: ${title(result.grounding.final_decision)}.`
            : " No final grounding review decision was recorded."
        }`
      : result.grounding.reviewed
        ? "Grounding review details from the backend response."
        : "No grounding review was performed for this request.";
    addDetail(groundingDetails, "Review performed", result.grounding.reviewed ? "Yes" : "No");
    addDetail(groundingDetails, "Revision performed", result.grounding.revision_performed ? "Yes" : "No");
    addGroundingSummary(groundingDetails, "Initial review", result.grounding.initial);
    addGroundingSummary(groundingDetails, "Final review", result.grounding.final);
  } else {
    groundingMessage.hidden = false;
    groundingMessage.textContent = "Grounding information will appear after generation.";
  }

  diagnosticDetails.replaceChildren();
  const errorMessages = result?.errors ?? [];
  diagnosticErrors.hidden = errorMessages.length === 0;
  errorItems.replaceChildren();
  for (const message of errorMessages) {
    const item = document.createElement("li");
    item.textContent = message;
    errorItems.append(item);
  }
  if (result) {
    addDetail(diagnosticDetails, "Request ID", result.request_id);
    addDetail(diagnosticDetails, "Task", title(result.task));
    addDetail(diagnosticDetails, "Profile", title(result.profile));
    addDetail(diagnosticDetails, "Status", title(result.status));
    addDetail(diagnosticDetails, "Total duration", formatSeconds(result.timing.total_seconds));
    addDetail(diagnosticDetails, "Planner duration", formatSeconds(result.timing.planner_seconds));
    addDetail(diagnosticDetails, "Writer duration", formatSeconds(result.timing.writer_seconds));
    addDetail(
      diagnosticDetails,
      "Initial review duration",
      formatSeconds(result.timing.initial_review_seconds),
    );
    addDetail(diagnosticDetails, "Revision duration", formatSeconds(result.timing.revision_seconds));
    addDetail(
      diagnosticDetails,
      "Final review duration",
      formatSeconds(result.timing.final_review_seconds),
    );
    addDetail(
      diagnosticDetails,
      "Selected contexts",
      JSON.stringify(result.timing.selected_contexts),
    );
    addDetail(
      diagnosticDetails,
      "Estimated input tokens",
      JSON.stringify(result.timing.estimated_input_tokens),
    );
    addDetail(diagnosticDetails, "Saved run", result.saved_run ?? "Not saved");
  } else if (generation.error) {
    addDetail(diagnosticDetails, "Request", "No result received");
    addDetail(diagnosticDetails, "Transport error", generation.error);
  }
  savedRunPath = result?.saved_run ?? null;
  copyRunPathButton.disabled = savedRunPath === null;
}

function render(): void {
  byId<HTMLSpanElement>("generation-label").textContent = title(generation.status);
  byId<HTMLSpanElement>("generation-label").dataset.status = generation.status;
  byId<HTMLParagraphElement>("progress-message").textContent = generation.latestMessage;
  byId<HTMLSpanElement>("elapsed").textContent =
    `Elapsed: ${generation.elapsedSeconds.toFixed(1)} s`;
  byId<HTMLSpanElement>("estimate").textContent =
    generation.estimateSecondsRemaining === null
      ? ""
      : `Estimated stage time: ~${generation.estimateSecondsRemaining.toFixed(0)} s${
          generation.estimateIsApproximate ? " (approximate)" : ""
        }`;
  renderStages();
  renderTabs();
  renderResult();
  updateFormState();
}

function apiErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status === 503
      ? `Lecture SLM could not complete the request: ${error.message}`
      : error.message;
  }
  if (error instanceof TypeError) return "Lecture SLM server unavailable. Check that the local API is running.";
  return error instanceof Error ? error.message : String(error);
}

async function submitGeneration(event: SubmitEvent): Promise<void> {
  event.preventDefault();
  if (generation.status === "generating" || !instructionIsValid()) return;

  requestError.hidden = true;
  generation = startGeneration();
  render();
  const request: GenerationRequest = {
    task: taskSelect.value as TaskType,
    profile: profileSelect.value as GenerationRequest["profile"],
    instruction: instructionInput.value.trim(),
    retrieve: retrieveCheckbox.checked,
    save_run: saveRunCheckbox.checked,
  };
  if (retrieveCheckbox.checked) request.retrieval_top_k = Number(topKInput.value);

  try {
    const result = await generateStream(request, (progress) => {
      generation = applyProgress(generation, progress);
      render();
    });
    generation = applyResult(generation, result);
  } catch (error) {
    const message = apiErrorMessage(error);
    generation = applyTransportError(generation, message);
    requestError.textContent = message;
    requestError.hidden = false;
  }
  render();
}

async function loadOptions(): Promise<void> {
  try {
    const [tasks, availableProfiles] = await Promise.all([getTasks(), getProfiles()]);
    profiles = availableProfiles;
    setOptions(taskSelect, tasks, localStorage.getItem(storageKeys.task));
    setOptions(
      profileSelect,
      profiles.map((profile) => profile.id),
      localStorage.getItem(storageKeys.profile) ?? "standard",
    );
    updateProfilePurpose();
  } catch (error) {
    requestError.textContent = `Could not load generation options: ${apiErrorMessage(error)}`;
    requestError.hidden = false;
  }
}

function setPreference(key: string, value: string): void {
  localStorage.setItem(key, value);
}

function savePreferences(): void {
  if (taskSelect.value) setPreference(storageKeys.task, taskSelect.value);
  if (profileSelect.value) setPreference(storageKeys.profile, profileSelect.value);
  setPreference(storageKeys.retrieve, String(retrieveCheckbox.checked));
  setPreference(storageKeys.topK, topKInput.value);
  setPreference(storageKeys.saveRun, String(saveRunCheckbox.checked));
}

function initializePreferences(): void {
  retrieveCheckbox.checked = localStorage.getItem(storageKeys.retrieve) === "true";
  const savedRunPreference = localStorage.getItem(storageKeys.saveRun);
  saveRunCheckbox.checked = savedRunPreference === null ? true : savedRunPreference === "true";
  const topK = Number(localStorage.getItem(storageKeys.topK));
  if (Number.isInteger(topK) && topK >= 1 && topK <= 10) topKInput.value = String(topK);
}

function setActiveTab(tab: TabId): void {
  generation = selectTab(generation, tab);
  renderTabs();
}

form.addEventListener("submit", (event) => void submitGeneration(event as SubmitEvent));
instructionInput.addEventListener("input", updateFormState);
profileSelect.addEventListener("change", () => {
  updateProfilePurpose();
  savePreferences();
});
taskSelect.addEventListener("change", savePreferences);
retrieveCheckbox.addEventListener("change", () => {
  savePreferences();
  updateFormState();
});
saveRunCheckbox.addEventListener("change", savePreferences);
topKInput.addEventListener("change", savePreferences);
refreshHealthButton.addEventListener("click", () => void refreshHealth());
document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab as TabId));
  button.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const current = tabs.indexOf(button.dataset.tab as TabId);
    const next = (current + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
    setActiveTab(tabs[next]);
    byId<HTMLButtonElement>(`tab-${tabs[next]}`).focus();
  });
});
copyOutputButton.addEventListener("click", async () => {
  const output = generation.result?.output;
  if (!output) return;
  try {
    await navigator.clipboard.writeText(output);
    copyMessage.textContent = "Output copied.";
  } catch (error) {
    copyMessage.textContent = `Could not copy output: ${apiErrorMessage(error)}`;
  }
});
copyRunPathButton.addEventListener("click", async () => {
  if (!savedRunPath) return;
  try {
    await navigator.clipboard.writeText(savedRunPath);
    copyMessage.textContent = "Saved-run path copied.";
  } catch (error) {
    copyMessage.textContent = `Could not copy path: ${apiErrorMessage(error)}`;
  }
});

initializePreferences();
render();
void Promise.all([refreshHealth(), loadOptions()]);
