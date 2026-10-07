import {
  ApiError,
  createWorkspace,
  createWorkspaceFolder,
  createWorkspaceItem,
  deleteWorkspaceFolder,
  deleteWorkspaceItem,
  generateStream,
  getHealth,
  getProfiles,
  getTasks,
  getWorkspace,
  getWorkspaces,
  saveOutputToWorkspace,
  updateWorkspaceItem,
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
import type {
  GenerationRequest,
  GenerationResult,
  GenerationStage,
  HealthResponse,
  ProfileSummary,
  TaskType,
  WorkspaceDetail,
  WorkspaceItemRole,
  WorkspaceSummary,
} from "./types";
import { renderArtifactMarkdown } from "./markdown";
import { renderSourceCoverage } from "./sourceCoverage";
import "./styles.css";

const tabs: TabId[] = ["output", "sources", "grounding", "diagnostics"];
const storageKeys = {
  task: "lecture-slm.task",
  profile: "lecture-slm.profile",
  retrieve: "lecture-slm.retrieve",
  topK: "lecture-slm.top-k",
  saveRun: "lecture-slm.save-run",
  workspaceId: "lecture-slm.workspace-id",
};

let generation = initialGenerationState();
let profiles: ProfileSummary[] = [];
let health: HealthResponse | null = null;
let savedRunPath: string | null = null;
let workspaces: WorkspaceSummary[] = [];
let selectedWorkspace: WorkspaceDetail | null = null;

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
const sourceCoverage = byId<HTMLElement>("source-coverage");
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
const workspaceSelect = byId<HTMLSelectElement>("workspace-select");
const workspaceMessage = byId<HTMLParagraphElement>("workspace-message");
const workspaceItems = byId<HTMLUListElement>("workspace-items");
const workspaceFolders = byId<HTMLUListElement>("workspace-folders");
const workspaceItemForm = byId<HTMLFormElement>("workspace-item-form");
const workspaceFolderForm = byId<HTMLFormElement>("workspace-folder-form");
const workspaceItemTitle = byId<HTMLInputElement>("workspace-item-title");
const workspaceItemRole = byId<HTMLSelectElement>("workspace-item-role");
const workspaceItemContent = byId<HTMLTextAreaElement>("workspace-item-content");
const workspaceItemPinned = byId<HTMLInputElement>("workspace-item-pinned");
const workspaceItemFolder = byId<HTMLSelectElement>("workspace-item-folder");
const workspaceFolderName = byId<HTMLInputElement>("workspace-folder-name");
const workspaceFolderParent = byId<HTMLSelectElement>("workspace-folder-parent");
const workspaceSaveTitle = byId<HTMLInputElement>("workspace-save-title");
const saveOutputButton = byId<HTMLButtonElement>("save-output");

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
  workspaceSelect.disabled = isGenerating;
  for (const managedForm of [workspaceItemForm, workspaceFolderForm]) {
    managedForm.querySelectorAll("input, select, textarea, button").forEach((element) => {
      (
        element as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | HTMLButtonElement
      ).disabled = isGenerating || selectedWorkspace === null;
    });
  }
  saveOutputButton.disabled =
    isGenerating ||
    selectedWorkspace === null ||
    generation.result?.status !== "completed" ||
    !generation.result.output;
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

function renderApprovedOutput(result: GenerationResult | null): void {
  outputMarkdown.replaceChildren();
  renderSourceCoverage(sourceCoverage, result?.source_coverage ?? null);
  outputEmpty.hidden = result?.status === "completed" && result.output !== null;
  outputEmpty.textContent =
    result?.status === "failed"
      ? "No approved final artifact is available."
      : "Your approved output will appear here.";
  copyOutputButton.disabled = result?.status !== "completed" || !result.output;
  if (result?.status === "completed" && result.output && !workspaceSaveTitle.value) {
    workspaceSaveTitle.value = `${title(result.task)} ${result.request_id.slice(0, 8)}`;
  }
  saveOutputButton.disabled =
    selectedWorkspace === null || result?.status !== "completed" || !result.output;
  if (result?.status === "completed" && result.output) {
    outputMarkdown.innerHTML = renderArtifactMarkdown(result.output);
  }
}

async function refreshWorkspaceDetail(): Promise<void> {
    const workspaceId = workspaceSelect.value;
    if (!workspaceId) {
      selectedWorkspace = null;
      workspaceItems.replaceChildren();
      renderWorkspaceFolders();
      workspaceMessage.textContent = "Workspace is optional. Generation remains local.";
      updateFormState();
      return;
    }
    try {
      selectedWorkspace = await getWorkspace(workspaceId);
      workspaceMessage.textContent =
        `${selectedWorkspace.name}: Context and History guide continuity; only References ground facts.`;
      renderWorkspaceItems();
    } catch (error) {
      selectedWorkspace = null;
      workspaceMessage.textContent = `Could not load Workspace: ${apiErrorMessage(error)}`;
    }
    updateFormState();
}

function renderWorkspaceItems(): void {
    workspaceItems.replaceChildren();
    renderWorkspaceFolders();
    if (!selectedWorkspace || selectedWorkspace.items.length === 0) {
      const empty = document.createElement("li");
      empty.className = "empty-state";
      empty.textContent = selectedWorkspace ? "No items in this Workspace yet." : "Select a Workspace.";
      workspaceItems.append(empty);
      return;
    }
    for (const item of selectedWorkspace.items) {
      const row = document.createElement("li");
      row.className = "workspace-item";
      const info = document.createElement("div");
      const itemTitle = document.createElement("strong");
      itemTitle.textContent = item.title;
      const meta = document.createElement("p");
      meta.className = "field-help";
      const folder = selectedWorkspace.folders.find((candidate) => candidate.id === item.folder_id);
      meta.textContent = `${title(item.role)}${item.pinned ? " · Pinned" : ""}${
        folder ? ` · ${folderPath(folder.id)}` : ""
      }`;
      const snippet = document.createElement("p");
      snippet.textContent = item.content.length > 220 ? `${item.content.slice(0, 220)}…` : item.content;
      info.append(itemTitle, meta, snippet);
      const actions = document.createElement("div");
      actions.className = "inline-controls";
      if (item.role !== "reference") {
        const promote = document.createElement("button");
        promote.type = "button";
        promote.className = "button button-quiet";
        promote.textContent = "Promote to Reference";
        promote.addEventListener("click", () => void mutateWorkspaceItem(
          item.id,
          { role: "reference" },
          "Promoted item to Reference.",
        ));
        actions.append(promote);
      }
      const pin = document.createElement("button");
      pin.type = "button";
      pin.className = "button button-quiet";
      pin.textContent = item.pinned ? "Unpin" : "Pin";
      pin.addEventListener("click", () => void mutateWorkspaceItem(
        item.id,
        { pinned: !item.pinned },
        item.pinned ? "Item unpinned." : "Item pinned.",
      ));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "button button-quiet";
      remove.textContent = "Delete";
      remove.addEventListener("click", () => void removeWorkspaceItem(item.id));
      actions.append(pin, remove);
      row.append(info, actions);
      workspaceItems.append(row);
    }
}

function folderPath(folderId: string): string {
    if (!selectedWorkspace) return "";
    const names: string[] = [];
    let folder = selectedWorkspace.folders.find((item) => item.id === folderId);
    while (folder) {
      names.unshift(folder.name);
      const parentId = folder.parent_id;
      folder = parentId
        ? selectedWorkspace.folders.find((item) => item.id === parentId)
        : undefined;
    }
    return names.join(" / ");
}

function renderWorkspaceFolders(): void {
    const folders = selectedWorkspace?.folders ?? [];
    for (const select of [workspaceFolderParent, workspaceItemFolder]) {
      const selected = select.value;
      select.replaceChildren();
      const root = document.createElement("option");
      root.value = "";
      root.textContent = "Workspace root";
      select.append(root);
      for (const folder of folders) {
        const option = document.createElement("option");
        option.value = folder.id;
        option.textContent = folderPath(folder.id);
        select.append(option);
      }
      if (folders.some((folder) => folder.id === selected)) select.value = selected;
    }
    workspaceFolders.replaceChildren();
    if (!folders.length) {
      const empty = document.createElement("li");
      empty.className = "empty-state";
      empty.textContent = selectedWorkspace ? "No folders." : "Select a Workspace.";
      workspaceFolders.append(empty);
      return;
    }
    for (const folder of folders) {
      const row = document.createElement("li");
      row.className = "workspace-item";
      const label = document.createElement("span");
      label.textContent = folderPath(folder.id);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "button button-quiet";
      remove.textContent = "Delete empty folder";
      remove.addEventListener("click", () => void removeWorkspaceFolder(folder.id));
      row.append(label, remove);
      workspaceFolders.append(row);
    }
}

async function reloadWorkspaces(preferredId?: string): Promise<void> {
    workspaces = await getWorkspaces();
    const savedId = preferredId ?? localStorage.getItem(storageKeys.workspaceId);
    workspaceSelect.replaceChildren();
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "No Workspace";
    workspaceSelect.append(none);
    for (const workspace of workspaces) {
      const option = document.createElement("option");
      option.value = workspace.id;
      option.textContent = workspace.name;
      workspaceSelect.append(option);
    }
    workspaceSelect.value = savedId && workspaces.some((workspace) => workspace.id === savedId)
      ? savedId
      : "";
    if (workspaceSelect.value) localStorage.setItem(storageKeys.workspaceId, workspaceSelect.value);
    else localStorage.removeItem(storageKeys.workspaceId);
    await refreshWorkspaceDetail();
}

async function mutateWorkspaceItem(
    itemId: string,
    changes: { role?: WorkspaceItemRole; pinned?: boolean },
    message: string,
  ): Promise<void> {
    if (!selectedWorkspace) return;
    try {
      await updateWorkspaceItem(selectedWorkspace.id, itemId, changes);
      workspaceMessage.textContent = message;
      await refreshWorkspaceDetail();
    } catch (error) {
      workspaceMessage.textContent = `Could not update item: ${apiErrorMessage(error)}`;
    }
}

async function removeWorkspaceItem(itemId: string): Promise<void> {
    if (!selectedWorkspace) return;
    try {
      await deleteWorkspaceItem(selectedWorkspace.id, itemId);
      workspaceMessage.textContent = "Workspace item deleted.";
      await refreshWorkspaceDetail();
    } catch (error) {
      workspaceMessage.textContent = `Could not delete item: ${apiErrorMessage(error)}`;
    }
}

async function removeWorkspaceFolder(folderId: string): Promise<void> {
      if (!selectedWorkspace) return;
      try {
        await deleteWorkspaceFolder(selectedWorkspace.id, folderId);
        workspaceMessage.textContent = "Folder deleted.";
        await refreshWorkspaceDetail();
      } catch (error) {
        workspaceMessage.textContent = `Could not delete folder: ${apiErrorMessage(error)}`;
      }
}

function renderResultDetails(result: GenerationResult | null): void {
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

function renderResult(): void {
  const result = generation.result;
  renderApprovedOutput(result);
  renderResultDetails(result);
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
  if (workspaceSelect.value) request.workspace_id = workspaceSelect.value;
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
    const [tasks, availableProfiles] = await Promise.all([
      getTasks(),
      getProfiles(),
      reloadWorkspaces(),
    ]);
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

async function saveOutput(): Promise<void> {
    const result = generation.result;
    if (!selectedWorkspace || result?.status !== "completed" || !result.output) return;
    const itemTitle = workspaceSaveTitle.value.trim();
    if (!itemTitle) {
      workspaceMessage.textContent = "Enter a title before saving this output.";
      return;
    }
    saveOutputButton.disabled = true;
    try {
      await saveOutputToWorkspace(selectedWorkspace.id, result.request_id, itemTitle);
      workspaceMessage.textContent = "Approved output saved to Workspace History.";
      await refreshWorkspaceDetail();
    } catch (error) {
      workspaceMessage.textContent = `Could not save output: ${apiErrorMessage(error)}`;
    }
}

async function createWorkspaceFromForm(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    const name = byId<HTMLInputElement>("workspace-name").value.trim();
    if (!name) return;
    try {
      const workspace = await createWorkspace(name);
      byId<HTMLFormElement>("workspace-create-form").reset();
      await reloadWorkspaces(workspace.id);
      workspaceMessage.textContent = `Created Workspace ${workspace.name}.`;
    } catch (error) {
      workspaceMessage.textContent = `Could not create Workspace: ${apiErrorMessage(error)}`;
    }
}

async function createWorkspaceItemFromForm(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (!selectedWorkspace) return;
    try {
      await createWorkspaceItem(selectedWorkspace.id, {
        title: workspaceItemTitle.value.trim(),
        role: workspaceItemRole.value as WorkspaceItemRole,
        content: workspaceItemContent.value.trim(),
        pinned: workspaceItemPinned.checked,
        folder_id: workspaceItemFolder.value || null,
      });
      workspaceItemForm.reset();
      workspaceItemRole.value = "context";
      workspaceMessage.textContent = "Workspace item added.";
      await refreshWorkspaceDetail();
    } catch (error) {
      workspaceMessage.textContent = `Could not add Workspace item: ${apiErrorMessage(error)}`;
    }
}

async function createWorkspaceFolderFromForm(event: SubmitEvent): Promise<void> {
  event.preventDefault();
  if (!selectedWorkspace) return;
  try {
    await createWorkspaceFolder(
      selectedWorkspace.id,
      workspaceFolderName.value.trim(),
      workspaceFolderParent.value || null,
    );
    workspaceFolderForm.reset();
    workspaceMessage.textContent = "Folder created.";
    await refreshWorkspaceDetail();
  } catch (error) {
    workspaceMessage.textContent = `Could not create folder: ${apiErrorMessage(error)}`;
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
workspaceSelect.addEventListener("change", () => {
  if (workspaceSelect.value) localStorage.setItem(storageKeys.workspaceId, workspaceSelect.value);
  else localStorage.removeItem(storageKeys.workspaceId);
  void refreshWorkspaceDetail();
});
byId<HTMLButtonElement>("refresh-workspaces").addEventListener("click", () => {
  void reloadWorkspaces().catch((error: unknown) => {
    workspaceMessage.textContent = `Could not load Workspaces: ${apiErrorMessage(error)}`;
  });
});
byId<HTMLFormElement>("workspace-create-form").addEventListener("submit", (event) => {
  void createWorkspaceFromForm(event as SubmitEvent);
});
workspaceItemForm.addEventListener("submit", (event) => {
  void createWorkspaceItemFromForm(event as SubmitEvent);
});
workspaceFolderForm.addEventListener("submit", (event) => {
  void createWorkspaceFolderFromForm(event as SubmitEvent);
});
saveOutputButton.addEventListener("click", () => void saveOutput());
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
