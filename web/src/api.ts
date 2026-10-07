import type {
  GenerationRequest,
  GenerationResult,
  HealthResponse,
  ProfileSummary,
  ProgressEvent,
  SseMessage,
  TaskType,
  WorkspaceDetail,
  WorkspaceItem,
  WorkspaceItemRole,
  WorkspaceSummary,
} from "./types";
import { SseParser } from "./sse";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorMessage(response: Response): Promise<string> {
  const body: unknown = await response.json().catch(() => null);
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map(String).join("; ");
  }
  return `Lecture SLM API returned HTTP ${response.status}.`;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return (await response.json()) as T;
}

async function sendJson<T>(path: string, method: "POST" | "PATCH", body: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return (await response.json()) as T;
}

async function sendNoContent(path: string, method: "DELETE"): Promise<void> {
  const response = await fetch(path, { method, headers: { Accept: "application/json" } });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
}

export function getWorkspaces(): Promise<WorkspaceSummary[]> {
  return getJson<WorkspaceSummary[]>("/api/workspaces");
}

export function getWorkspace(workspaceId: string): Promise<WorkspaceDetail> {
  return getJson<WorkspaceDetail>(`/api/workspaces/${encodeURIComponent(workspaceId)}`);
}

export function createWorkspace(name: string): Promise<WorkspaceSummary> {
  return sendJson<WorkspaceSummary>("/api/workspaces", "POST", { name });
}

export function createWorkspaceItem(
  workspaceId: string,
  item: {
    title: string;
    role: WorkspaceItemRole;
    content: string;
    pinned: boolean;
    folder_id: string | null;
  },
): Promise<WorkspaceItem> {
  return sendJson<WorkspaceItem>(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/items`,
    "POST",
    item,
  );
}

export function createWorkspaceFolder(
  workspaceId: string,
  name: string,
  parentId: string | null,
): Promise<{ id: string; name: string }> {
  return sendJson(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/folders`,
    "POST",
    { name, parent_id: parentId },
  );
}

export function deleteWorkspaceFolder(workspaceId: string, folderId: string): Promise<void> {
  return sendNoContent(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/folders/${encodeURIComponent(folderId)}`,
    "DELETE",
  );
}

export function updateWorkspaceItem(
  workspaceId: string,
  itemId: string,
  changes: { role?: WorkspaceItemRole; pinned?: boolean },
): Promise<WorkspaceItem> {
  return sendJson<WorkspaceItem>(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/items/${encodeURIComponent(itemId)}`,
    "PATCH",
    changes,
  );
}

export function deleteWorkspaceItem(workspaceId: string, itemId: string): Promise<void> {
  return sendNoContent(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/items/${encodeURIComponent(itemId)}`,
    "DELETE",
  );
}

export function saveOutputToWorkspace(
  workspaceId: string,
  requestId: string,
  title: string,
): Promise<WorkspaceItem> {
  return sendJson<WorkspaceItem>(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/history`,
    "POST",
    { request_id: requestId, title },
  );
}

export function getHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/health");
}

export async function getTasks(): Promise<TaskType[]> {
  return (await getJson<{ tasks: TaskType[] }>("/api/tasks")).tasks;
}

export async function getProfiles(): Promise<ProfileSummary[]> {
  return (await getJson<{ profiles: ProfileSummary[] }>("/api/profiles")).profiles;
}

export function getRun(requestId: string): Promise<GenerationResult> {
  return getJson<GenerationResult>(`/api/runs/${encodeURIComponent(requestId)}`);
}

function parseData<T>(message: SseMessage): T {
  try {
    return JSON.parse(message.data) as T;
  } catch (error) {
    throw new ApiError(
      `Lecture SLM sent an invalid ${message.event} event: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

export async function generateStream(
  request: GenerationRequest,
  onProgress: (progress: ProgressEvent) => void,
): Promise<GenerationResult> {
  const response = await fetch("/api/generate/stream", {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  if (!response.body) throw new ApiError("The streaming response did not include a body.");

  let result: GenerationResult | undefined;
  const parser = new SseParser();
  const handleMessage = (message: SseMessage): void => {
    if (message.event === "progress") {
      onProgress(parseData<ProgressEvent>(message));
    } else if (message.event === "result") {
      result = parseData<GenerationResult>(message);
    } else if (message.event === "error") {
      const detail = parseData<{ detail?: string }>(message).detail;
      throw new ApiError(detail ?? "The generation stream failed on the server.");
    }
  };

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }), handleMessage);
    }
    parser.push(decoder.decode(), handleMessage);
    parser.finish(handleMessage);
  } finally {
    reader.releaseLock();
  }

  if (!result) throw new ApiError("The generation stream ended before a result was received.");
  return result;
}
