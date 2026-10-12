// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { connectGenerationHistory, truncateHistoryPrompt } from "../src/history";
import type {
  GenerationHistoryDetail,
  GenerationHistorySummary,
  WorkspaceDetail,
} from "../src/types";

const api = vi.hoisted(() => ({
  addGenerationHistoryToWorkspace: vi.fn(),
  deleteGenerationHistoryItem: vi.fn(),
  getGenerationHistory: vi.fn(),
  getGenerationHistoryItem: vi.fn(),
  getWorkspace: vi.fn(),
  getWorkspaces: vi.fn(),
}));

vi.mock("../src/api", () => api);

const completedSummary: GenerationHistorySummary = {
  archive_id: "20261012T010203Z-request-1-a1b2c3",
  request_id: "12345678-abcd",
  instruction: "1234567890123456789012345678901",
  task: "lecture",
  profile: "deep",
  status: "completed",
  created_at: "2026-10-12T01:02:03Z",
  has_output: true,
  workspace_id: null,
};

const failedSummary: GenerationHistorySummary = {
  ...completedSummary,
  archive_id: "20261012T010203Z-request-2-a1b2c3",
  request_id: "request-2",
  status: "failed",
  has_output: false,
};

const completedDetail: GenerationHistoryDetail = {
  request_id: "12345678-abcd",
  status: "completed",
  task: "lecture",
  profile: "deep",
  output: "Archived approved output",
  errors: [],
  source_coverage: null,
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
    total_seconds: 2,
    planner_seconds: null,
    writer_seconds: 2,
    initial_review_seconds: null,
    revision_seconds: null,
    final_review_seconds: null,
    selected_contexts: {},
    estimated_input_tokens: {},
  },
  saved_run: "artifacts/generations/archive-1",
  archive_id: completedSummary.archive_id,
  instruction: completedSummary.instruction,
  created_at: completedSummary.created_at,
};

function mountHistory(selectedWorkspaceId = ""): void {
  document.body.innerHTML = `
    <button id="trigger">History</button>
    <dialog id="dialog">
      <button id="close">Close</button>
      <button id="refresh">Refresh</button>
      <p id="message"></p>
      <ul id="list"></ul>
    </dialog>`;
  connectGenerationHistory(
    {
      trigger: document.querySelector<HTMLButtonElement>("#trigger")!,
      dialog: document.querySelector<HTMLDialogElement>("#dialog")!,
      close: document.querySelector<HTMLButtonElement>("#close")!,
      refresh: document.querySelector<HTMLButtonElement>("#refresh")!,
      message: document.querySelector<HTMLParagraphElement>("#message")!,
      list: document.querySelector<HTMLUListElement>("#list")!,
    },
    {
      selectedWorkspaceId: () => selectedWorkspaceId,
      onOpen: onOpenMock,
      onWorkspaceSaved: onWorkspaceSavedMock,
    },
  );
}

const onOpenMock = vi.fn();
const onWorkspaceSavedMock = vi.fn().mockResolvedValue(undefined);

afterEach(() => {
  document.body.replaceChildren();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("Generation History UI", () => {
  it("truncates only prompts longer than 30 characters using three periods", () => {
    expect(truncateHistoryPrompt("123456789012345678901234567890")).toBe(
      "123456789012345678901234567890",
    );
    expect(truncateHistoryPrompt("1234567890123456789012345678901")).toBe(
      "123456789012345678901234567890...",
    );
  });

  it("loads History lazily and opens a saved result without generating", async () => {
    api.getGenerationHistory.mockResolvedValue([completedSummary]);
    api.getGenerationHistoryItem.mockResolvedValue(completedDetail);
    mountHistory();
    expect(api.getGenerationHistory).not.toHaveBeenCalled();

    document.querySelector<HTMLButtonElement>("#trigger")!.click();
    await vi.waitFor(() => expect(api.getGenerationHistory).toHaveBeenCalledOnce());
    const prompt = document.querySelector(".history-entry-info strong")!;
    expect(prompt.textContent).toBe("123456789012345678901234567890...");
    expect(prompt.getAttribute("title")).toBe(completedSummary.instruction);

    document.querySelector<HTMLButtonElement>(".history-entry button")!.click();
    await vi.waitFor(() => expect(onOpenMock).toHaveBeenCalledWith(completedDetail));
    expect(api.getGenerationHistoryItem).toHaveBeenCalledWith(completedSummary.archive_id);
    expect(document.querySelector<HTMLDialogElement>("#dialog")!.open).toBe(false);
  });

  it("confirms deletion and omits Workspace save for failed entries", async () => {
    api.getGenerationHistory.mockResolvedValue([failedSummary]);
    api.deleteGenerationHistoryItem.mockResolvedValue(undefined);
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(true));
    mountHistory();
    document.querySelector<HTMLButtonElement>("#trigger")!.click();
    await vi.waitFor(() => expect(document.querySelector(".history-entry")).not.toBeNull());

    expect(document.querySelector(".history-entry button")?.textContent).toBe("Open");
    expect(
      [...document.querySelectorAll(".history-entry button")].some(
        (button) => button.textContent === "Add to Workspace",
      ),
    ).toBe(false);
    document.querySelector<HTMLButtonElement>(".history-entry button:last-child")!.click();
    await vi.waitFor(() =>
      expect(api.deleteGenerationHistoryItem).toHaveBeenCalledWith(failedSummary.archive_id),
    );
    expect(window.confirm).toHaveBeenCalledWith(
      "Delete this generation from History?\n\n" +
        "This permanently deletes the saved generation archive and its diagnostics.",
    );
    await vi.waitFor(() => expect(document.querySelector(".history-entry")).toBeNull());
  });

  it("copies a completed approved archive to the selected Workspace History folder", async () => {
    const workspaceDetail: WorkspaceDetail = {
      id: "workspace-1",
      name: "CSC 220",
      description: null,
      created_at: "2026-10-12T00:00:00Z",
      updated_at: "2026-10-12T00:00:00Z",
      folders: [
        {
          id: "folder-1",
          workspace_id: "workspace-1",
          parent_id: null,
          name: "Lectures",
        },
      ],
      items: [],
    };
    api.getGenerationHistory.mockResolvedValue([completedSummary]);
    api.getWorkspaces.mockResolvedValue([{ id: "workspace-1", name: "CSC 220" }]);
    api.getWorkspace.mockResolvedValue(workspaceDetail);
    api.addGenerationHistoryToWorkspace.mockResolvedValue({ role: "history" });
    mountHistory("workspace-1");
    document.querySelector<HTMLButtonElement>("#trigger")!.click();
    await vi.waitFor(() => expect(document.querySelector(".history-entry")).not.toBeNull());
    const buttons = [...document.querySelectorAll<HTMLButtonElement>(".history-entry button")];
    buttons.find((button) => button.textContent === "Add to Workspace")!.click();
    await vi.waitFor(() => expect(document.querySelector(".history-workspace-editor")).not.toBeNull());

    const form = document.querySelector<HTMLFormElement>(".history-workspace-editor")!;
    const selects = form.querySelectorAll("select");
    await vi.waitFor(() => expect(selects[1].querySelector('option[value="folder-1"]')).not.toBeNull());
    selects[1].value = "folder-1";
    form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() =>
      expect(api.addGenerationHistoryToWorkspace).toHaveBeenCalledWith(
        completedSummary.archive_id,
        {
          workspace_id: "workspace-1",
          title: "Lecture 12345678",
          folder_id: "folder-1",
        },
      ),
    );
    expect(onWorkspaceSavedMock).toHaveBeenCalledWith("workspace-1");
  });
});
