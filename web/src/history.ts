import {
  addGenerationHistoryToWorkspace,
  deleteGenerationHistoryItem,
  getGenerationHistory,
  getGenerationHistoryItem,
  getWorkspace,
  getWorkspaces,
} from "./api";
import type {
  GenerationHistoryDetail,
  GenerationHistorySummary,
  WorkspaceSummary,
} from "./types";

export function truncateHistoryPrompt(value: string): string {
  return value.length > 30 ? `${value.slice(0, 30)}...` : value;
}

interface HistoryElements {
  trigger: HTMLButtonElement;
  dialog: HTMLDialogElement;
  close: HTMLButtonElement;
  refresh: HTMLButtonElement;
  message: HTMLParagraphElement;
  list: HTMLUListElement;
}

interface HistoryCallbacks {
  selectedWorkspaceId: () => string;
  onOpen: (detail: GenerationHistoryDetail) => void;
  onWorkspaceSaved: (workspaceId: string) => Promise<void>;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function taskLabel(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function actionButton(label: string, className = "button button-quiet"): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  return button;
}

export function connectGenerationHistory(
  elements: HistoryElements,
  callbacks: HistoryCallbacks,
): void {
  const closeDialog = (): void => {
    if (elements.dialog.open) {
      if (typeof elements.dialog.close === "function") elements.dialog.close();
      else elements.dialog.removeAttribute("open");
    }
    elements.trigger.focus();
  };

  const loadHistory = async (): Promise<void> => {
    elements.refresh.disabled = true;
    elements.message.textContent = "Loading saved generations…";
    try {
      const archives = await getGenerationHistory();
      renderRows(archives);
      elements.message.textContent =
        archives.length === 0 ? "No saved generations yet." : `${archives.length} saved generations.`;
    } catch (error) {
      elements.list.replaceChildren();
      elements.message.textContent = `Could not load History: ${errorText(error)}`;
    } finally {
      elements.refresh.disabled = false;
    }
  };

  const showWorkspaceEditor = async (
    row: HTMLLIElement,
    summary: GenerationHistorySummary,
  ): Promise<void> => {
    row.querySelector(".history-workspace-editor")?.remove();
    const editor = document.createElement("form");
    editor.className = "history-workspace-editor";
    const workspaces = await getWorkspaces();
    if (workspaces.length === 0) {
      elements.message.textContent = "Create a Workspace before saving this generation.";
      return;
    }
    const workspaceLabel = document.createElement("label");
    workspaceLabel.textContent = "Workspace";
    const workspaceSelect = document.createElement("select");
    workspaceSelect.required = true;
    const currentWorkspaceId = callbacks.selectedWorkspaceId();
    const initialWorkspaceId = workspaces.some((item) => item.id === currentWorkspaceId)
      ? currentWorkspaceId
      : workspaces[0].id;
    for (const workspace of workspaces) {
      const option = document.createElement("option");
      option.value = workspace.id;
      option.textContent = workspace.name;
      workspaceSelect.append(option);
    }
    workspaceSelect.value = initialWorkspaceId;
    workspaceLabel.append(workspaceSelect);

    const titleLabel = document.createElement("label");
    titleLabel.textContent = "Title";
    const titleInput = document.createElement("input");
    titleInput.required = true;
    titleInput.maxLength = 300;
    titleInput.value = `${taskLabel(summary.task)} ${summary.request_id.slice(0, 8)}`;
    titleLabel.append(titleInput);

    const folderLabel = document.createElement("label");
    folderLabel.textContent = "Folder";
    const folderSelect = document.createElement("select");
    const rootOption = document.createElement("option");
    rootOption.value = "";
    rootOption.textContent = "Workspace root";
    folderSelect.append(rootOption);
    folderLabel.append(folderSelect);

    const feedback = document.createElement("p");
    feedback.className = "field-help";
    feedback.setAttribute("role", "status");
    const saveButton = actionButton("Add to Workspace", "button button-primary");
    saveButton.type = "submit";
    const cancelButton = actionButton("Cancel");
    cancelButton.addEventListener("click", () => editor.remove());
    const controls = document.createElement("div");
    controls.className = "inline-controls";
    controls.append(saveButton, cancelButton);
    editor.append(workspaceLabel, titleLabel, folderLabel, controls, feedback);
    row.append(editor);

    let activeWorkspace: WorkspaceSummary | undefined;
    const refreshFolders = async (): Promise<void> => {
      folderSelect.replaceChildren(rootOption);
      try {
        const workspace = await getWorkspace(workspaceSelect.value);
        activeWorkspace = workspaces.find((item) => item.id === workspace.id);
        for (const folder of workspace.folders) {
          const option = document.createElement("option");
          option.value = folder.id;
          option.textContent = folder.name;
          folderSelect.append(option);
        }
      } catch (error) {
        activeWorkspace = undefined;
        feedback.textContent = `Could not load Workspace folders: ${errorText(error)}`;
      }
    };
    workspaceSelect.addEventListener("change", () => void refreshFolders());
    await refreshFolders();

    editor.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!activeWorkspace || !titleInput.value.trim()) return;
      saveButton.disabled = true;
      try {
        await addGenerationHistoryToWorkspace(summary.archive_id, {
          workspace_id: activeWorkspace.id,
          title: titleInput.value.trim(),
          folder_id: folderSelect.value || null,
        });
        feedback.textContent = "Generation copied to Workspace History.";
        await callbacks.onWorkspaceSaved(activeWorkspace.id);
      } catch (error) {
        feedback.textContent = `Could not save to Workspace: ${errorText(error)}`;
      } finally {
        saveButton.disabled = false;
      }
    });
    titleInput.focus();
  };

  const renderRows = (archives: GenerationHistorySummary[]): void => {
    elements.list.replaceChildren();
    for (const summary of archives) {
      const row = document.createElement("li");
      row.className = "history-entry";
      row.dataset.archiveId = summary.archive_id;
      const info = document.createElement("div");
      info.className = "history-entry-info";
      const prompt = document.createElement("strong");
      prompt.textContent = truncateHistoryPrompt(summary.instruction);
      prompt.title = summary.instruction;
      const metadata = document.createElement("p");
      metadata.className = "field-help";
      const created = new Date(summary.created_at);
      const createdLabel = Number.isNaN(created.valueOf())
        ? summary.created_at
        : created.toLocaleString();
      metadata.textContent =
        `${taskLabel(summary.task)} · ${taskLabel(summary.profile)} · ` +
        `${taskLabel(summary.status)} · ${createdLabel}`;
      info.append(prompt, metadata);

      const actions = document.createElement("div");
      actions.className = "inline-controls history-actions";
      const open = actionButton("Open");
      open.addEventListener("click", async () => {
        open.disabled = true;
        elements.message.textContent = "Opening saved generation…";
        try {
          const detail = await getGenerationHistoryItem(summary.archive_id);
          closeDialog();
          callbacks.onOpen(detail);
        } catch (error) {
          elements.message.textContent = `Could not open generation: ${errorText(error)}`;
          open.disabled = false;
        }
      });
      actions.append(open);
      if (summary.status === "completed" && summary.has_output) {
        const add = actionButton("Add to Workspace");
        add.addEventListener("click", () => {
          void showWorkspaceEditor(row, summary).catch((error: unknown) => {
            elements.message.textContent = `Could not load Workspaces: ${errorText(error)}`;
          });
        });
        actions.append(add);
      }
      const remove = actionButton("Delete");
      remove.addEventListener("click", async () => {
        const confirmed = window.confirm(
          "Delete this generation from History?\n\n" +
            "This permanently deletes the saved generation archive and its diagnostics.",
        );
        if (!confirmed) return;
        remove.disabled = true;
        try {
          await deleteGenerationHistoryItem(summary.archive_id);
          row.remove();
          elements.message.textContent = "Generation deleted from History.";
        } catch (error) {
          elements.message.textContent = `Could not delete generation: ${errorText(error)}`;
          remove.disabled = false;
        }
      });
      actions.append(remove);
      row.append(info, actions);
      elements.list.append(row);
    }
  };

  elements.trigger.addEventListener("click", () => {
    if (typeof elements.dialog.showModal === "function") elements.dialog.showModal();
    else elements.dialog.setAttribute("open", "");
    elements.close.focus();
    void loadHistory();
  });
  elements.close.addEventListener("click", closeDialog);
  elements.refresh.addEventListener("click", () => void loadHistory());
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) closeDialog();
  });
  elements.dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeDialog();
    }
  });
}
