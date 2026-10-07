import type { SourceCoverage } from "./types";

function coverageNote(coverage: SourceCoverage): string {
  if (coverage.scope_note) return coverage.scope_note;
  if (coverage.status === "sufficient") {
    return "The available sources support the requested scope.";
  }
  if (coverage.status === "insufficient") {
    return "The available sources do not support enough of the requested topic for a substantive artifact.";
  }

  const supported = coverage.supported_topics.join(", ");
  const unsupported = coverage.unsupported_requested_topics.join(", ");
  if (supported && unsupported) {
    return `The available materials support ${supported}, but not ${unsupported}. This response is narrowed to the supported material.`;
  }
  return "This response is narrowed to the topics supported by the available materials.";
}

export function renderSourceCoverage(
  element: HTMLElement,
  coverage: SourceCoverage | null,
): void {
  element.replaceChildren();
  element.hidden = coverage === null;
  if (coverage === null) {
    delete element.dataset.status;
    return;
  }

  element.dataset.status = coverage.status;
  const heading = document.createElement("strong");
  heading.textContent = `Source coverage: ${coverage.status[0].toUpperCase()}${coverage.status.slice(1)}`;
  element.append(heading);

  const note = document.createElement("p");
  note.textContent = coverageNote(coverage);
  element.append(note);
}
