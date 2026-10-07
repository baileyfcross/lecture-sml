// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import { renderSourceCoverage } from "../src/sourceCoverage";
import type { SourceCoverage } from "../src/types";

describe("renderSourceCoverage", () => {
  it("shows partial coverage with its explanatory note", () => {
    const element = document.createElement("aside");
    const coverage: SourceCoverage = {
      status: "partial",
      supported_topics: ["quantum annealing"],
      unsupported_requested_topics: ["error correction"],
      scope_note:
        "The available materials support quantum annealing, but not a complete treatment of error correction.",
    };

    renderSourceCoverage(element, coverage);

    expect(element.hidden).toBe(false);
    expect(element.dataset.status).toBe("partial");
    expect(element.querySelector("strong")?.textContent).toBe("Source coverage: Partial");
    expect(element.querySelector("p")?.textContent).toBe(coverage.scope_note);
  });

  it("does not display a notice when coverage was not assessed", () => {
    const element = document.createElement("aside");

    renderSourceCoverage(element, null);

    expect(element.hidden).toBe(true);
    expect(element.childElementCount).toBe(0);
    expect(element.dataset.status).toBeUndefined();
  });

  it("builds a grounded explanation when a partial assessment has no note", () => {
    const element = document.createElement("aside");
    const coverage: SourceCoverage = {
      status: "partial",
      supported_topics: ["quantum annealing"],
      unsupported_requested_topics: ["error correction"],
      scope_note: null,
    };

    renderSourceCoverage(element, coverage);

    expect(element.querySelector("p")?.textContent).toBe(
      "The available materials support quantum annealing, but not error correction. " +
        "This response is narrowed to the supported material.",
    );
  });
});
