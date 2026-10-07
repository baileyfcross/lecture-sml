// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import { renderArtifactMarkdown } from "../src/markdown";

describe("renderArtifactMarkdown", () => {
  it("renders inline and display math with accessible MathML", () => {
    const html = renderArtifactMarkdown(
      "A universal statement is $\\forall x \\, P(x)$.\n\n$$\n\\exists x \\, Q(x)\n$$",
    );

    expect(html).toContain("katex");
    expect(html).toContain("katex-display");
    expect(html).toContain("<math");
    expect(html).toContain("A universal statement is");
  });

  it("renders parenthesis and bracket delimiters", () => {
    const html = renderArtifactMarkdown(
      "Inline \\(P(x)\\).\n\n\\[\n\\forall x \\, P(x)\n\\]",
    );

    expect(html.match(/class="katex"/g)).toHaveLength(2);
    expect(html).toContain("katex-display");
    expect(html).toContain("<math");
  });

  it("renders common predicate, Boolean-algebra, proof, and arithmetic notation", () => {
    const formulas = [
      "$\\forall x \\, P(x)$",
      "$\\exists x \\, P(x)$",
      "$\\neg \\forall x \\, P(x) \\equiv \\exists x \\, \\neg P(x)$",
      "$P(x) \\rightarrow Q(x)$",
      "$A + A' = 1$",
      "$A \\cdot A' = 0$",
      "$(A \\cdot B)' = A' + B'$",
      "$n = 2k + 1$",
      "$2^4 = 16$",
      "$x_1 + x_2$",
      "$\\frac{a}{b}$",
    ];
    const html = renderArtifactMarkdown(formulas.join("\n\n"));

    expect(html.match(/class="katex"/g)).toHaveLength(formulas.length);
    expect(html).not.toContain("katex-error");
  });

  it("leaves math-like text literal inside fenced and inline code", () => {
    const html = renderArtifactMarkdown(
      "```text\n$P(x)$\n\\(Q(x)\\)\n\\[\nR(x)\n\\]\n```\n\n`$P(x)$`",
    );

    expect(html).toContain("<code>$P(x)$");
    expect(html).toContain("\\(Q(x)\\)");
    expect(html).toContain("\\[");
    expect(html).not.toContain("katex");
  });

  it("does not treat ordinary dollar amounts as math", () => {
    const html = renderArtifactMarkdown(
      "The computer costs $500. The costs are $500 and $700.",
    );

    expect(html).toContain("$500");
    expect(html).toContain("$700");
    expect(html).not.toContain("katex");
  });

  it("keeps malformed LaTeX and surrounding Markdown visible", () => {
    const html = renderArtifactMarkdown("Before $\\frac{a$ after.");

    expect(html).toContain("Before");
    expect(html).toContain("after.");
    expect(html).toContain("katex-error");
  });

  it("removes executable markup adjacent to rendered math", () => {
    const html = renderArtifactMarkdown(
      '$P(x)$ <script>alert(1)</script> <img src="x" onerror="alert(1)"> ' +
        '<a href="javascript:alert(1)">unsafe</a> <div onclick="alert(1)">test</div>',
    );
    const container = document.createElement("div");
    container.innerHTML = html;

    expect(container.querySelector(".katex")).not.toBeNull();
    expect(container.querySelector("math")).not.toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")?.hasAttribute("onerror")).toBe(false);
    expect(container.querySelector("[onclick], [onerror], [onload]")).toBeNull();
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
  });

  it("preserves headings, lists, emphasis, links, quotes, rules, and code", () => {
    const html = renderArtifactMarkdown(
      "# Heading\n\n- Item 1\n- Item 2\n\n**bold** *italic* [link](https://example.com)\n\n" +
        "> quote\n\n---\n\n`code` and $P(x)$",
    );

    expect(html).toContain("<h1>Heading</h1>");
    expect(html).toContain("<ul>");
    expect(html).toContain("<strong>bold</strong>");
    expect(html).toContain("<em>italic</em>");
    expect(html).toContain('href="https://example.com"');
    expect(html).toContain("<blockquote>");
    expect(html).toContain("<hr>");
    expect(html).toContain("<code>code</code>");
    expect(html).toContain("katex");
  });
});
