// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import { renderArtifactMarkdown } from "../src/markdown";

function renderContainer(markdown: string): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = renderArtifactMarkdown(markdown);
  return container;
}

function expectMathExpressions(container: HTMLElement, expressions: string[]): void {
  const mathExpressions = container.querySelectorAll(".katex-mathml math");
  expect(mathExpressions).toHaveLength(expressions.length);
  expressions.forEach((expression, index) => {
    expect(mathExpressions[index]?.textContent).toContain(expression);
  });
}

function expectValidMath(container: HTMLElement, count: number): void {
  expect(container.querySelectorAll(".katex")).toHaveLength(count);
  expect(container.querySelectorAll(".katex-error")).toHaveLength(0);
}

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

  it("renders separate inline math expressions in the live modular-arithmetic list item", () => {
    const container = renderContainer(
      "- **Modular arithmetic systems** (like $\\mathbb{Z}_n$) provide concrete finite examples where groups, rings, and fields emerge depending on $n$.",
    );
    const item = container.querySelector("li");

    expect(container.querySelectorAll("li")).toHaveLength(1);
    expect(item?.querySelector("strong")?.textContent).toBe("Modular arithmetic systems");
    expect(item?.textContent).toContain("(like");
    expect(item?.textContent).toContain(") provide concrete finite examples where groups, rings, and fields emerge depending on");
    expectMathExpressions(container, ["\\mathbb{Z}_n", "n"]);
    expectValidMath(container, 2);
    expect(item?.textContent).not.toContain("$");
  });

  it.each([
    ["two expressions", "$A$ and $B$", ["A", "B"], " and "],
    ["three expressions", "$A$, $B$, and $C$", ["A", "B", "C"], ", "],
    ["parenthesized expressions", "$(a+b)$ then $(c+d)$", ["(a+b)", "(c+d)"], " then "],
    ["mixed inline text", "The result is $x$ and the next value is $y$.", ["x", "y"], " and the next value is "],
    ["parenthesized math", "(like $\\mathbb{Z}_n$)", ["\\mathbb{Z}_n"], "(like"],
  ])("tokenizes %s independently", (_label, markdown, expressions, prose) => {
    const container = renderContainer(markdown);

    expectMathExpressions(container, expressions);
    expectValidMath(container, expressions.length);
    expect(container.textContent).toContain(prose);
  });

  it("renders common math commands as independent expressions", () => {
    const expressions = [
      "\\mathbb{Z}_n",
      "\\forall x \\, P(x)",
      "\\exists x \\, P(x)",
      "P(x) \\rightarrow Q(x)",
      "\\frac{a}{b}",
      "x_1 + x_2",
    ];
    const container = renderContainer(expressions.map((expression) => `$${expression}$`).join(" and "));

    expectMathExpressions(container, expressions);
    expectValidMath(container, expressions.length);
  });

  it("preserves escaped-dollar and currency text before later math", () => {
    const cases = [
      String.raw`The price is \$500 and the variable is $x$.`,
      "The computer costs $500 and the variable is $x$.",
      "The computer costs $500, the monitor costs $200, and the variable is $x$.",
    ];

    for (const markdown of cases) {
      const container = renderContainer(markdown);
      expectMathExpressions(container, ["x"]);
      expectValidMath(container, 1);
      expect(container.textContent).toContain("$500");
      expect(container.textContent).not.toContain("$x$");
    }
  });

  it("does not interpret math inside inline or fenced code", () => {
    const container = renderContainer("`$A$ and $B$`\n\n```text\n$A$ and $B$\n```");

    expect(container.querySelectorAll(".katex")).toHaveLength(0);
    expect(container.querySelector("code")?.textContent).toContain("$A$ and $B$");
    expect(container.textContent).toContain("$A$ and $B$");
  });

  it("renders display math and parenthesis delimiters without errors", () => {
    const container = renderContainer(
      "$$\n\\forall x \\, P(x)\n$$\n\n\\[\nP(x)\n\\]\n\n\\(P(x)\\)",
    );

    expectValidMath(container, 3);
    expect(container.querySelectorAll(".katex-display")).toHaveLength(2);
  });
});
