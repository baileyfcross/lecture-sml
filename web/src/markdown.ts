import DOMPurify from "dompurify";
import katex from "katex";
import { marked, type MarkedExtension, type Tokens } from "marked";
import markedKatex from "marked-katex-extension";
import "katex/dist/katex.min.css";

const katexOptions = {
  output: "htmlAndMathml" as const,
  throwOnError: false,
  trust: false,
};

interface DollarMathMatch {
  raw: string;
  text: string;
}

function isEscaped(source: string, index: number): boolean {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && source[cursor] === "\\"; cursor -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function hasMathStartBoundary(source: string, index: number): boolean {
  return index === 0 || /\s/u.test(source[index - 1] ?? "");
}

function hasMathEndBoundary(source: string, index: number): boolean {
  if (index === source.length) return true;
  return /[\s?!.,:;()[\]{}…？！。，：；、]/u.test(source[index] ?? "");
}

function parseDollarMath(source: string, delimiter: string): DollarMathMatch | null {
  if (!source.startsWith(delimiter) || source.startsWith(`${delimiter}$`)) return null;
  const contentStart = delimiter.length;

  for (let index = contentStart; index < source.length; index += 1) {
    const character = source[index];
    if (character === "\n" || character === "\r") return null;
    if (character !== "$" || isEscaped(source, index)) continue;

    let delimiterLength = 1;
    while (source[index + delimiterLength] === "$") delimiterLength += 1;
    if (delimiterLength !== delimiter.length) return null;

    const end = index + delimiterLength;
    const text = source.slice(contentStart, index).trim();
    if (!text || !hasMathEndBoundary(source, end)) return null;
    return { raw: source.slice(0, end), text };
  }
  return null;
}

function dollarMathExtension(
  name: string,
  delimiter: string,
  displayMode: boolean,
): MarkedExtension {
  return {
    extensions: [
      {
        name,
        level: "inline",
        start(source) {
          let index = 0;
          while (index < source.length) {
            const candidate = source.indexOf("$", index);
            if (candidate < 0) return undefined;
            if (
              !isEscaped(source, candidate)
              && hasMathStartBoundary(source, candidate)
              && source.startsWith(delimiter, candidate)
              && !source.startsWith(`${delimiter}$`, candidate)
              && parseDollarMath(source.slice(candidate), delimiter)
            ) {
              return candidate;
            }
            index = candidate + 1;
          }
          return undefined;
        },
        tokenizer(source) {
          const match = parseDollarMath(source, delimiter);
          if (!match) return undefined;
          return {
            type: name,
            raw: match.raw,
            text: match.text,
          };
        },
        renderer(token: Tokens.Generic) {
          if (typeof token.text !== "string") return token.raw;
          return katex.renderToString(token.text, { ...katexOptions, displayMode });
        },
      },
    ],
  };
}

function mathExtension(
  name: string,
  level: "block" | "inline",
  pattern: RegExp,
  displayMode: boolean,
): MarkedExtension {
  return {
    extensions: [
      {
        name,
        level,
        start(source) {
          const index = source.indexOf(level === "block" ? "\\[" : "\\(");
          return index < 0 ? undefined : index;
        },
        tokenizer(source) {
          const match = pattern.exec(source);
          if (!match) return undefined;
          return {
            type: name,
            raw: match[0],
            text: match[1].trim(),
          };
        },
        renderer(token: Tokens.Generic) {
          if (typeof token.text !== "string") return token.raw;
          return katex.renderToString(token.text, {
            ...katexOptions,
            displayMode,
          });
        },
      },
    ],
  };
}

const katexBlockExtension = markedKatex(katexOptions).extensions?.find(
  (extension) => extension.name === "blockKatex",
);
if (!katexBlockExtension) {
  throw new Error("The configured Marked KaTeX extension does not provide block math.");
}

// The plugin's inline rule rejects closing delimiters followed by ')', so inline math is scanned here.
marked.use(
  { extensions: [katexBlockExtension] },
  dollarMathExtension("inlineDollarMath", "$", false),
  dollarMathExtension("inlineDisplayDollarMath", "$$", true),
);
marked.use(
  mathExtension(
    "displayMathBracket",
    "block",
    /^\\\[(?:[ \t]*\r?\n)?([\s\S]+?)\\\](?:[ \t]*\r?\n|$)/,
    true,
  ),
  mathExtension("inlineMathParen", "inline", /^\\\(([^\n]+?)\\\)/, false),
);

export function renderArtifactMarkdown(markdown: string): string {
  const html = marked.parse(markdown, { async: false });
  return DOMPurify.sanitize(html);
}
