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

marked.use(markedKatex(katexOptions));
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
