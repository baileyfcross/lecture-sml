import type { SseMessage } from "./types";

export class SseParser {
  private buffer = "";
  private eventName = "";
  private dataLines: string[] = [];

  push(chunk: string, onMessage: (message: SseMessage) => void): void {
    this.buffer += chunk;
    let offset = 0;

    for (let index = 0; index < this.buffer.length; index += 1) {
      const character = this.buffer[index];
      if (character !== "\r" && character !== "\n") continue;
      if (character === "\r" && index === this.buffer.length - 1) break;

      const line = this.buffer.slice(offset, index);
      if (character === "\r" && this.buffer[index + 1] === "\n") index += 1;
      offset = index + 1;
      this.consumeLine(line, onMessage);
    }

    this.buffer = this.buffer.slice(offset);
  }

  finish(onMessage: (message: SseMessage) => void): void {
    if (this.buffer.length > 0) this.consumeLine(this.buffer, onMessage);
    this.buffer = "";
    this.dispatch(onMessage);
  }

  private consumeLine(line: string, onMessage: (message: SseMessage) => void): void {
    if (line === "") {
      this.dispatch(onMessage);
      return;
    }
    if (line.startsWith(":")) return;

    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") this.eventName = value;
    if (field === "data") this.dataLines.push(value);
  }

  private dispatch(onMessage: (message: SseMessage) => void): void {
    if (this.dataLines.length > 0) {
      onMessage({
        event: this.eventName || "message",
        data: this.dataLines.join("\n"),
      });
    }
    this.eventName = "";
    this.dataLines = [];
  }
}
