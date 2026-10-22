// Source rendering and exact-quote navigation for immutable captures.
// Navigation adapts Magellan's contentScript_highlightElementsById and
// navigateToMatchOnPage. It uses a local root, not extension injection.
// https://github.com/magellan-extension/magellan
//
// MIT License
//
// Copyright (c) 2025 magellan-extension
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import createDOMPurify from "dompurify";
import type { WindowLike } from "dompurify";
import type { SourceDocument } from "./research-client";
import { safeSourceUrl } from "../../../needle-harness/lib/agent/types";

export type QuoteResolution =
  | { status: "resolved"; start: number; end: number }
  | { status: "ambiguous"; count: number }
  | { status: "unresolved" };

// PDF text items and adjacent HTML table cells can omit intervening spaces.
// Only whitespace changes for matching. Marks retain the captured characters.
function indexedText(text: string): { value: string; offsets: number[] } {
  const offsets: number[] = [];
  let value = "";
  for (let i = 0; i < text.length; i++) {
    if (/\s/u.test(text[i])) continue;
    value += text[i];
    offsets.push(i);
  }
  return { value, offsets };
}

type QuoteMatch = { start: number; end: number };

function quoteMatches(text: string, needle: string): QuoteMatch[] {
  const indexed = indexedText(text);
  const matches: QuoteMatch[] = [];
  for (let from = 0; from < indexed.value.length;) {
    const at = indexed.value.indexOf(needle, from);
    if (at === -1) break;
    matches.push({ start: indexed.offsets[at], end: indexed.offsets[at + needle.length - 1] + 1 });
    from = at + 1;
  }
  return matches;
}

function coordinateMatch(matches: QuoteMatch[], location: SourceDocument["location"] | undefined): QuoteMatch | undefined {
  if (location?.offset === undefined) return undefined;
  // Coordinates win only when the quote reproduces at those coordinates.
  const coordinate = matches.filter((m) => m.start === location.offset && (location.end === undefined || m.end === location.end));
  return coordinate.length === 1 ? coordinate[0] : undefined;
}

function contextualMatch(text: string, matches: QuoteMatch[], location: SourceDocument["location"] | undefined): QuoteResolution | undefined {
  const prefix = location?.prefix?.replace(/\s/gu, "");
  const suffix = location?.suffix?.replace(/\s/gu, "");
  if (!prefix && !suffix) return undefined;
  const contextual = matches.filter((m) =>
    (!prefix || text.slice(0, m.start).replace(/\s/gu, "").endsWith(prefix)) &&
    (!suffix || text.slice(m.end).replace(/\s/gu, "").startsWith(suffix)),
  );
  if (contextual.length === 1) return { status: "resolved", ...contextual[0] };
  return contextual.length === 0 ? { status: "unresolved" } : { status: "ambiguous", count: contextual.length };
}


export function resolveQuote(text: string, quote: string, location?: SourceDocument["location"]): QuoteResolution {
  const needle = quote.replace(/\s/gu, "");
  if (!needle) return { status: "unresolved" };
  const matches = quoteMatches(text, needle);
  if (matches.length === 0) return { status: "unresolved" };
  const coordinate = coordinateMatch(matches, location);
  if (coordinate) return { status: "resolved", ...coordinate };
  const contextual = contextualMatch(text, matches, location);
  if (contextual) return contextual;
  return matches.length === 1 ? { status: "resolved", ...matches[0] } : { status: "ambiguous", count: matches.length };
}

// ponytail: one allowlist owns sanitization (keep tag/attribute checks together, not scattered callers).
export function sanitizeSourceHtml(sourceWindow: WindowLike, html: string): DocumentFragment {
  return createDOMPurify(sourceWindow).sanitize(html, {
    RETURN_DOM_FRAGMENT: true,
    ALLOWED_TAGS: ["p", "div", "span", "article", "section", "main", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "dl", "dt", "dd", "blockquote", "pre", "code", "br", "hr", "strong", "b", "em", "i", "u", "s", "sub", "sup", "a", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption"],
    ALLOWED_ATTR: ["href", "colspan", "rowspan", "scope"],
    ADD_URI_SAFE_ATTR: ["colspan", "rowspan", "scope"],
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
    ALLOWED_URI_REGEXP: /^https?:/i,
  });
}

export function tagFragmentAnchors(fragment: DocumentFragment): void {
  for (const anchor of fragment.querySelectorAll("a")) {
    const href = safeSourceUrl(anchor.getAttribute("href") ?? undefined);
    if (href) {
      anchor.setAttribute("href", href);
      anchor.setAttribute("target", "_blank");
      anchor.setAttribute("rel", "noopener noreferrer");
    } else anchor.removeAttribute("href");
  }
}

type QuotePart = { node: Text; start: number; end: number };

function quoteParts(root: HTMLElement, start: number, end: number): QuotePart[] {
  const owner = root.ownerDocument;
  const view = owner.defaultView as (Window & { NodeFilter?: typeof NodeFilter; Node?: typeof Node }) | null;
  const showText = view?.NodeFilter?.SHOW_TEXT ?? 4;
  const textNode = view?.Node?.TEXT_NODE ?? 3;
  const walker = owner.createTreeWalker(root, showText);
  const parts: QuotePart[] = [];
  let offset = 0;
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeType !== textNode) continue;
    const text = node as Text;
    const next = offset + (text.data.length ?? 0);
    if (next > start && offset < end) parts.push({ node: text, start: Math.max(0, start - offset), end: Math.min(next, end) - offset });
    offset = next;
  }
  return parts;
}

function paintQuotePart(owner: Document, part: QuotePart): HTMLElement {
  const range = owner.createRange();
  range.setStart(part.node, part.start);
  range.setEnd(part.node, part.end);
  const mark = owner.createElement("mark");
  mark.className = "mgl-cited-element-highlight";
  range.surroundContents(mark);
  return mark;
}


export function markQuote(root: HTMLElement, start: number, end: number): HTMLElement[] {
  const owner = root.ownerDocument;
  return quoteParts(root, start, end).map((part) => paintQuotePart(owner, part));
}

type MappedCharacter = { value: string; node?: Text; offset?: number; hidden?: boolean };
export type SourceTextMap = { text: string; characters: MappedCharacter[] };

const SEC_BLOCK_TAGS: Record<string, true> = { br: true, p: true, div: true, tr: true, h1: true, h2: true, h3: true, h4: true, h5: true, h6: true, li: true, table: true };
const SEC_END_BLOCK_TAGS: Record<string, true> = { p: true, div: true, h1: true, h2: true, h3: true, h4: true, h5: true, h6: true, li: true, table: true };
const SEC_CELL_TAGS: Record<string, true> = { td: true, th: true };
const SEC_SPACES = /^[ \t\v\f\r]$/u;
const PYTHON_SPACES = /^[\t-\r\x1c-\x20\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]$/u;
const PYTHON_LINES = /^[\n\r\v\f\x1c-\x1e\u0085\u2028\u2029]$/u;
const NOISE_START = "\uE000";
const NOISE_END = "\uE001";

function collapseSecSpaces(characters: MappedCharacter[]): MappedCharacter[] {
  const collapsed: MappedCharacter[] = [];
  for (const character of characters) {
    if (!SEC_SPACES.test(character.value)) collapsed.push(character);
    else if (collapsed.at(-1)?.value !== " ") collapsed.push({ ...character, value: " " });
  }
  return collapsed;
}

function trimCharacters(characters: MappedCharacter[], spaces: RegExp): MappedCharacter[] {
  let start = 0;
  let end = characters.length;
  while (start < end && spaces.test(characters[start].value)) start++;
  while (end > start && spaces.test(characters[end - 1].value)) end--;
  return characters.slice(start, end);
}

// Mirror app/sec/documents.py, including its synthetic table separators.
// The exact server comparison rejects DOM parser repairs that change this basis.
function prepareSecMarkup(html: string): { markup: string; hasReserved: boolean } {
  const hasReserved = html.includes(NOISE_START) || html.includes(NOISE_END) || /&#(?:0*5734[45]|x0*e00[01]);?/iu.test(html);
  let markup = html.replace(/<(style|script)[^>]*>[\s\S]*?<\/\1>/gi, "\n").replace(/<!--[\s\S]*?-->/g, "");
  if (!hasReserved) markup = markup.replace(
    /<[A-Za-z][\p{L}\p{N}_\u0300-\u036f.-]*:[^>]*>[\s\S]*?<\/[A-Za-z][\p{L}\p{N}_\u0300-\u036f.-]*:[^>]*>|<[A-Za-z][\p{L}\p{N}_\u0300-\u036f.-]*:[^>]*\/>/giu,
    (noise) => `${NOISE_START}${noise}${NOISE_END}`,
  );
  // Leading whitespace has no canonical characters. DOMPurify otherwise moves it into the body.
  markup = markup.replace(/^[\r\n\t ]+/u, "");
  return { markup, hasReserved };
}

function mountSecMarkup(root: HTMLElement, sourceWindow: WindowLike, markup: string): Document | undefined {
  const fragment = sanitizeSourceHtml(sourceWindow, markup);
  tagFragmentAnchors(fragment);
  root.append(fragment);
  return new sourceWindow.DOMParser().parseFromString(markup, "text/html") as Document | undefined;
}

type SecWalkState = { hasReserved: boolean; noise: boolean; invalid: boolean; chunks: MappedCharacter[]; cell: MappedCharacter[] | undefined; row: MappedCharacter[][] | undefined };

function pushSecSynthetic(state: SecWalkState, value: string): void {
  for (const character of value) state.chunks.push({ value: character });
}

function consumeSecNoiseMarker(state: SecWalkState, value: string): boolean {
  if (state.hasReserved) return false;
  if (value !== NOISE_START && value !== NOISE_END) return false;
  const opening = value === NOISE_START;
  if (state.noise === opening) state.invalid = true;
  state.noise = opening;
  return true;
}

function visitSecText(state: SecWalkState, node: Text, visible: boolean): void {
  const data = node.data;
  let kept = "";
  for (let offset = 0; offset < data.length; offset++) {
    const value = data[offset];
    if (consumeSecNoiseMarker(state, value)) continue;
    if (!state.noise) (state.cell ?? state.chunks).push(visible ? { value, node, offset: kept.length } : { value, hidden: true });
    kept += value;
  }
  if (visible && kept !== data) node.data = kept;
}

function enterSecElement(state: SecWalkState, name: string): void {
  if (state.noise) return;
  if (Object.hasOwn(SEC_BLOCK_TAGS, name)) pushSecSynthetic(state, "\n");
  if (name === "tr") state.row = [];
  else if (Object.hasOwn(SEC_CELL_TAGS, name)) state.cell = [];
}

function finishSecRow(state: SecWalkState): void {
  let first = true;
  for (const characters of state.row ?? []) {
    if (!characters.length) continue;
    if (!first) pushSecSynthetic(state, " | ");
    for (const character of characters) state.chunks.push(character);
    first = false;
  }
  if (!first) pushSecSynthetic(state, "\n");
  state.row = undefined;
}

function leaveSecElement(state: SecWalkState, name: string): void {
  if (state.noise) return;
  if (Object.hasOwn(SEC_CELL_TAGS, name)) {
    if (state.cell && state.row) {
      state.row.push(trimCharacters(collapseSecSpaces(state.cell), PYTHON_SPACES));
      state.cell = undefined;
    }
    return;
  }
  if (name === "tr") {
    finishSecRow(state);
    return;
  }
  if (Object.hasOwn(SEC_END_BLOCK_TAGS, name)) pushSecSynthetic(state, "\n");
}

function secElementName(node: Node, root: HTMLElement): string {
  if (node === root || node.nodeType !== 1) return "";
  return (node as Element).localName.toLowerCase().split(":").at(-1) ?? "";
}

function traverseSecDom(state: SecWalkState, parsed: Document | undefined, root: HTMLElement): void {
  // Traverse without recursion. Deep source markup cannot exhaust the call stack.
  const stack: { node: Node; visible: boolean; exit?: boolean }[] = [
    { node: parsed?.documentElement ?? root, visible: !parsed },
  ];
  while (stack.length) {
    const frame = stack.pop();
    if (!frame) break;
    if (frame.node === parsed?.body) {
      stack.push({ node: root, visible: true });
      continue;
    }
    if (frame.node.nodeType === 3) {
      visitSecText(state, frame.node as Text, frame.visible);
      continue;
    }
    const name = secElementName(frame.node, root);
    if (frame.exit) {
      leaveSecElement(state, name);
      continue;
    }
    enterSecElement(state, name);
    stack.push({ ...frame, exit: true });
    for (let child = frame.node.lastChild; child; child = child.previousSibling) stack.push({ node: child, visible: frame.visible });
  }
}

function assembleSecText(chunks: MappedCharacter[]): MappedCharacter[] {
  const lines: MappedCharacter[][] = [[]];
  for (const character of chunks) {
    if (PYTHON_LINES.test(character.value)) lines.push([]);
    else lines.at(-1)?.push(character);
  }
  const characters: MappedCharacter[] = [];
  for (const line of lines) {
    const normalized = trimCharacters(collapseSecSpaces(line), /^[ |]$/u);
    if (!normalized.length) continue;
    if (characters.length) characters.push({ value: "\n" });
    for (const character of normalized) characters.push(character);
  }
  return characters;
}

export function mapSourceHtml(root: HTMLElement, sourceWindow: WindowLike, html: string): SourceTextMap | undefined {
  const { markup, hasReserved } = prepareSecMarkup(html);
  const parsed = mountSecMarkup(root, sourceWindow, markup);
  const state: SecWalkState = { hasReserved, noise: false, invalid: hasReserved, chunks: [], cell: undefined, row: undefined };
  traverseSecDom(state, parsed, root);
  if (state.noise) state.invalid = true;
  const characters = assembleSecText(state.chunks);
  return state.invalid ? undefined : { text: characters.map((character) => character.value).join(""), characters };
}

function pythonLocation(text: string, location: SourceDocument["location"]): SourceDocument["location"] {
  if (!location) return undefined;
  const result = { ...location };
  let point = 0;
  let unit = 0;
  delete result.offset;
  delete result.end;
  for (const character of text) {
    if (point === location.offset) result.offset = unit;
    if (point === location.end) result.end = unit;
    point++;
    unit += character.length;
  }
  if (point === location.offset) result.offset = unit;
  if (point === location.end) result.end = unit;
  return result;
}

function mappedQuoteLocation(mapText: string, location: SourceDocument["location"]): SourceDocument["location"] {
  if (!location) return undefined;
  if (location.basis === "raw") return { prefix: location.prefix, suffix: location.suffix };
  return pythonLocation(mapText, location);
}

function mappedQuoteResolution(mapText: string, quote: string, location: SourceDocument["location"], hasCoordinates: boolean): QuoteResolution {
  if (!hasCoordinates) return resolveQuote(mapText, quote, location);
  const needle = quote.replace(/\s/gu, "");
  const matches = needle ? quoteMatches(mapText, needle) : [];
  const coordinate = coordinateMatch(matches, location);
  if (coordinate) return { status: "resolved", ...coordinate };
  return contextualMatch(mapText, matches, location) ?? { status: "unresolved" };
}

function mappedQuoteNodes(map: SourceTextMap, start: number, end: number): Map<Text, QuotePart> | undefined {
  const parts = new Map<Text, QuotePart>();
  for (let index = start; index < end; index++) {
    const character = map.characters[index];
    if (character.hidden) return undefined;
    if (!character.node || character.offset === undefined) continue;
    const part = parts.get(character.node);
    if (part) part.end = character.offset + 1;
    else parts.set(character.node, { node: character.node, start: character.offset, end: character.offset + 1 });
  }
  return parts.size ? parts : undefined;
}

export function highlightMappedQuote(root: HTMLElement, map: SourceTextMap | undefined, source: SourceDocument): { resolution: QuoteResolution; marks: HTMLElement[] } {
  // Never apply coordinates from a basis that this DOM cannot reproduce.
  if (!map || map.text !== source.matchingText) return { resolution: { status: "unresolved" }, marks: [] };
  const location = mappedQuoteLocation(map.text, source.location);
  const resolution = mappedQuoteResolution(map.text, source.quote, location, source.location?.offset !== undefined);
  if (resolution.status !== "resolved") return { resolution, marks: [] };
  const parts = mappedQuoteNodes(map, resolution.start, resolution.end);
  if (!parts) return { resolution: { status: "unresolved" }, marks: [] };
  return { resolution, marks: [...parts.values()].map((part) => paintQuotePart(root.ownerDocument, part)) };
}

export function activateQuote(root: HTMLElement, marks: HTMLElement[]): void {
  root.querySelectorAll(".mgl-active-element-highlight").forEach((element) => element.classList.remove("mgl-active-element-highlight"));
  for (const mark of marks) mark.classList.add("mgl-cited-element-highlight", "mgl-active-element-highlight");
  marks[0]?.scrollIntoView?.({ behavior: "smooth", block: "center" });
}

// Readable "publisher · title · date" label. Returns "" when the document names nothing; callers pick their own fallback.
export function sourceLabel(source: Pick<SourceDocument, "publisher" | "title" | "publishedAt" | "knownAt">): string {
  const parts = [source.publisher, source.title, (source.publishedAt || source.knownAt)?.slice(0, 10)];
  return [...new Set(parts.filter((part) => part !== undefined && part !== ""))].join(" · ");
}

export type OriginalLink = { href: string; host: string; highlighted: boolean };

// The official page link for a loaded source: the highlighted page when the backend verified a fragment, else the plain page.
// fetchSourceDocument already checked both URLs with safeSourceUrl. A missing official page has no link.
export function originalLink(source: Pick<SourceDocument, "linkStatus" | "originalUrl" | "highlightUrl">): OriginalLink | undefined {
  const highlighted = source.linkStatus === "highlighted_original";
  const href = highlighted ? source.highlightUrl : source.linkStatus === "stockbot_render" ? source.originalUrl : undefined;
  if (!href) return undefined;
  const hostname = new URL(href).hostname;
  const host = hostname === "www.sec.gov" || hostname === "sec.gov" ? "SEC.gov" : hostname === "finra.org" || hostname.endsWith(".finra.org") ? "FINRA.org" : hostname;
  return { href, host, highlighted };
}
