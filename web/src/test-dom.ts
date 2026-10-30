// Test harness: the jsdom globals must exist before react-dom is evaluated.
// React reads canUseDOM once at module load. Without a document it takes the legacy
// change-event polyfill, and reported input changes never reach a controlled onChange.
// A static import of react-dom cannot work in the testing module itself, because module
// evaluation runs before any statement of that module, so this setup is its own module.
import { JSDOM } from "jsdom";

export const dom = new JSDOM(`<body><div id="root"></div></body>`, { url: "https://example.com/" });

// Unchecked cast: the registry holds the jsdom globals a browser would own.
const registry = globalThis as Record<string, unknown>;
registry.IS_REACT_ACT_ENVIRONMENT = true;
registry.window = dom.window;
registry.document = dom.window.document;
registry.navigator = dom.window.navigator;
registry.requestAnimationFrame = (callback: FrameRequestCallback) => setTimeout(() => callback(0), 0) as unknown as number;
registry.cancelAnimationFrame = (frame: number) => clearTimeout(frame);

// jsdom does not measure layout. Real browser tests check resize behavior.
registry.ResizeObserver = class {
  observe() { }
  unobserve() { }
  disconnect() { }
};
