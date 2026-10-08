import type { Tool } from "../agent/types";
import { get_current_time } from "./current-time";
import { invoke } from "./stockbot";

// Gateway-backed entries: parameters stay {} because validation is owned
function gateway(name: string): Tool {
  return {
    description: name,
    parameters: { type: "object", properties: {} },
    async execute(args, opts) {
      return invoke(name, args, opts.sessionId);
    },
  };
}

export const tools: Record<string, Tool> = {
  search_web: gateway("search_web"),
  find_sec_entities: gateway("find_sec_entities"),
  find_sec_entities_bounded: gateway("find_sec_entities_bounded"),
  search_sec_filings: gateway("search_sec_filings"),
  search_sec_filings_bounded: gateway("search_sec_filings_bounded"),
  get_sec_document: gateway("get_sec_document"),
  get_current_time,
};
