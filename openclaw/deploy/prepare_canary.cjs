// Keep production history and credentials outside the canary.
const fs = require("node:fs");
const crypto = require("node:crypto");
const source = JSON.parse(fs.readFileSync("/source/openclaw.json", "utf8"));
const primary = typeof source.agents?.defaults?.model === "string" ? source.agents.defaults.model : source.agents?.defaults?.model?.primary;
if (!String(primary).startsWith("ollama/")) throw new Error("local_model_required");
const provider = structuredClone(source.models?.providers?.ollama);
if (!provider || new URL(provider.baseUrl).hostname !== "ollama") throw new Error("local_ollama_required");
provider.apiKey = "local-only";
const target = {
  models: { mode: "merge", providers: { ollama: provider } },
  agents: { defaults: { model: { primary }, workspace: "/home/node/.openclaw/workspace", sandbox: { mode: "off" } } },
  channels: { telegram: { enabled: false } },
  plugins: { allow: ["ai-server-monitor"], load: { paths: ["/extensions/ai-server-monitor"] }, entries: { "ai-server-monitor": { enabled: true }, telegram: { enabled: false } } },
  gateway: { mode: "local", bind: "lan", port: 18791, auth: { mode: "token", token: crypto.randomBytes(32).toString("hex") } },
  tools: { deny: ["group:runtime", "group:fs", "group:web", "browser", "message", "sessions_send", "sessions_spawn"] },
  logging: { level: "warn", consoleLevel: "warn" },
};
fs.mkdirSync("/target/workspace", { recursive: true, mode: 0o700 });
fs.writeFileSync("/target/openclaw.json", JSON.stringify(target, null, 2), { mode: 0o600 });
