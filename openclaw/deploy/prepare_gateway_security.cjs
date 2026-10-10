// Run only on the server with protected configuration mounted at /state.
// Never print credentials or copy production history into the test instance.
const fs = require("node:fs");
const p = "/state/openclaw.json";
const c = JSON.parse(fs.readFileSync(p, "utf8"));
const merge = (...lists) => [...new Set(lists.flat().filter(Boolean))];
const sandbox = {
  mode: "all", backend: "ssh", scope: "session", workspaceAccess: "none",
  ssh: { target: "sandbox@openclaw-sandbox:2222", workspaceRoot: "/srv/workspaces",
    identityFile: "/run/secrets/sandbox_identity", knownHostsFile: "/run/secrets/sandbox_known_hosts",
    strictHostKeyChecking: true },
};
c.agents.defaults.sandbox = sandbox;
for (const agent of c.agents.list || []) agent.sandbox = structuredClone(sandbox);
for (const agent of Object.values(c.agents.entries || {})) agent.sandbox = structuredClone(sandbox);
c.tools ||= {};
c.tools.exec = { ...c.tools.exec, host: "sandbox" };
c.tools.deny = merge(c.tools.deny || [], ["group:web", "browser", "message", "sessions_send", "sessions_spawn"]);
c.tools.elevated = { ...c.tools.elevated, enabled: false };
c.channels.telegram.dmPolicy = "allowlist";
c.channels.telegram.allowFrom = ["156841688"];
c.channels.telegram.groupAllowFrom = ["156841688"];
c.channels.telegram.historyLimit = 0;
for (const id of ["-5236901876", "-1003599248220"]) {
  const group = c.channels?.telegram?.groups?.[id];
  if (!group) throw new Error("report_group_missing");
  group.tools ||= {};
  group.tools.deny = merge(group.tools.deny || [], ["group:runtime", "group:fs", "group:sessions", "group:web", "browser", "memory_search", "memory_get"]);
}
c.gateway.auth.rateLimit = { maxAttempts: 10, windowMs: 60000, lockoutMs: 300000, exemptLoopback: false };
c.gateway.controlUi ||= {};
c.gateway.controlUi.allowedOrigins = merge(c.gateway.controlUi.allowedOrigins || [], ["http://127.0.0.1:18789", "http://localhost:18789", "http://172.16.0.28:18789"]);
c.plugins ||= {};
c.plugins.load ||= {};
c.plugins.load.paths = merge((c.plugins.load.paths || []).filter(p => !p.endsWith("ai-server-monitor")), ["/extensions/ai-server-monitor"]);
c.plugins.entries ||= {};
c.plugins.entries["ai-server-monitor"] = { ...c.plugins.entries["ai-server-monitor"], enabled: true };
c.plugins.allow = merge(c.plugins.allow || [], ["ai-server-monitor", "telegram"]);
const tmp = p + ".security-new";
fs.writeFileSync(tmp, JSON.stringify(c, null, 2), { mode: 0o600 });
fs.renameSync(tmp, p);
