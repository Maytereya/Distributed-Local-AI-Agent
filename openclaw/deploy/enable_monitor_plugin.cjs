// Run locally in a maintenance container. Never print the configuration.
const fs = require("node:fs");
const file = "/state/openclaw.json";
const config = JSON.parse(fs.readFileSync(file, "utf8"));
config.plugins ||= {};
config.plugins.entries ||= {};
config.plugins.entries["ai-server-monitor"] ||= {};
config.plugins.entries["ai-server-monitor"].enabled = true;
fs.writeFileSync(file + ".security.tmp", JSON.stringify(config, null, 2) + "\n", { mode: 0o600 });
fs.chownSync(file + ".security.tmp", 1000, 1000);
fs.renameSync(file + ".security.tmp", file);
