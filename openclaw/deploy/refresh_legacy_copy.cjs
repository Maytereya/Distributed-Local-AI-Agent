// Preserve updates from the still-running legacy Gateway after a failed newer
// migration. Start from the pre-update archive; never reuse newer SQLite files
// for the bridge release. All contents remain inside protected server storage.
const fs = require("node:fs");
const path = require("node:path");
const source = "/current";
const target = "/incoming/.openclaw";
function copy(relative, accept = () => true) {
  const src = path.join(source, relative);
  if (!fs.existsSync(src)) return;
  fs.cpSync(src, path.join(target, relative), {
    recursive: true, preserveTimestamps: true,
    filter(p) {
      const stat = fs.lstatSync(p);
      if (stat.isSymbolicLink() || /\.migrated\.|\.sqlite(?:-|$)|\.db(?:-|$)/.test(p)) return false;
      return stat.isDirectory() || accept(p);
    },
  });
}
for (const id of fs.readdirSync(path.join(source, "agents"))) copy(`agents/${id}/sessions`, p => /\.jsonl?$/.test(p));
copy("telegram", p => /\.json$/.test(p));
copy("cron", p => /\.jsonl?$/.test(p));
copy("workspace");
// The active config has already been restored to the legacy version.
fs.copyFileSync(path.join(source, "openclaw.json"), path.join(target, "openclaw.json"));
