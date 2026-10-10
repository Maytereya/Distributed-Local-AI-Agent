import { readFileSync } from "node:fs";

const ADMIN = "156841688";
const GROUPS = new Set(["-5236901876", "-1003599248220"]);
const ORIGIN = "http://openclaw-monitor:18080";
const PRIVATE_ONLY = "Полный отчёт доступен администратору в личном чате с ботом.";
const DENIED = "Недостаточно прав для этой команды.";
const GROUP_HELP = "В группе доступны /chatbot_status и /telegram_pulse. Подробные запросы — в личном чате с ботом.";
const REPORTS = [
  { command: "server_status", tool: "ai_server_monitor_report", url: "/report", config: "reportUrl", private: true, description: "Состояние сервера и служб — личный чат" },
  { command: "chatbot_status", tool: "ai_server_monitor_brief_report", url: "/report?format=aggregator_brief", config: "briefReportUrl", description: "Отчёт корпоративного чатбота" },
  { command: "telegram_pulse", tool: "ai_server_telegram_pulse_report", url: "/telegram_pulse", config: "pulseReportUrl", description: "Доступность Telegram и статистика сбоев" },
  { command: "failures", tool: "ai_server_failure_summary", url: "/failures", private: true, description: "Локальный разбор проблемных обращений — личный чат" },
  { command: "failure", tool: "ai_server_failure_card", url: "/failure", private: true, description: "Карточка обращения по номеру — личный чат" },
  { command: "security_status", tool: "ai_server_security_report", url: "/security_status", private: true, description: "Последняя проверка безопасности — личный чат" },
];

const PERIOD_HELP = "Периоды: day (24 часа), today (сегодня), week (прошлая неделя), month (прошлый месяц), all (с начала учёта), YYYY-MM-DD..YYYY-MM-DD.";
export function validPeriod(value) {
  if (["day","today","week","month","all"].includes(value)) return true;
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}(?:\.\.\d{4}-\d{2}-\d{2})?$/.test(value)) return false;
  const dates=value.split("..");
  try { if (!dates.every(d=>new Date(d+"T00:00:00Z").toISOString().slice(0,10)===d)) return false; } catch { return false; }
  return dates.at(-1)>=dates[0];
}
export function commandURL(definition, args="") {
  args=String(args||"").trim().toLowerCase();
  if (definition.command==="server_status") {
    if (args && !/^(json|detail|подробно|full|полный)$/.test(args)) throw new Error("server_arguments");
    return /^(json|detail|подробно)$/.test(args) ? ORIGIN+"/check" : ORIGIN+"/report";
  }
  const target=new URL(ORIGIN+definition.url);
  if (definition.command==="failure") {
    if (!/^[1-9][0-9]{0,17}$/.test(args)) throw new Error("case_id_required");
    target.searchParams.set("id",args);
  } else if (definition.command!=="security_status") {
    const aliases={сутки:"day",сегодня:"today",неделя:"week",месяц:"month",все:"all"};
    const period=aliases[args]||args||"day";
    if (!validPeriod(period)) throw new Error("period_required");
    target.searchParams.set("period",period);
  } else if(args) throw new Error("no_arguments");
  return target.href;
}

export function groupDispatchGuard(event = {}, ctx = {}) {
  const key = String(ctx.sessionKey || event.sessionKey || "");
  const telegram = (event.channel || ctx.channelId) === "telegram" || key.includes(":telegram:group:");
  const group = event.isGroup === true || /:telegram:group:-\d+(?::topic:\d+)?$/.test(key)
    || /(?:^|:)-\d+$/.test(String(ctx.conversationId || ""));
  // Native commands have already been handled by the command router. Never
  // send ordinary group prompts to a model or load a private session for them.
  if (telegram && group) return { handled: true, text: GROUP_HELP };
}

export function permittedCommand(ctx, privateOnly) {
  if (ctx.channel !== "telegram" || !ctx.isAuthorizedSender || String(ctx.senderId).replace(/^telegram:/, "") !== ADMIN) return false;
  const locations = [ctx.from, ctx.to].filter(Boolean).map(String);
  const group = locations.map(x => x.match(/(?:^|:)(-\d+)$/)?.[1]).find(Boolean);
  if (group) return !privateOnly && GROUPS.has(group);
  return locations.some(x => x === ADMIN || x === `telegram:${ADMIN}`);
}

export function permittedTool(ctx, privateOnly) {
  if (ctx.messageChannel !== "telegram") return false;
  const key = String(ctx.sessionKey || "");
  if (key === "agent:main:main" || key === `agent:main:telegram:direct:${ADMIN}`) return true;
  const group = key.match(/^agent:[^:]+:telegram:group:(-\d+)(?::topic:\d+)?$/)?.[1];
  return !privateOnly && GROUPS.has(group);
}

export async function fetchReport(url, readToken = () => readFileSync("/run/secrets/monitor_api_token", "utf8").trim()) {
  const target = new URL(url);
  if (target.origin !== ORIGIN || target.username || target.password || target.hash || !["/report", "/check", "/telegram_pulse", "/failures", "/failure", "/security_status"].includes(target.pathname)) throw new Error("invalid_monitor_endpoint");
  if ([...target.searchParams.keys()].some(key => !["format","period","id"].includes(key))) throw new Error("invalid_monitor_endpoint");
  if (target.searchParams.has("period") && !validPeriod(target.searchParams.get("period"))) throw new Error("invalid_monitor_endpoint");
  if (target.searchParams.has("id") && !/^[1-9][0-9]{0,17}$/.test(target.searchParams.get("id"))) throw new Error("invalid_monitor_endpoint");
  const token = readToken();
  if (!token) throw new Error("monitor_credential_unavailable");
  const response = await fetch(target, { headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(120_000), redirect: "error" });
  if (!response.ok) return `Мониторинг временно недоступен (HTTP ${response.status}).`;
  const text = await response.text();
  if (text.length > 32_000) return "Отчёт превышает допустимый размер. Обратитесь к администратору.";
  return text;
}

export default {
  id: "ai-server-monitor",
  name: "AI Server Monitor",
  description: "Защищённые отчёты локального мониторинга.",
  register(api) {
    api.on("before_dispatch", groupDispatchGuard, { priority: 1000 });
    const report = async (definition, args = "") => {
      try {
        return await fetchReport(commandURL(definition,args));
      } catch(error) {
        if(error.message==="case_id_required")return "Укажите номер обращения: /failure 123.";
        if(error.message==="period_required")return PERIOD_HELP;
        if(error.message==="no_arguments")return "Команда /security_status выполняется без аргументов.";
        if(error.message==="server_arguments")return "Используйте /server_status или /server_status json.";
        return "Мониторинг временно недоступен. Повторите запрос позже.";
      }
    };
    for (const definition of REPORTS) {
      api.registerCommand({
        name: definition.command, description: definition.description, acceptsArgs: true, requireAuth: true,
        async handler(ctx) {
          if (!permittedCommand(ctx, definition.private)) return { text: definition.private ? PRIVATE_ONLY : DENIED };
          return { text: await report(definition, ctx.args) };
        },
      });
      api.registerTool(ctx => {
        if (!permittedTool(ctx, definition.private)) return null;
        return {
          name: definition.tool, description: definition.description,
          parameters: { type: "object", additionalProperties: false, properties: { command: { type: "string" }, commandName: { type: "string" }, skillName: { type: "string" } } },
          async execute(_id, params = {}) { return { content: [{ type: "text", text: await report(definition, params.command) }] }; },
        };
      }, { name: definition.tool });
    }
  },
};
