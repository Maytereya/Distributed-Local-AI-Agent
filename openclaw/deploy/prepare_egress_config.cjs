// Server-only. Preserve existing credentials; never print secret values.
const fs = require('node:fs');
const crypto = require('node:crypto');
const configPath = '/state/openclaw.json';
const c = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const token = c.channels?.telegram?.botToken;
if (typeof token !== 'string' || !token.trim()) throw new Error('telegram_token_unavailable');
fs.writeFileSync('/secrets/telegram_monitor_bot_token', token, {mode:0o600});
fs.chmodSync('/secrets/telegram_monitor_bot_token', 0o600);
const monitorKey = fs.readFileSync('/secrets/monitor_api_token', 'utf8').trim();
if (!/^[a-f0-9]{64}$/.test(monitorKey)) throw new Error('invalid_monitor_key');
fs.writeFileSync('/secrets/monitor_health.curl', `header = "Authorization: Bearer ${monitorKey}"\n`, {mode:0o600});
const password = c.gateway?.auth?.password;
if (typeof password === 'string' && password.length) {
  // Telegram credentials are unchanged. A weak UI password is rotated locally.
  fs.writeFileSync('/secrets/gateway_password', password.length < 20 ? crypto.randomBytes(32).toString('base64url') : password, {mode:0o600});
  fs.chmodSync('/secrets/gateway_password', 0o600);
} else if (!fs.existsSync('/secrets/gateway_password')) {
  throw new Error('gateway_password_unavailable');
}
c.secrets ||= {};
c.secrets.providers ||= {};
c.secrets.providers.gateway_password = {source:'file',path:'/run/secrets/gateway_password',mode:'singleValue'};
c.gateway.auth.password = {source:'file',provider:'gateway_password',id:'value'};
c.channels.telegram.proxy = 'http://reporting-egress-guard:3128';
const temporary = configPath + '.egress-new';
fs.writeFileSync(temporary, JSON.stringify(c, null, 2), {mode:0o600});
fs.renameSync(temporary, configPath);
console.log('protected_secrets_and_proxy_prepared');
