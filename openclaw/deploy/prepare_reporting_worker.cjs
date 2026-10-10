// Actual env is projected entirely on the server; stdout never includes values.
const fs=require('fs');
const env=Object.fromEntries(JSON.parse(fs.readFileSync('/source/aggregator.env.json','utf8')).map(v=>{const i=v.indexOf('=');return [v.slice(0,i),v.slice(i+1)];}));
const cfg=JSON.parse(fs.readFileSync('/gateway/openclaw.json','utf8'));
const primary=cfg.agents.defaults.model.primary;
if(typeof primary!=='string'||!/^ollama\/[A-Za-z0-9_.:/-]{1,96}$/.test(primary))throw Error('unsupported_local_model');
const selected={DATABASE_URL:env.DATABASE_URL,DJANGO_SECRET_KEY:env.DJANGO_SECRET_KEY,DJANGO_DEBUG:'False',DJANGO_ALLOWED_HOSTS:'localhost',
 OLLAMA_API_BASE:'http://ollama:11434',REPORTING_ANALYZER_MODEL:primary.slice(7),REPORTING_WORKER:'1'};
for(const value of Object.values(selected))if(typeof value!=='string'||!value||/[\r\n\0]/.test(value))throw Error('worker_environment_unavailable');
const path='/target/reporting-worker.env';fs.writeFileSync(path,Object.entries(selected).map(([k,v])=>k+'='+v).join('\n')+'\n',{mode:0o600});fs.chownSync(path,1000,1000);fs.chmodSync(path,0o600);
console.log('limited_worker_environment_prepared');
