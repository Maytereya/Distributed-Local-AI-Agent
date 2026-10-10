// Patch only the server's existing diagnostics, preserving its application.
const fs=require('fs');
let body=fs.readFileSync('/source/container_managenment/system_data.py','utf8');
body='import secure_runtime_metrics as _safe_runtime\n'+body;
const patches=[
  ['def get_host_cpu_count() -> int:', '    if _safe_runtime.configured(): return _safe_runtime.cpu_count()'],
  ['def get_docker_containers_stats(container_names: List[str]) -> Dict[str, Any]:','    if _safe_runtime.configured(): return _safe_runtime.container_stats(container_names)'],
  ['def get_container_diagnostics(', null],
];
for(const [needle,addition] of patches){
  if(!body.includes(needle))throw new Error('diagnostic_signature_changed');
  if(addition)body=body.replace(needle+'\n',needle+'\n'+addition+'\n');
}
const marker='def get_container_diagnostics(';
const first=body.indexOf(marker), end=body.indexOf(') -> Dict[str, Any]:',first);
if(end<0)throw new Error('diagnostic_signature_changed');
const pos=body.indexOf('\n',end);
body=body.slice(0,pos+1)+'    if _safe_runtime.configured(): return _safe_runtime.diagnostics(container_name, log_tail)\n'+body.slice(pos+1);
fs.writeFileSync('/target/system_data.py',body);
let restart=fs.readFileSync('/source/container_managenment/restart_container.py','utf8');
const signature='def restart_ollama_container(container_name="ollama") -> str:';
if(!restart.includes(signature))throw new Error('restart_signature_changed');
restart='import secure_runtime_metrics as _safe_runtime\n'+restart.replace(signature+'\n',signature+'\n    if _safe_runtime.configured(): return _safe_runtime.restart_ollama(container_name)\n');
fs.writeFileSync('/target/restart_container.py',restart);
let ui=fs.readFileSync('/source/gradio_interface.py','utf8');
if(ui.includes('from __future__ import annotations'))ui=ui.replace('from __future__ import annotations','from __future__ import annotations\nfrom dashboard_security import admin_callbacks');
else ui='from dashboard_security import admin_callbacks\n'+ui;
let lines=ui.split('\n');
const docsStart=lines.findIndex(l=>l.includes('with gr.Tab("\\U0001F4E4 Документы") as documents_tab:'));
const docsEnd=lines.findIndex((l,i)=>i>docsStart&&l.includes('# -------- FUNCTIONS SECTION'));
if(docsStart<0||docsEnd<0)throw Error('dashboard_documents_signature_changed');
const docsIndent=lines[docsStart].match(/^\s*/)[0];
lines.splice(docsStart,docsEnd-docsStart,docsIndent+'with admin_callbacks():',...lines.slice(docsStart,docsEnd).map(l=>'    '+l));
for(const builder of ['build_messenger_settings_tab','build_system_settings_tab','build_benchmark_tab','build_monitoring_tab']){
 const start=lines.findIndex(l=>new RegExp('^\\s+\\w+ = '+builder+'\\($').test(l));
 if(start<0)throw Error('dashboard_builder_signature_changed');
 const indent=lines[start].match(/^\s*/)[0];
 const end=lines.findIndex((l,i)=>i>start&&l===indent+')');
 if(end<0)throw Error('dashboard_builder_signature_changed');
 lines.splice(start,end-start+1,indent+'with admin_callbacks():',...lines.slice(start,end+1).map(l=>'    '+l));
}
fs.writeFileSync('/target/gradio_interface.py',lines.join('\n'));
console.log('fixed_dashboard_diagnostics_prepared');
