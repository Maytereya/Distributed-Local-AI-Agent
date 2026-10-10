// Server-side patch of the preserved production source, including its menu API.
const fs=require('fs');
function replaceOne(text,needle,value){if(text.split(needle).length!==2)throw Error('source_signature_changed');return text.replace(needle,value);}
for(const name of ['endpoint.py','mess_types.py','router.py','orchestrator.py','recovery_policy.py']){
 let body=fs.readFileSync('/source/'+name,'utf8');
 if(name==='mess_types.py'){
  const start=body.indexOf('class ResponseEnvelope:');if(start<0)throw Error('source_signature_changed');
  body=body.slice(0,start)+replaceOne(body.slice(start),'    state_update: dict[str, Any] = field(default_factory=dict)','    state_update: dict[str, Any] = field(default_factory=dict)\n    reporting: dict[str, Any] = field(default_factory=dict)');
 }else if(name==='orchestrator.py'){
  body=replaceOne(body,'    if prebuilt is not None:\n        ctx.response = prebuilt','    if prebuilt is not None:\n        ctx._reporting_structured = True\n        ctx.response = prebuilt');
 }else if(name==='router.py'){
  const matches=body.match(/^    response = _maybe_offer_operator_on_repeat\([^\n]+\)$/gm);if(!matches||matches.length!==1)throw Error('source_signature_changed');
  body=replaceOne(body,matches[0],matches[0]+'\n\n    from .reporting_facts import scenario_facts\n    ctx.response = response\n    response.reporting = scenario_facts(ctx, memory)');
 }else if(name==='recovery_policy.py'){
  body=replaceOne(body,'def explicit_operator_requested(user_text: str) -> bool:', 'def explicit_operator_requested(user_text: str) -> bool:\n    from operator_signals import operator_refused\n    if operator_refused(user_text): return False');
  body=replaceOne(body,'def contextual_reply_kind(user_text: str) -> Literal["yes", "no", "other"]:', 'def contextual_reply_kind(user_text: str) -> Literal["yes", "no", "other"]:\n    from operator_signals import operator_refused\n    if operator_refused(user_text): return "no"');
 }else{
  body=replaceOne(body,'from api_security import messenger_debug_allowed','from api_security import messenger_debug_allowed\nfrom reporting_contract import unknown_contract, validate_contract');
  body=replaceOne(body,'class MessengerGenerateRequest(BaseModel):','class MessengerGenerateRequest(BaseModel):\n    operator_offer_pending: bool = False');
  if(body.split('    state = await memory.aget(session_id)').length!==3)throw Error('source_signature_changed');
  body=body.replaceAll('    state = await memory.aget(session_id)','    state = await memory.aget(session_id)\n    if payload.operator_offer_pending: state.last_entities["_operator_offer_pending"] = True');
  body=replaceOne(body,'                    "handoff": bool(env.handoff),','                    "handoff": bool(env.handoff),\n                    "reporting": validate_contract(getattr(env,"reporting",None)),\n                    "is_final": True,');
  body=replaceOne(body,'                "handoff": True,','                "handoff": True,\n                "reporting": unknown_contract("technical_error"),\n                "is_final": True,');
  const once=body.indexOf('async def messenger_generate_once(');if(once<0)throw Error('source_signature_changed');
  let tail=body.slice(once);
  tail=replaceOne(tail,'        handoff = False','        handoff = False\n        reporting = unknown_contract()');
  tail=replaceOne(tail,'                if env.text:','                if not (env.state_update or {}).get("debug"):\n                    reporting = validate_contract(getattr(env,"reporting",None))\n                if env.text:');
  tail=replaceOne(tail,'            handoff = True\n            if payload.debug:', '            handoff = True\n            reporting = unknown_contract("technical_error")\n            if payload.debug:');
  tail=replaceOne(tail,'            state_update=state_update,','            state_update=state_update,\n            reporting=reporting,');
  body=body.slice(0,once)+tail;
  const out=body.indexOf('class ResponseEnvelopeOut('),line=body.indexOf('class ResponseEnvelopeLine(');
  if(out<0||line<0)throw Error('source_signature_changed');
  body=body.slice(0,out)+body.slice(out,line).replace('    text: str = Field(', '    reporting: dict[str, Any] = Field(default_factory=unknown_contract)\n    text: str = Field(')+body.slice(line);
 }
 fs.writeFileSync('/target/'+name,body);
}
console.log('production_reporting_patch_prepared');
