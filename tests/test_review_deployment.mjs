import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const code=fs.readFileSync(new URL('../openclaw/deploy/prepare_aggregator_review.cjs',import.meta.url),'utf8');
const urls='from django.urls import include, path\nurlpatterns = [\n    path("colleague/", include("colleague.urls")),\n    path("", include("operator_panel.urls")),\n]\n';
const base=`{% load static %}
<link href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
        <li><a href="/patients/" {% if nav_active == "patients" %}aria-current="page"{% endif %}>Пациенты</a></li>
<div id="colleague-feature">UNCHANGED</div>`;

function prepare(input){
 const result={};
 const bridge={readFileSync(name){assert.ok(Object.hasOwn(input,name));return input[name];},writeFileSync(name,value){result[name]=value;}};
 vm.runInNewContext(code,{require(name){assert.equal(name,'fs');return bridge;},console:{log(){}}});
 return result;
}

test('review overlay keeps existing routes and UI while removing the CDN',()=>{
 const result=prepare({'/source/urls.py':urls,'/source/base.html':base});
 assert.ok(result['/target/urls.py'].includes('include("colleague.urls")'));
 assert.ok(result['/target/urls.py'].includes('include("reporting.urls")'));
 assert.ok(result['/target/base.html'].includes('id="colleague-feature">UNCHANGED'));
 assert.ok(result['/target/base.html'].includes("operator_panel/vendor/pico-2.1.1.min.css"));
 assert.ok(!result['/target/base.html'].includes('cdn.jsdelivr.net'));
});

test('changed route signature stops before output',()=>{
 assert.throws(()=>prepare({'/source/urls.py':urls.replace('operator_panel.urls','new_panel.urls'),'/source/base.html':base}),/source_signature_changed/);
});

test('an already patched route cannot be silently patched twice',()=>{
 assert.throws(()=>prepare({'/source/urls.py':urls+'\ninclude("reporting.urls")','/source/base.html':base}),/review_routes_already_present/);
});
