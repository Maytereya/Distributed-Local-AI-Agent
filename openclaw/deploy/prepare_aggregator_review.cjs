// Patch the CURRENT deployed UI sources; never overwrite a colleague's source
// with the July checkout. /source and /target are server-only build directories.
const fs=require('fs');
function replaceOne(body,needle,value){
 if(body.split(needle).length!==2)throw Error('source_signature_changed');
 return body.replace(needle,value);
}
let urls=fs.readFileSync('/source/urls.py','utf8');
if(urls.includes('include("reporting.urls")'))throw Error('review_routes_already_present');
urls=replaceOne(urls,'    path("", include("operator_panel.urls")),','    path("reporting/", include("reporting.urls")),\n    path("", include("operator_panel.urls")),');
let base=fs.readFileSync('/source/base.html','utf8');
base=replaceOne(base,'https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css',"{% static 'operator_panel/vendor/pico-2.1.1.min.css' %}");
const nav='        <li><a href="/patients/" {% if nav_active == "patients" %}aria-current="page"{% endif %}>Пациенты</a></li>';
base=replaceOne(base,nav,nav+'\n        <li><a href="/reporting/appeals/" {% if nav_active == "reporting" %}aria-current="page"{% endif %}>Проверка обращений</a></li>');
fs.writeFileSync('/target/urls.py',urls);
fs.writeFileSync('/target/base.html',base);
console.log('current_aggregator_review_overlay_prepared');
