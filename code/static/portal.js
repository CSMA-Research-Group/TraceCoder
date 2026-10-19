"use strict";
let portalReady=false, officialExamples=[], savedGeneration=null;
const categoryNames={array:"数组",math:"数学",algorithm:"算法","data-structure":"数据结构",string:"字符串"};
const difficultyNames={easy:"基础",medium:"进阶",hard:"挑战"};
function generationInput(){return {title:$("#gen-title").value.trim(),description:$("#gen-description").value,test:$("#gen-tests").value,initial_code:$("#gen-start").value==="frozen"?$("#gen-code").value:"",start_mode:$("#gen-start").value,entry_point:$("#gen-entry").value.trim(),source_ref:$("#gen-source").value.trim()};}
function generationSignature(){return JSON.stringify(generationInput());}
function updateGenerationMode(){const frozen=$("#gen-start").value==="frozen";$("#gen-code-field").hidden=!frozen;$("#gen-code").required=frozen;$("#gen-run").textContent=frozen?"执行轨迹修复 ↗":"生成并修复 ↗";}
function refreshGenerationStatus(){
  const live=state.config.live_enabled;
  $("#gen-readiness").innerHTML=`<i class="status-dot ${live?"":"pending"}"></i><div><strong>${live?"执行环境已配置":"等待模型与执行环境"}</strong><p>${live?"运行将调用服务器模型并执行固定测试。":"可以先保存任务版本，稍后接入模型再运行。"}</p></div>`;
  $("#gen-run").disabled=!live||!state.config.projects_enabled;
  $("#gen-save").disabled=!state.config.projects_enabled;
  if(!state.config.projects_enabled)$("#gen-saved").textContent="当前为公开演示模式；在私有工作台中可保存业务任务。";
}
async function saveGeneration(){
  if(!$("#generation-form").reportValidity())return null;
  const signature=generationSignature();
  if(savedGeneration?.signature===signature)return savedGeneration.task;
  const body={task:generationInput()};
  if(savedGeneration){body.project_id=savedGeneration.task.project_id;body.task_id=savedGeneration.task.id;}
  const task=await api("/api/generation/tasks",body);
  savedGeneration={task,signature};
  $("#gen-saved").textContent=`已保存 v${task.revision} · ${task.revision_sha256.slice(0,12)}。输入已冻结，尚未执行。`;
  return task;
}
async function generateAndRepair(){
  const button=$("#gen-run");button.disabled=true;notice("");
  try{
    const task=await saveGeneration();if(!task)return;
    const run=await api("/api/run_tracecoder",{mode:"live",task_id:task.id,revision:task.revision,max_debug_attempts:Number($("#gen-rounds").value),max_no_improvement:Number($("#gen-patience").value),timeout:Number($("#gen-timeout").value),enable_instrumentation:$("#gen-instrumentation").checked});
    await loadRun(run.id);
  }catch(e){notice(e.message,true);}finally{refreshGenerationStatus();}
}
function loadExample(id){
  const example=officialExamples.find(value=>value.id===id);if(!example)return;
  savedGeneration=null;$("#gen-saved").textContent="";
  $("#gen-title").value=example.name;$("#gen-start").value="generate";$("#gen-code").value="";
  // The generation agent sees the specification only; make required symbol names
  // explicit without leaking test inputs or expected outputs into generation.
  const builtins=new Set(["assert","sorted","list","range","set","len","str","int","float","bool","tuple"]);
  const symbols=[...new Set([...example.test_case.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)].map(m=>m[1]).filter(s=>!builtins.has(s)))];
  $("#gen-description").value=example.description+"\n\n接口名称约定（从官方示例测试提取，请核对参数与行为）："+symbols.join("、")+"。";
  $("#gen-tests").value=example.test_case;$("#gen-entry").value="";
  $("#gen-source").value=`TraceCoder-WebDemo@49beb48:examples/${example.id}`;
  updateGenerationMode();showView("generate");notice("已载入官方原始测试，并在需求中补充测试引用的接口名称。保存前可调整任务说明。");
}
function renderOfficialExamples(){
  const query=$("#example-search").value.trim().toLowerCase();
  const examples=officialExamples.filter(e=>[e.name,e.description,e.category,categoryNames[e.category]].join(" ").toLowerCase().includes(query));
  $("#example-count").textContent=`${examples.length} / ${officialExamples.length} 个官方示例`;
  $("#official-examples").innerHTML=examples.length?examples.map((e)=>`<article class="example-card"><div class="example-meta"><span>${escapeHTML(categoryNames[e.category]||e.category)}</span><span class="difficulty ${e.difficulty}">${escapeHTML(difficultyNames[e.difficulty])}</span></div><span class="example-number">${escapeHTML(e.id.replace("example","").padStart(2,"0"))}</span><h2>${escapeHTML(e.name)}</h2><p>${escapeHTML(e.description)}</p><div class="example-tests"><span>FIXED ASSERTIONS</span><code>${escapeHTML(e.test_case.split("\n").find(line=>line.startsWith("assert"))||"结构化测试")}</code></div><button class="text-button" data-load-example="${e.id}">载入任务 <span>↗</span></button></article>`).join(""):empty("没有匹配的示例","尝试搜索算法、数组、字符串或任务名称。");
  $$("[data-load-example]").forEach(b=>b.onclick=()=>loadExample(b.dataset.loadExample));
}
function applyDefaults(value){
  if(!value||!["rounds","patience"].every(k=>Number.isInteger(value[k])&&value[k]>=1&&value[k]<=5)||![5,10,20,30].includes(value.timeout))return;
  for(const [key,workbench] of [["rounds","max-rounds"],["patience","patience"],["timeout","timeout"]]){
    $("#default-"+key).value=String(value[key]);$("#gen-"+key).value=String(value[key]);
    if(!state.run)$("#"+workbench).value=String(value[key]);
  }
  renderRun();
}
async function refreshPortal(view){
  if(view==="generate")refreshGenerationStatus();
  if(view==="examples"){
    try{if(!officialExamples.length)officialExamples=await api("/api/examples");renderOfficialExamples();}catch(e){notice(e.message,true);}
  }
  if(view==="settings"){
    $("#settings-model").innerHTML=`<span class="outline-chip">${escapeHTML(state.config.model||"未配置模型")}</span><p class="form-help">${state.config.model_configured?"配置已读取；实际请求效果尚需联调。":"本实例尚未配置模型连接。"}</p>`;
  }
  if(view==="help"){
    try{const p=await api("/api/provenance");$("#upstream-summary").textContent=`main ${p.main.slice(0,12)} · WebDemo ${p["TraceCoder-WebDemo"].slice(0,12)} · 本系统 2.1.0 为本地整合版本。`;}catch(e){notice(e.message,true);}
  }
}
function initializePortal(){
  if(portalReady){refreshPortal(state.view);return;}portalReady=true;
  $("#gen-start").onchange=updateGenerationMode;
  $("#generation-form").addEventListener("input",()=>{if(savedGeneration&&savedGeneration.signature!==generationSignature())$("#gen-saved").textContent=`基于 v${savedGeneration.task.revision} 修改中；保存将创建新版本。`;});
  $("#generation-form").onsubmit=async e=>{e.preventDefault();$("#gen-save").disabled=true;try{await saveGeneration();notice("任务版本已保存，可在项目空间查看。");}catch(err){notice(err.message,true);}finally{refreshGenerationStatus();}};
  $("#gen-run").onclick=generateAndRepair;
  $("#gen-export").onclick=()=>{if($("#generation-form").reportValidity())download("tracecoder-generation.task.json",JSON.stringify(generationInput(),null,2),"application/json");};
  $("#example-search").oninput=renderOfficialExamples;$("#example-demo").onclick=()=>launchFixture("paper-sunday");
  $("#defaults-form").onsubmit=e=>{e.preventDefault();const value={rounds:Number($("#default-rounds").value),patience:Number($("#default-patience").value),timeout:Number($("#default-timeout").value)};try{localStorage.setItem("tracecoder.preferences.v1",JSON.stringify(value));applyDefaults(value);notice("默认参数已保存并应用到尚未启动的任务。");}catch{notice("浏览器未允许保存偏好，参数已应用于当前页面。",true);applyDefaults(value);}};
  $("#settings-check").onclick=async()=>{try{state.config=await api("/api/validate_config",{});refreshPortal("settings");notice("已读取服务器配置；未发起模型请求。");}catch(e){notice(e.message,true);}};
  try{applyDefaults(JSON.parse(localStorage.getItem("tracecoder.preferences.v1")));}catch{}
  updateGenerationMode();refreshPortal(state.view);
}
