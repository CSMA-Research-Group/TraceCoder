"use strict";
const pagePaths={overview:"/",generate:"/generate",workbench:"/workbench",projects:"/projects",runs:"/history",examples:"/examples",settings:"/settings",deployment:"/deployment",help:"/help"};
const viewPath=view=>pagePaths[view]||"/";
const pathView=path=>Object.keys(pagePaths).find(key=>pagePaths[key]===path)||"overview";
const statusNames={queued:"排队中",running:"运行中",completed:"已完成",error:"执行错误",cancelled:"已取消"};
const taskFields=["title","description","initial_code","test","entry_point","source_ref","start_mode"];
let systemTimer=null, searchTimer=null;
const formatTime=value=>new Date(value).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
const badge=value=>`<span class="status-pill ${escapeHTML(value)}">${escapeHTML(statusNames[value]||value)}</span>`;

function showView(view,hash=true){
  if(!["overview","generate","workbench","projects","runs","examples","settings","deployment","help"].includes(view))view="overview";
  state.view=view;
  $$(".system-view").forEach(node=>node.hidden=node.id!==`view-${view}`);
  $$(".primary-nav [data-view]").forEach(button=>{button.classList.toggle("active",button.dataset.view===view);button.setAttribute("aria-current",button.dataset.view===view?"page":"false");});
  if(hash)history.pushState(null,"",view==="workbench"&&state.run?`/history/${state.run.id}`:viewPath(view));
  if(typeof refreshPortal==="function")refreshPortal(view);
  window.scrollTo({top:0,behavior:"instant"});
  if(view==="overview")refreshOverview();
  if(view==="projects")refreshProjects();
  if(view==="runs")refreshRuns();
  if(view==="deployment")refreshDeployment();
}
function bindViews(){
  $$("[data-view]").forEach(button=>button.onclick=()=>{notice("");showView(button.dataset.view);});
}
function runRows(records,full=false){
  if(!records.length)return empty("还没有运行记录","从修复工作台启动案例，或在项目空间提交任务。结果会保存在本实例中。");
  return `<table class="${full?"runs-table":""}"><thead><tr>${full?"<th>对比</th>":""}<th>任务 / 运行</th><th>状态</th><th>执行模式</th><th>最优测试</th>${full?"<th>轮次 / 用时</th><th>操作</th>":""}</tr></thead><tbody>${records.map(r=>`<tr>${full?`<td><input type="checkbox" class="compare-checkbox" data-compare="${r.id}" aria-label="对比 ${escapeHTML(r.title)} ${r.id}" ${state.selectedRuns.has(r.id)?"checked":""}></td>`:""}<td><button class="row-title" data-open-run="${r.id}" title="${escapeHTML(r.title)}">${escapeHTML(r.title)}</button><small>${formatTime(r.created_at)}${r.revision?` · v${r.revision}`:""}</small>${full?`<small>${r.id}</small>`:""}</td><td>${badge(r.status)}${full?`<small>${escapeHTML(reasons[r.stop_reason]||names[r.stage]||"等待执行")}</small>`:""}</td><td><span class="mode-pill">${r.mode==="fixture"?"预设响应":"真实模型"}</span></td><td class="mono">${escapeHTML(r.score)}</td>${full?`<td class="mono">${r.rounds} <small>${r.elapsed_seconds??"—"} s</small></td><td><button class="text-button" data-open-run="${r.id}">查看 ↗</button>${["queued","running"].includes(r.status)?`<button class="text-button" data-cancel-run="${r.id}">停止</button>`:`<button class="text-button" data-review="${r.id}">评审包 ↓</button>`}</td>`:""}</tr>`).join("")}</tbody></table>`;
}
function bindRuns(container){
  container.querySelectorAll("[data-open-run]").forEach(b=>b.onclick=()=>loadRun(b.dataset.openRun));
  container.querySelectorAll("[data-review]").forEach(b=>b.onclick=()=>downloadAPI(`/api/runs/${b.dataset.review}/bundle`,`${b.dataset.review}-review.zip`));
  container.querySelectorAll("[data-cancel-run]").forEach(b=>b.onclick=async()=>{try{await api(`/api/runs/${b.dataset.cancelRun}/cancel`,{});await refreshRuns();notice("停止请求已提交，当前阶段结束后保留结果。");}catch(e){notice(e.message,true);}});
  container.querySelectorAll("[data-compare]").forEach(b=>b.onchange=()=>{
    if(b.checked&&state.selectedRuns.size===2){b.checked=false;notice("最多选择两条记录进行对比。");return;}
    b.checked?state.selectedRuns.add(b.dataset.compare):state.selectedRuns.delete(b.dataset.compare);updateCompareButton();
  });
}
function checkRows(config,compact=false){
  const checks=compact?config.checks.filter(c=>["service","model","sandbox"].includes(c.key)):config.checks;
  const labels={ready:"已就绪",configured:"已配置",missing:"待配置",local:"本机模式"};
  return checks.map(c=>`<div class="readiness-row"><span class="check-indicator ${c.status}">${c.status==="ready"?"✓":c.status==="configured"?"◇":"·"}</span><div><strong>${escapeHTML(c.title)}</strong><small>${escapeHTML(compact?(c.key==="model"?config.model:c.key==="sandbox"?(config.sandbox_ready?"执行镜像可用":"等待隔离执行环境"):"演示案例可直接运行"):c.detail)}</small></div><span class="check-state">${labels[c.status]}</span></div>`).join("");
}
async function refreshOverview(){
  try{
    const overview=await api("/api/overview");
    $("#overview-runs").textContent=overview.runs;
    $("#overview-mode-counts").textContent=`预设响应 ${overview.by_mode.fixture||0} · 真实模型 ${overview.by_mode.live||0}`;
    $("#overview-projects").innerHTML=`${overview.projects}<small> 个项目</small>`;
    $("#overview-task-count").textContent=`${overview.tasks} 个任务 · 输入版本独立保存`;
    $("#overview-active").innerHTML=`${overview.capacity.active}<small> / ${overview.capacity.workers} 并发</small>`;
    $("#overview-queue").textContent=`排队 ${overview.capacity.queued} · 等待上限 ${overview.capacity.queue_limit}`;
    $("#overview-benchmarks").textContent=Object.values(state.benchmarks).reduce((n,rows)=>n+rows.length,0).toLocaleString();
    $("#overview-recent").innerHTML=runRows(overview.recent);bindRuns($("#overview-recent"));
    if(state.config.checks)$("#overview-readiness").innerHTML=checkRows(state.config,true);
  }catch(e){notice(e.message,true);}
}
async function refreshRuns(){
  try{
    const params=new URLSearchParams({query:$("#run-search").value,mode:$("#run-filter-mode").value,status:$("#run-filter-status").value,limit:15,offset:state.runsOffset});
    const result=await api(`/api/runs/search?${params}`);
    $("#runs-table").innerHTML=runRows(result.items,true);bindRuns($("#runs-table"));
    $("#run-total").textContent=`共 ${result.total} 条 · ${result.total?result.offset+1:0}–${Math.min(result.offset+result.items.length,result.total)}`;
    $("#prev-runs").disabled=state.runsOffset===0;$("#next-runs").disabled=result.offset+result.limit>=result.total;updateCompareButton();
  }catch(e){notice(e.message,true);}
}
function updateCompareButton(){$("#compare-runs").disabled=state.selectedRuns.size!==2;$("#compare-runs").textContent=`对比所选 (${state.selectedRuns.size}/2)`;}
async function compareRuns(){
  try{
    const values=await Promise.all([...state.selectedRuns].map(id=>api(`/api/runs/${id}`)));
    const signature=r=>[r.hashes.tests_sha256,r.hashes.initial_code_sha256,r.hashes.description_sha256,r.start_mode,r.mode,JSON.stringify(r.config),JSON.stringify(r.implementation_sha256),r.model].join("|");
    const comparable=signature(values[0])===signature(values[1]);
    const rows=[["任务",r=>r.task.title||r.task.id],["运行 ID",r=>r.id],["模式",r=>r.mode==="fixture"?"预设响应 / 人工编写":"真实模型"],["状态",r=>statusNames[r.status]],["停止原因",r=>reasons[r.stop_reason]||r.stop_reason||"未结束"],["初始测试",r=>`${r.initial_eval?.passed_count??"—"} / ${r.initial_eval?.total_count??"—"}`],["最优测试",r=>`${r.final_eval?.passed_count??"—"} / ${r.final_eval?.total_count??"—"}`],["修复轮次",r=>r.rounds.length],["失败历史",r=>r.lessons.length],["回滚次数",r=>r.rounds.filter(x=>x.decision?.action==="rollback").length],["用时",r=>`${r.elapsed_seconds??"—"} s`],["测试摘要",r=>r.hashes.tests_sha256.slice(0,16)],["输入代码摘要",r=>(r.hashes.initial_code_sha256||"尚未生成").slice(0,16)]];
    modal("运行记录对比",`<p class="compare-note">${comparable?"两次运行的输入、测试、方法版本、模式、模型与预算一致；以下是记录对照，不能单凭两次运行推断修复性能。":"两次运行的输入、测试、方法版本、模式、模型或预算不同；仅供记录对照，不能作为修复效果的公平比较。"}</p><table class="compare-table"><thead><tr><th>维度</th><th>记录 A</th><th>记录 B</th></tr></thead><tbody>${rows.map(([label,get])=>`<tr><th>${label}</th>${values.map(v=>`<td>${escapeHTML(get(v))}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
  }catch(e){notice(e.message,true);}
}
async function downloadAPI(path,filename){
  try{
    const response=await fetch(path,{headers:state.token?{"X-TraceCoder-Token":state.token}:{}});
    if(!response.ok){let msg=await response.json();throw new Error(msg.error||"下载失败");}
    const url=URL.createObjectURL(await response.blob()),a=document.createElement("a");a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){notice(e.message,true);}
}
async function refreshProjects(selected=state.project?.id){
  try{
    const projects=await api("/api/projects");state.projects=projects;
    $("#new-project").disabled=!state.config.projects_enabled;
    $("#project-list").innerHTML=projects.length?projects.map(p=>`<button class="project-card ${p.id===selected?"active":""}" data-project="${p.id}"><span class="project-icon">▱</span><h3>${escapeHTML(p.name)}</h3><p>${escapeHTML(p.description||"待补充项目说明")}</p><span class="project-count">${p.task_count} 个任务</span><small>${p.id}</small></button>`).join(""):"";
    $$("[data-project]").forEach(b=>b.onclick=()=>refreshProjects(b.dataset.project));
    if(!projects.length){state.project=null;$("#project-detail").innerHTML=empty(state.config.projects_enabled?"建立第一个项目空间":"公开演示模式",state.config.projects_enabled?"创建项目，导入失败代码与固定测试。模型配置可以稍后完成，任务版本将独立保存。":"当前部署只开放固定案例。私有工作台可开启项目、版本管理与实时模型修复。");return;}
    if(!projects.some(p=>p.id===selected))selected=projects[0].id;
    state.project=await api(`/api/projects/${selected}`);
    $$("[data-project]").forEach(b=>b.classList.toggle("active",b.dataset.project===selected));
    renderProject();
  }catch(e){notice(e.message,true);}
}
function renderProject(){
  const p=state.project;
  $("#project-detail").innerHTML=`<div class="project-header"><div><h2>${escapeHTML(p.name)}</h2><p>${escapeHTML(p.description||"代码修复项目")}</p><small>${escapeHTML(p.source_ref||p.id)}</small></div><div><button id="import-task" class="secondary">导入 JSON ↑</button><button id="new-task" class="primary">新增任务 ＋</button><input id="task-upload" type="file" accept="application/json,.json" hidden></div></div><div>${p.tasks.length?p.tasks.map(t=>`<article class="project-task"><div class="project-task-head"><h3>${escapeHTML(t.title)}</h3><span class="outline-chip">v${t.revision}</span></div><p class="project-task-meta">${t.task_id} · ${t.sha256.slice(0,16)} · ${formatTime(t.created_at)}</p><div class="project-task-actions"><button class="primary" data-project-run="${t.task_id}" data-revision="${t.revision}">打开修复工作台 ↗</button><button class="text-button" data-edit-task="${t.task_id}">编辑新版本</button><button class="text-button" data-task-revisions="${t.task_id}">版本记录</button><button class="text-button" data-export-task="${t.task_id}">导出输入 ↓</button></div></article>`).join(""):empty("项目已创建，添加一个失败任务","可使用 API 请求窗口模板，或导入自己的单文件 Python 任务 JSON。")}</div>`;
  $("#new-task").onclick=()=>taskDialog();$("#import-task").onclick=()=>$("#task-upload").click();
  $("#task-upload").onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>250000)throw new Error("文件不能超过 250 KB");const value=JSON.parse(await file.text());if(!value||Array.isArray(value)||typeof value!=="object")throw new Error("需要 JSON 任务对象");if(Object.keys(value).some(k=>!taskFields.includes(k)))throw new Error("导入文件包含未定义字段，请使用任务模板或“导出输入”的 JSON");taskDialog(value,false);}catch(err){notice(err.message,true);}e.target.value="";};
  $$("[data-project-run]").forEach(b=>b.onclick=()=>openProjectTask(b.dataset.projectRun,Number(b.dataset.revision)));
  $$("[data-edit-task]").forEach(b=>b.onclick=async()=>{try{taskDialog(await api(`/api/project-tasks/${b.dataset.editTask}`),true);}catch(e){notice(e.message,true);}});
  $$("[data-task-revisions]").forEach(b=>b.onclick=()=>revisionDialog(b.dataset.taskRevisions));
  $$("[data-export-task]").forEach(b=>b.onclick=async()=>{try{const t=await api(`/api/project-tasks/${b.dataset.exportTask}`);download(`${t.id}-v${t.revision}.task.json`,JSON.stringify(Object.fromEntries(taskFields.map(k=>[k,t[k]||(k==="start_mode"?"frozen":"")])),null,2),"application/json");}catch(e){notice(e.message,true);}});
}
function projectDialog(){
  modal("创建项目空间",`<form id="project-form"><div class="form-field"><label for="project-name">项目名称</label><input id="project-name" maxlength="80" placeholder="例如 API 网关 / 业务规则服务" required></div><div class="form-field"><label for="project-description">项目说明</label><textarea id="project-description" maxlength="2000" placeholder="记录业务范围与当前需要修复的问题"></textarea></div><div class="form-field"><label for="project-source">来源标识（可选）</label><input id="project-source" maxlength="300" placeholder="仓库地址、模块名或工单编号"></div><p class="form-help">来源标识用于追溯，不会自动访问仓库或上传任何本地文件。</p><p class="form-error" id="form-error" role="alert"></p><div class="form-actions"><button class="primary" type="submit">创建项目 →</button></div></form>`);
  $("#project-form").onsubmit=async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;try{const p=await api("/api/projects",{name:$("#project-name").value,description:$("#project-description").value,source_ref:$("#project-source").value});$("#modal").close();await refreshProjects(p.id);}catch(err){$("#form-error").textContent=err.message;}finally{b.disabled=false;}};
}
function taskDialog(task={},editing=false){
  const projectId=state.project.id;
  modal(editing?`保存新版本 · 当前 v${task.revision}`:"新增修复任务",`<form id="task-form"><div class="form-grid"><div class="form-field"><label for="task-form-title">任务名称</label><input id="task-form-title" maxlength="120" required></div><div class="form-field"><label for="task-form-entry">入口函数（check 模式必填）</label><input id="task-form-entry" maxlength="120" placeholder="count_recent"></div></div><div class="form-field"><label for="task-form-start">起始方式</label><select id="task-form-start"><option value="frozen">修复已有失败代码</option><option value="generate">由模型生成初始实现</option></select></div><div class="form-field"><label for="task-form-description">需求与预期行为</label><textarea id="task-form-description" maxlength="16000" required></textarea></div><div class="form-grid"><div class="form-field"><label for="task-form-code">初始代码 · 生成模式留空</label><textarea class="code-input" id="task-form-code" spellcheck="false" maxlength="60000"></textarea></div><div class="form-field"><label for="task-form-test">固定测试 · assert / unittest / check(candidate)</label><textarea class="code-input" id="task-form-test" spellcheck="false" maxlength="60000" required></textarea></div></div><div class="form-field"><label for="task-form-source">来源版本 / 工单编号（可选）</label><input id="task-form-source" maxlength="300" placeholder="repository@commit:path/to/module.py / ISSUE-42"></div><p class="form-help">保存时进行语法与结构检查，不执行代码。每次保存新增不可变版本；已运行的输入与测试保持不变。</p><p id="form-error" class="form-error" role="alert"></p><div class="form-actions">${editing?"":'<button type="button" class="secondary" id="fill-template">填入 API 窗口模板</button>'}<button class="primary" type="submit">${editing?"保存新版本":"保存任务"} →</button></div></form>`);
  const inputs={start_mode:"start",title:"title",description:"description",entry_point:"entry",initial_code:"code",test:"test",source_ref:"source"};
  const fill=value=>Object.entries(inputs).forEach(([k,id])=>{$(`#task-form-${id}`).value=value[k]||(k==="start_mode"?"frozen":"");});fill(task);
  const toggleStart=()=>{$("#task-form-code").required=$("#task-form-start").value==="frozen";};toggleStart();$("#task-form-start").onchange=toggleStart;
  if(!editing)$("#fill-template").onclick=async()=>{try{const templates=await api("/api/templates");fill(templates[0]);toggleStart();}catch(e){$("#form-error").textContent=e.message;}};
  $("#task-form").onsubmit=async e=>{e.preventDefault();const button=e.submitter;button.disabled=true;try{const body={...Object.fromEntries(Object.entries(inputs).map(([k,id])=>[k,$(`#task-form-${id}`).value]))};await api(`/api/projects/${projectId}/tasks${editing?`/${task.id}`:""}`,body);$("#modal").close();await refreshProjects(projectId);}catch(err){$("#form-error").textContent=err.message;}finally{button.disabled=false;}};
}
async function revisionDialog(id){
  try{const revisions=await api(`/api/project-tasks/${id}/revisions`);modal("不可变的任务版本",`<p>选择版本可在工作台查看。已有运行始终引用提交时的版本。</p>${revisions.map(r=>`<button class="history-item" data-revision-open="${r.revision}"><span>v${r.revision} · ${escapeHTML(r.title)}<small>${formatTime(r.created_at)} · ${r.sha256.slice(0,20)}</small></span><span>打开 ↗</span></button>`).join("")}`);$$("[data-revision-open]").forEach(b=>b.onclick=()=>{$("#modal").close();openProjectTask(id,Number(b.dataset.revisionOpen));});}catch(e){notice(e.message,true);}
}
async function openProjectTask(id,revision){
  try{
    if(isRunning()){notice("当前工作台正在跟踪运行，可先从运行中心查看其他记录，再打开项目任务。",true);return;}
    const task=await api(`/api/project-tasks/${id}?revision=${revision}`);setMode("live");selectTask(task);$("#dataset").value="project";$("#start-mode").value=task.start_mode||"frozen";$("#start-mode").disabled=true;$("#benchmark-picker").hidden=false;$("#benchmark-task").innerHTML=`<option>${escapeHTML(task.title)} · v${task.revision}</option>`;$("#benchmark-task").disabled=true;$("#case-list").hidden=true;$(".case-heading").hidden=true;
    showView("workbench");renderTask();renderRun();
    if(!state.config.live_enabled)notice("任务版本已冻结。配置模型接口与独立执行环境后，即可对该版本启动真实修复；可在部署中心检查准备情况。");
  }catch(e){notice(e.message,true);}
}
async function refreshDeployment(){
  try{state.config=await api("/api/deployment");$("#deployment-checks").innerHTML=checkRows(state.config);$("#deployment-version").textContent=`v${state.config.version}`;renderRun();}catch(e){notice(e.message,true);}
}
async function initializeSystem(hash){
  bindViews();
  $(".brand").onclick=event=>{event.preventDefault();notice("");showView("overview");};
  $("#case-count").textContent=String(state.tasks.length).padStart(2,"0");
  $("#launch-demo").onclick=()=>launchFixture("paper-sunday");$("#scenario-api").onclick=()=>launchFixture("api-window");
  $("#new-project").onclick=projectDialog;$("#refresh-runs").onclick=refreshRuns;$("#refresh-deployment").onclick=refreshDeployment;
  $("#run-search").oninput=()=>{clearTimeout(searchTimer);state.runsOffset=0;searchTimer=setTimeout(refreshRuns,250);};
  ["run-filter-mode","run-filter-status"].forEach(id=>$("#"+id).onchange=()=>{state.runsOffset=0;refreshRuns();});
  $("#prev-runs").onclick=()=>{state.runsOffset=Math.max(0,state.runsOffset-15);refreshRuns();};$("#next-runs").onclick=()=>{state.runsOffset+=15;refreshRuns();};
  $("#compare-runs").onclick=compareRuns;
  $("#bundle-button").onclick=()=>downloadAPI(`/api/runs/${state.run.id}/bundle`,`${state.run.id}-review.zip`);
  $("#package-button").onclick=async()=>{try{const info=await api("/api/release");if(!info.available){notice(info.message,true);return;}downloadAPI("/api/release/download",info.filename);}catch(e){notice(e.message,true);}};
  $("#api-contract").onclick=()=>modal("API 接入契约",`<p>API 使用同一执行引擎。通过 <code>X-TraceCoder-Token</code> 提供服务器访问口令，模型密钥留在服务端。</p><table><tr><th>接口</th><th>用途</th></tr><tr><td>POST /api/projects</td><td>创建项目</td></tr><tr><td>POST /api/projects/{id}/tasks</td><td>保存任务版本</td></tr><tr><td>POST /api/runs</td><td>指定 task_id、revision 和 live 模式提交运行</td></tr><tr><td>GET /api/runs/{id}</td><td>查询状态与完整证据</td></tr><tr><td>POST /api/runs/{id}/cancel</td><td>取消排队或在当前阶段结束后停止</td></tr><tr><td>GET /api/runs/{id}/bundle</td><td>下载补丁、测试、报告与运行证据</td></tr></table><p>完整请求示例、退出码和配置说明已包含在代码包的 API 与部署文档中。CI 脚本只提交与取回结果，不自动应用补丁。</p>`);
  showView(hash.startsWith("#run=")||document.body.dataset.runId?"workbench":hash.replace("#","")||document.body.dataset.initialView||"overview",false);
  if(typeof initializePortal==="function")initializePortal();
  clearInterval(systemTimer);systemTimer=setInterval(()=>{if(document.hidden)return;if(state.view==="overview")refreshOverview();if(state.view==="runs")refreshRuns();},5000);
}
function launchFixture(id){
  if(isRunning()){showView("workbench");notice("当前运行正在执行，请完成或停止后再切换案例。");return;}
  setMode("fixture");selectTask(state.tasks.find(t=>t.id===id)||state.tasks[0]);showView("workbench");startRun();
}
window.addEventListener("popstate",()=>{const id=location.pathname.match(/^\/history\/([^/]+)$/)?.[1]||location.hash.match(/^#run=(.+)$/)?.[1];if(id)loadRun(id,false);else showView(location.hash.slice(1)||pathView(location.pathname),false);});
