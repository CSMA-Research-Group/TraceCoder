"use strict";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const state = {view:"overview", project:null, selectedRuns:new Set(), runsOffset:0, mode:"fixture", tasks:[], benchmarks:{}, task:null, run:null, config:{}, tab:"trace", codeTab:"initial", round:0, traceType:"trace", failuresOnly:false, token:"", polling:null, busy:false, historical:false};
const names = {generation:"初始生成", initial_test:"初始测试", instrumentation:"插桩", trace:"轨迹执行", analysis:"分析", repair:"修复", candidate_test:"候选测试", decision:"回滚决策", finished:"结束", queued:"排队", instrumentation_disabled:"插桩消融"};
const reasons = {all_tests_passed:"全部测试通过", direct_pass:"初始实现已通过", patience_exhausted:"达到连续无提升上限", round_budget_exhausted:"达到修复轮数上限", cancelled:"已停止", error:"执行中断", server_restarted:"进程已停止"};
const decisions = {accept:"接受提升", rollback:"回滚至最优", continue:"继续探索"};

async function api(path, body) {
  const headers = {};
  if (state.token) headers["X-TraceCoder-Token"] = state.token;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {method:body === undefined ? "GET" : "POST", headers, body:body === undefined ? undefined : JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401) accessDialog();
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  return result;
}
function notice(message, error=false) { $("#notice").hidden = !message; $("#notice").textContent = message; $("#notice").classList.toggle("error", error); }
function modal(title, content) { $("#modal-title").textContent = title; $("#modal-content").innerHTML = content; if (!$("#modal").open) $("#modal").showModal(); }
function empty(title, message) { return `<div class="empty-state"><div class="empty-symbol">⌁</div><strong>${escapeHTML(title)}</strong><p>${escapeHTML(message)}</p></div>`; }
function selectedRound() { return state.run?.rounds.find(round => round.number === state.round); }
function isRunning() { return state.run && ["queued","running"].includes(state.run.status); }

function renderCases() {
  $("#case-list").innerHTML = state.tasks.map((task,index) => `<button class="case ${state.task?.id===task.id?"active":""}" data-case="${escapeHTML(task.id)}" aria-pressed="${state.task?.id===task.id}"><strong><span class="case-icon">0${index+1}</span>${escapeHTML(task.title)}</strong><small>${escapeHTML(task.subtitle.split(" · ")[0])}<br>${escapeHTML(task.subtitle.split(" · ")[1] || "")}</small></button>`).join("");
  $$("[data-case]").forEach(button => { button.disabled = !!isRunning(); button.onclick = () => selectTask(state.tasks.find(task => task.id === button.dataset.case)); });
}
function selectTask(task) {
  state.task = task; state.run = null; state.round = 0; state.historical = false;
  state.codeTab = "initial"; state.failuresOnly = false;
  if(state.view==="workbench") history.replaceState(null,"","/workbench");
  notice(""); renderCases(); renderTask(); renderRun();
}
function renderTask() {
  const task = state.run?.task || state.task;
  if (!task) return;
  $("#task-title").textContent = task.title || task.id;
  $("#task-subtitle").textContent = task.subtitle || `${task.dataset || $("#dataset").value} · ${task.id}`;
  $("#task-tags").innerHTML = (task.tags || ["本地数据集", "固定测试", task.id]).map(tag=>`<span>${escapeHTML(tag)}</span>`).join("");
  $("#task-provenance").textContent = task.provenance;
  if(task.dataset==="project") $("#task-subtitle").textContent = `${task.project_name} · v${task.revision} · ${task.revision_sha256.slice(0,12)}`;
  $("#task-description").textContent = task.description;
  $("#code-editor").value = state.task?.initial_code || "";
  renderCode();
}
function setMode(mode) {
  if (isRunning()) return;
  state.mode=mode; state.run=null; state.round=0; state.historical=false;
  $("#fixture-mode").classList.toggle("active", mode==="fixture");
  $("#live-mode").classList.toggle("active", mode==="live");
  $("#fixture-mode").setAttribute("aria-pressed",mode==="fixture");
  $("#live-mode").setAttribute("aria-pressed",mode==="live");
  $("#live-controls").hidden = mode!=="live";
  $("#mode-note").textContent = mode==="fixture" ? "无需密钥。使用预设的智能体响应，每次都真实执行代码与测试。" : (state.config.live_enabled?"三个智能体共用已配置的模型，生成代码在独立 Docker 容器中执行。":"尚未配置模型或 Docker。可查看数据与编辑输入，配置就绪后才能运行。");
  if (mode==="live" && !state.config.live_enabled) notice("真实模型模式尚未就绪：需在服务器配置模型接口，并构建 Docker 执行镜像。当前可使用预设响应演示。"); else notice("");
  $("#dataset").value="demo"; $("#benchmark-picker").hidden=true; $("#case-list").hidden=false;$(".case-heading").hidden=false;
  if (!state.tasks.some(t=>t.id===state.task?.id)) state.task=state.tasks[0];
  renderTask(); renderRun(); renderCases();
}
function renderCode() {
  const round = selectedRound();
  const task = state.run?.task || state.task;
  let code="", filename="initial.py", caption="固定输入";
  if (state.codeTab==="initial") code=state.run?.initial_code ?? state.task?.initial_code ?? "";
  if (state.codeTab==="instrumented") { code=round?.instrumented_code || "# 运行后选择修复轮次，查看实际执行的插桩代码。"; filename="instrumented.py"; caption=round?.instrumentation_check?.enabled===false ? "未启用插桩 · 消融配置" : round?.instrumentation_check?.ast_preserved ? `${round.instrumentation_check.probe_count} 个探针 · AST 校验通过` : "等待插桩"; }
  if (state.codeTab==="final") { code=state.run?.final_code || "# 运行后显示历史最优实现。"; filename="best.py"; caption="历史最优测试通过数对应的实现"; }
  if (state.codeTab==="tests") { code=task?.test || ""; filename="fixed_tests.py"; caption="所有轮次使用同一测试"; }
  const editable=state.task?.dataset!=="project" && state.mode==="live" && state.codeTab==="initial" && !state.run && $("#start-mode").value==="frozen";
  $("#source-code").hidden=editable; $("#code-editor").hidden=!editable;
  $("#code-filename").textContent=filename;
  state.displayCode=code;
  $("#code-hash").textContent=caption;
  const highlight = line => line.split(/(#.*$|'[^']*'|"[^"]*"|\b(?:def|class|return|for|in|if|else|elif|import|from|as|True|False|None|while|not|and|or|try|except|assert|with|pass)\b|\b\d+\b)/g).map(token=>{
    const type=token.startsWith("#")?"comment":/^['"]/.test(token)?"string":/^\d+$/.test(token)?"number":/^(def|class|return|for|in|if|else|elif|import|from|as|True|False|None|while|not|and|or|try|except|assert|with|pass)$/.test(token)?"keyword":"";
    return type?`<span class="code-${type}">${escapeHTML(token)}</span>`:escapeHTML(token);
  }).join("");
  $("#source-code").innerHTML=code.split("\n").map((line,i)=>`<span class="code-line"><span class="line-number">${i+1}</span><span>${highlight(line)}</span></span>`).join("");
  $("#code-lines").textContent=`${code.split("\n").length} lines`;
  $$("[data-code]").forEach(b=>b.classList.toggle("active",b.dataset.code===state.codeTab));
}
function chart() {
  const run=state.run;
  if (!run?.initial_eval) return '<span>运行后显示每轮实际结果</span>';
  const attempts=[run.initial_eval.passed_count,...run.rounds.filter(r=>r.evaluation).map(r=>r.evaluation.valid?r.evaluation.passed_count:0)];
  const best=[]; let top=0; attempts.forEach(score=>{top=Math.max(top,score);best.push(top);});
  const total=run.initial_eval.total_count || 1;
  const coords = values => values.map((score,i)=>`${12+i*196/Math.max(1,attempts.length-1)},${37-score/total*30}`).join(" ");
  return `<svg viewBox="0 0 220 47" role="img" aria-label="候选通过数 ${attempts.join(', ')}；历史最优 ${best.join(', ')}"><path d="M10 38H210" stroke="#2a4056"/><polyline points="${coords(attempts)}" fill="none" stroke="#898ac9" stroke-width="1.7" stroke-dasharray="3 3"/><polyline points="${coords(best)}" fill="none" stroke="#4bdac8" stroke-width="2"/>${attempts.map((score,i)=>`<circle cx="${12+i*196/Math.max(1,attempts.length-1)}" cy="${37-score/total*30}" r="3" fill="#aaa1ec"/><text x="${12+i*196/Math.max(1,attempts.length-1)}" y="${Math.max(8,30-score/total*30)}" text-anchor="middle" fill="#a9b9cb" font-size="7">${score}</text>`).join("")}</svg>`;
}
function renderRun() {
  const run=state.run, running=!!isRunning();
  $("#run-button").disabled=state.busy || running || !state.task || (state.mode==="live"&&!state.config.live_enabled);
  $("#run-button span:first-child").textContent=running?"正在运行":state.mode==="fixture"?"开始演示":"运行 TraceCoder";
  $("#cancel-button").hidden=!running;
  ["fixture-mode","live-mode","max-rounds","patience","timeout","enable-instrumentation","dataset","benchmark-task","start-mode"].forEach(id=>$("#"+id).disabled=running);
  $$("[data-case]").forEach(b=>b.disabled=running);
  $("#evidence-label").textContent=(state.historical?"已保存记录 · ":"") + ((run?.mode||state.mode)==="fixture"?"预设智能体响应 · 实际代码执行":"真实模型调用 · 独立容器执行")+(run?.config.enable_instrumentation===false?" · 消融：未插桩":"");
  $("#run-status").textContent=run?(running?`${names[run.stage]||run.stage}进行中`:`${reasons[run.stop_reason]||run.status} · ${run.elapsed_seconds ?? "—"} s`):"准备就绪";
  const evaluation=run?.final_eval;
  $("#metric-tests").innerHTML=evaluation?`${evaluation.passed_count}<small> / ${evaluation.total_count}</small>`:'—<small> / —</small>';
  $("#metric-baseline").textContent=run?.initial_eval?`初始 ${run.initial_eval.passed_count}/${run.initial_eval.total_count} → 最优 ${evaluation?.passed_count??"—"}/${evaluation?.total_count??"—"}`:"等待初始测试";
  $("#metric-rounds").innerHTML=`${run?.rounds.length||0}<small> / ${run?.config.max_rounds||$("#max-rounds").value}</small>`;
  const rollbacks=run?.rounds.filter(r=>r.decision?.action==="rollback").length||0;
  $("#metric-decisions").textContent=run?`${rollbacks} 次回滚 · ${run.calls.length} 次${run.mode==="fixture"?"预设响应":"模型调用"}`:`最多 ${$("#max-rounds").value} 轮，连续 ${$("#patience").value} 轮无提升时停止`;
  $("#metric-lessons").innerHTML=`${run?.lessons.length||0}<small> 条</small>`;
  $("#lesson-badge").textContent=run?.lessons.length||0;
  $("#score-chart").innerHTML=chart();
  const seen=new Set(run?.events.map(e=>e.stage)||[]);
  $$("#pipeline>div").forEach(element=>{
    const stage=element.dataset.stage;
    const active=running&&(run.stage===stage || (stage==="decision"&&run.stage==="candidate_test"));
    element.classList.toggle("active",active); element.classList.toggle("done",seen.has(stage)&&!active);
  });
  $("#round-select").innerHTML='<option value="0">初始状态</option>'+(run?.rounds||[]).map(r=>`<option value="${r.number}">第 ${r.number} 轮${r.decision?" · "+decisions[r.decision.action]:""}</option>`).join("");
  $("#round-select").value=String(state.round);
  $("#export-button").disabled=!run;
  $("#download-code").disabled=!run?.final_code;
  $("#bundle-button").disabled=!run || running;
  if(state.task?.dataset==="project") {$("#start-mode").disabled=true;$("#benchmark-task").disabled=true;}
  $("#event-log").innerHTML=run?run.events.map(event=>`<div class="event-row"><span class="event-time">${escapeHTML(new Date(event.time).toLocaleTimeString("zh-CN",{hour12:false}))}</span><span class="event-stage">${escapeHTML(names[event.stage]||event.stage)}</span><span class="event-message">${event.round?`R${event.round} · `:""}${escapeHTML(reasons[event.message]||event.message)}${event.action?` · ${event.attempted_score} → best ${event.best_score}`:""}</span></div>`).join(""):'<p>启动后，逐步记录智能体调用、测试执行和回滚决策。</p>';
  if (running) $("#event-log").scrollTop=$("#event-log").scrollHeight;
  renderEvidence();renderCode();
}
function renderEvidence() {
  const run=state.run, round=selectedRound(), content=$("#evidence-content");
  $$("[data-tab]").forEach(button=>{button.classList.toggle("active",button.dataset.tab===state.tab);button.setAttribute("aria-selected",button.dataset.tab===state.tab);});
  if (!run) { content.innerHTML=empty("一次修复，一条可追溯的路径","选择左侧案例，点击“开始演示”，观察代码如何从失败走向修复。")+'<div class="empty-state"><p class="example-hint">推荐从论文案例开始：观察一次失败修复触发回滚，再看下一轮如何使用历史教训。</p></div>';return; }
  if (state.tab==="trace") {
    const evaluation=round?(state.traceType==="trace"?round.trace:round.evaluation):run.initial_eval;
    let toolbar=round?`<div class="evidence-toolbar"><span>${state.traceType==="trace"?"插桩轨迹 · 修复前基线":"候选验证 · 修复后结果"}</span><button class="text-button" id="toggle-trace">${state.traceType==="trace"?"查看候选测试 →":"← 查看插桩轨迹"}</button></div>`:"";
    if (!evaluation) {content.innerHTML=toolbar+empty("等待执行结果","本阶段尚未返回测试结果。");}
    else {
      const cases=evaluation.cases.filter(c=>!state.failuresOnly||c.status!=="passed").sort((a,b)=>Number(a.status==="passed")-Number(b.status==="passed"));
      toolbar+=`<div class="evidence-toolbar"><span class="result-label ${evaluation.passed?"":"warn"}">${evaluation.passed_count} / ${evaluation.total_count} 通过</span><label><input id="failures-only" type="checkbox" ${state.failuresOnly?"checked":""}>只看失败</label></div>`;
      const check=round?.instrumentation_check;
      if(state.traceType==="trace" && check) toolbar+=`<p class="call-meta">${check.probe_count} 个打印探针 · AST 保持 ${check.ast_preserved?"✓":"✕"} · 逐项结果保持 ${check.test_outcomes_preserved?"✓":"未通过"}</p>`;
      content.innerHTML=toolbar+(evaluation.error?`<div class="notice error">${escapeHTML(evaluation.error)}</div>`:"")+
        (cases.length?cases.slice(0,120).map((c,i)=>`<details class="test-card ${c.status==="passed"?"":"failed"}" ${c.status!=="passed"&&i<3?"open":""}><summary><span class="test-name" title="${escapeHTML(c.id)}">${escapeHTML(c.id.replace("tracecoder_tests.",""))}</span><span class="test-state">${c.status==="passed"?"✓ 通过":c.status==="skipped"?"跳过":"× 失败"}</span></summary>${c.stdout||c.stdout_truncated?`<div class="trace-label">STDOUT · 本次执行的打印输出${c.stdout_truncated?" · 已按预算截断":""}</div><pre>${escapeHTML(c.stdout||"[输出预算已耗尽]")}</pre>`:'<pre>此测试没有打印输出。</pre>'}${c.error?`<div class="trace-label">TEST FEEDBACK${c.error_truncated?" · 保留末 5000 字符":""}</div><pre class="error-trace">${escapeHTML(c.error)}</pre>`:""}</details>`).join(""):empty("当前筛选下没有测试","可取消“只看失败”，查看全部测试。"))+
        (cases.length>120?`<p class="call-meta">显示前 120 项，完整 ${cases.length} 项见导出记录。</p>`:"");
      $("#failures-only").onchange=event=>{state.failuresOnly=event.target.checked;renderEvidence();};
    }
    if($("#toggle-trace")) $("#toggle-trace").onclick=()=>{state.traceType=state.traceType==="trace"?"evaluation":"trace";renderEvidence();};
  } else if(state.tab==="analysis") {
    if (!round?.analysis) {content.innerHTML=empty("选择一个修复轮次","诊断需要结合该轮实际轨迹与历史失败记录。");return;}
    const a=round.analysis;
    content.innerHTML=`<div class="analysis-block"><h3>01 / 根因定位</h3><p>${escapeHTML(a.root_cause)}</p></div><div class="analysis-block"><h3>02 / 轨迹依据</h3><ul>${(a.evidence||[]).map(x=>`<li>${escapeHTML(x)}</li>`).join("")}</ul></div><div class="analysis-block"><h3>03 / 历史反思 · 使用 ${round.lesson_count_used} 条记录</h3><p>${escapeHTML(a.lesson_feedback)}</p></div><div class="analysis-block"><h3>04 / 修复计划</h3><ol>${a.repair_plan.map(x=>`<li>${escapeHTML(x)}</li>`).join("")}</ol></div><div class="analysis-block"><h3>05 / 下一轮插桩建议</h3><p>${escapeHTML(a.instrumentation_suggestions)}</p></div>`;
  } else if(state.tab==="diff") {
    content.innerHTML=round?.diff?`<div class="evidence-toolbar"><span>${escapeHTML(round.change_summary)}</span><span class="result-label ${round.decision?.action==="rollback"?"warn":""}">${decisions[round.decision?.action]||"验证中"}</span></div><pre class="diff-code">${round.diff.split("\n").map(line=>`<span class="${line.startsWith("+")?"add":line.startsWith("-")?"del":""}">${escapeHTML(line)}</span>`).join("")}</pre>`:empty("选择一个修复轮次","对比该轮修复前的实现和新候选；回滚候选也会保留。");
  } else if(state.tab==="lessons") {
    content.innerHTML=run.lessons.length?run.lessons.map(lesson=>`<article class="lesson-card"><div class="lesson-meta">ROUND ${lesson.round} · ${lesson.passed_count} 项通过 · ${decisions[lesson.decision]}</div><h3>保留失败方案，供后续分析使用</h3><p>${escapeHTML(lesson.repair_plan.root_cause)}</p><p><strong>尝试的计划：</strong>${escapeHTML(lesson.repair_plan.repair_plan.join(" "))}</p><p>已保存候选代码、修复计划、变更部分、通过数与错误反馈。</p><details><summary class="text-button">展开失败证据</summary><pre class="diff-code">${escapeHTML(JSON.stringify(lesson.error_feedback,null,2))}</pre></details></article>`).join(""):empty("尚无失败修复记录","初始失败不是修复教训；只有未通过全部测试的修复候选会写入 HLLM。");
  } else {
    const calls=state.round?run.calls.filter(c=>c.round===state.round):run.calls;
    content.innerHTML=`<p class="call-meta">${run.mode==="fixture"?"以下为教学用预设响应，无实时模型推理或 token 消耗。":"记录实际请求与响应；密钥不进入导出。"}</p>`+(calls.map(call=>`<details class="call-card"><summary>R${call.round} / ${names[call.role]} <span class="call-meta"> · ${escapeHTML(call.response_source||"error")}</span></summary><p class="call-meta">${escapeHTML(call.model)} · ${call.latency_seconds??"—"}s ${call.usage?"· "+escapeHTML(JSON.stringify(call.usage)):""}</p><strong>请求</strong><pre>${escapeHTML(JSON.stringify(call.messages,null,2))}</pre><strong>响应</strong><pre>${escapeHTML(call.raw_response||call.error||"")}</pre></details>`).join("")||empty("暂无调用","开始运行后会记录每个角色的输入和输出。"));
  }
}
async function startRun() {
  if(state.busy||isRunning()) return;
  state.busy=true;notice("");renderRun();
  const body={mode:state.mode,dataset:state.mode==="fixture"?"demo":$("#dataset").value,task_id:state.task.id,max_rounds:Number($("#max-rounds").value),patience:Number($("#patience").value),timeout:Number($("#timeout").value),enable_instrumentation:$("#enable-instrumentation").checked};
  if(state.mode==="live") {body.start_mode=$("#start-mode").value;if(state.task.dataset==="project"){body.dataset="project";body.revision=state.task.revision;body.start_mode=state.task.start_mode||"frozen";}else if(body.start_mode==="frozen") body.initial_code=$("#code-editor").value;}
  try {
    state.run=await api("/api/runs",body);state.round=0;state.historical=false;state.traceType="trace";state.codeTab="initial";
    history.pushState(null,"",`/history/${state.run.id}`);renderRun();pollRun(state.run.id);
  } catch(error) {notice(error.message,true);} finally {state.busy=false;renderRun();}
}
async function pollRun(id) {
  clearTimeout(state.polling);
  try {
    const run=await api(`/api/runs/${id}`);
    if(state.run?.id!==id)return;
    const previousLength=state.run.rounds.length;
    state.run=run;
    if(run.rounds.length>previousLength || (run.status==="completed"&&state.round===0)) state.round=run.rounds.at(-1)?.number||0;
    if(run.error)notice(run.error,true);
    renderRun();
    if(isRunning())state.polling=setTimeout(()=>pollRun(id),500);
  } catch(error){notice(`获取运行状态失败：${error.message}。可从“运行记录”重新打开。`,true);}
}
function download(filename, text, type) {const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function historyDialog() {
  try {
    const records=await api("/api/runs");
    modal("已保存的运行记录",'<p>打开记录只读取已保存证据，不重新调用模型或执行代码。</p>'+ (records.length?records.map(r=>`<button class="history-item" data-history="${r.id}"><span>${escapeHTML(r.title)}<small>${escapeHTML(new Date(r.created_at).toLocaleString("zh-CN"))} · ${r.mode==="fixture"?"预设响应":"实时模型"} · ${escapeHTML(r.status)}</small></span><span>${escapeHTML(r.score)} ↗</span></button>`).join(""):"<p>尚无运行记录。</p>"));
    $$("[data-history]").forEach(button=>{button.disabled=!!isRunning();button.onclick=()=>loadRun(button.dataset.history);});
  } catch(error){notice(error.message,true);}
}
async function loadRun(id,navigate=true) {
  try {
    const loaded=await api(`/api/runs/${encodeURIComponent(id)}`);
    // Loading evidence must detach from another run without cancelling it.
    state.run=null;setMode(loaded.mode);
    state.task={...loaded.task,initial_code:loaded.initial_code};
    state.run=loaded;state.round=loaded.rounds.at(-1)?.number||0;state.historical=true;state.codeTab="initial";
    const dataset=loaded.task.dataset||"demo";
    $("#dataset").value=dataset;$("#benchmark-picker").hidden=dataset==="demo";$("#case-list").hidden=dataset!=="demo";$(".case-heading").hidden=dataset!=="demo";
    if(dataset==="project"){$("#benchmark-task").innerHTML=`<option>${escapeHTML(loaded.task.title)} · v${loaded.task.revision}</option>`;}
    else if(dataset!=="demo"){$("#benchmark-task").innerHTML=(state.benchmarks[dataset]||[]).map(t=>`<option value="${escapeHTML(t.id)}">${escapeHTML(t.id)} · ${escapeHTML(t.title)}</option>`).join("");$("#benchmark-task").value=loaded.task.id;}
    $("#start-mode").value=loaded.start_mode||"frozen";
    $("#max-rounds").value=String(loaded.config.max_rounds);$("#patience").value=String(loaded.config.patience);$("#timeout").value=String(loaded.config.timeout);
    $("#enable-instrumentation").checked=loaded.config.enable_instrumentation!==false;
    $("#modal").close();if(navigate)history.pushState(null,"",`/history/${loaded.id}`);notice(loaded.error||"");renderCases();renderTask();renderRun();if(typeof showView==="function")showView("workbench",false);if(isRunning())pollRun(id);
  }catch(error){notice(error.message,true);}
}
function accessDialog() {
  modal("服务器访问口令",'<p>仅用于已开启访问保护的部署。口令只保存在当前页面内存中，不写入运行记录。</p><input id="access-input" type="password" autocomplete="off" aria-label="服务器访问口令"><button id="access-submit" class="primary">连接服务器</button>');
  $("#access-submit").onclick=()=>{state.token=$("#access-input").value;$("#modal").close();initialize();};
}
$("#about-button").onclick=()=>modal("方法与复现范围",`<p>本应用按论文第 3 节实现 TraceCoder 的三个角色：插桩智能体收集动态状态，分析智能体结合轨迹与失败历史制定计划，修复智能体生成候选，再由固定测试验证。</p><table><tr><th>论文机制</th><th>本应用行为</th></tr><tr><td>§3.2 插桩</td><td>只插入 print；验证 AST 和逐项测试状态保持。</td></tr><tr><td>§3.6 / Algorithm 1</td><td>记录每个未全通过候选的计划、代码、反馈和通过数。</td></tr><tr><td>§3.7 / Algorithm 2</td><td>超过最优则接受；持平则继续；低于最优且低于上一候选才回滚。</td></tr><tr><td>§4.2.4</td><td>同一模型、最多 5 轮；初始生成不提供测试或参考解。</td></tr></table><h3>两种运行模式</h3><p><strong>预设响应演示：</strong>智能体输出由人工编写，每次执行代码和测试，适合稳定展示方法。该模式不构成 LLM 性能实验。</p><p><strong>真实模型：</strong>配置兼容接口与 Docker 执行器后，运行同一套方法。模型只接收题面、当前实现和实际诊断反馈，参考解不进入请求。</p><h3>复现边界</h3><p>论文正文概述与 Algorithm 2 的持平分支存在表述差异，这里以算法为准。通过随附测试仅说明该测试集上的结果，未复现论文全量 Pass@1、成本或排名。完整映射和部署步骤见 <code>code/docs/PAPER_MAPPING.md</code> 与 <code>code/README.md</code>。</p>`);
$("#fixture-mode").onclick=()=>setMode("fixture");$("#live-mode").onclick=()=>setMode("live");
$("#run-button").onclick=startRun;$("#history-button").onclick=historyDialog;$("#close-modal").onclick=()=>$("#modal").close();$("#access-button").onclick=accessDialog;
$("#modal").addEventListener("click",event=>{if(event.target===$("#modal"))$("#modal").close();});
$("#round-select").onchange=event=>{state.round=Number(event.target.value);renderEvidence();renderCode();};
$$("[data-tab]").forEach(button=>button.onclick=()=>{state.tab=button.dataset.tab;renderEvidence();});
$$("[data-code]").forEach(button=>button.onclick=()=>{state.codeTab=button.dataset.code;renderCode();});
$("#start-mode").onchange=renderCode;["max-rounds","patience"].forEach(id=>$("#"+id).onchange=renderRun);
$("#cancel-button").onclick=async()=>{try{await api(`/api/runs/${state.run.id}/cancel`,{});notice("已请求停止，将在当前阶段结束后保留结果。");}catch(error){notice(error.message,true);}};
$("#copy-code").onclick=async()=>{try{const text=$("#code-editor").hidden?state.displayCode:$("#code-editor").value;await navigator.clipboard.writeText(text);$("#copy-code").textContent="已复制";setTimeout(()=>$("#copy-code").textContent="复制",1500);}catch{notice("浏览器不允许访问剪贴板，请直接选择代码复制。");}};
$("#export-button").onclick=async()=>{try{const run=await api(`/api/runs/${state.run.id}/export`);download(`${run.id}.json`,JSON.stringify(run,null,2),"application/json");}catch(error){notice(error.message,true);}};
$("#download-code").onclick=()=>download("tracecoder_best.py",state.run.final_code,"text/x-python");
$("#dataset").onchange=async()=>{
  const dataset=$("#dataset").value;
  if(dataset==="project"){showView("projects");return;}
  $("#benchmark-picker").hidden=dataset==="demo";$("#case-list").hidden=dataset!=="demo";$(".case-heading").hidden=dataset!=="demo";
  if(dataset==="demo"){selectTask(state.tasks[0]);return;}
  $("#benchmark-task").innerHTML=state.benchmarks[dataset].map(t=>`<option value="${escapeHTML(t.id)}">${escapeHTML(t.id)} · ${escapeHTML(t.title)}</option>`).join("");
  $("#start-mode").value="generate";await changeBenchmark();
};
async function changeBenchmark(){try{selectTask(await api(`/api/task?dataset=${encodeURIComponent($("#dataset").value)}&id=${encodeURIComponent($("#benchmark-task").value)}`));}catch(error){notice(error.message,true);}}
$("#benchmark-task").onchange=changeBenchmark;
async function initialize(){
  try{const initialHash=location.hash;const id=document.body.dataset.runId||location.hash.match(/^#run=([a-zA-Z0-9-]+)$/)?.[1];const [config,catalog]=await Promise.all([api("/api/config"),api("/api/tasks")]);state.config=config;state.tasks=catalog.demos;state.benchmarks=catalog.benchmarks;$("#connection-text").textContent="服务在线";selectTask(state.tasks[0]);if(id)await loadRun(id,false);if(typeof initializeSystem==="function")await initializeSystem(initialHash);}catch(error){$("#connection-text").textContent="连接未完成";notice(error.message,true);}
}
initialize();
