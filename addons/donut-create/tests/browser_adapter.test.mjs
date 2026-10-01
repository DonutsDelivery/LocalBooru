// Synthetic ComfyUI app/API contract fixture; no services or user data.
// Run: node --experimental-vm-modules --test addons/donut-create/tests/browser_adapter.test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

assert.equal(typeof vm.SourceTextModule, 'function', 'Run this fixture with node --experimental-vm-modules --test.');

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = fs.readFileSync(path.join(root, 'assets/donut-create.js'), 'utf8');
const preset = JSON.parse(fs.readFileSync(path.join(root, 'workflow.json'), 'utf8'));
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'runtime.json'), 'utf8'));
const catalog = JSON.parse(fs.readFileSync(path.join(root, 'model_sources.json'), 'utf8'));
const copy = value => JSON.parse(JSON.stringify(value));
const stock = {nodes:[{id:1,type:'KSampler',widgets_values:[42]}],links:[],extra:{}};
const draftKey = backend => 'dmc-create:v5:draft:' + backend;

function nodeAt(workflow, wanted) {
  for (const graph of [workflow, ...(workflow.definitions?.subgraphs || [])]) {
    const node = graph.nodes?.find(node => node.id === wanted);
    if (node) return node;
  }
}

function fixture({initial = {nodes:[],links:[]}, store = new Map(), backend = 'synthetic-device-a'} = {}) {
  const timers = [], events = new Map(), elements = new Map();
  function element(tag) {
    const result = {
      tag,children:[],textContent:'',className:'',
      classList:{toggle() {}}, setAttribute() {}, addEventListener() {},
      append(...children) { this.children.push(...children); },
      querySelector(selector) { return this.children.find(child => selector === '.' + child.className); },
    };
    Object.defineProperty(result, 'id', {set(value) { elements.set(value, result); }, get() { return [...elements].find(([, el]) => el === result)?.[0]; }});
    return result;
  }
  const document = {body:element('body'),createElement:element,getElementById:id=>elements.get(id),querySelector:()=>null,
    addEventListener(name, fn) { events.set(name, fn); },hidden:false};
  let current = copy(initial);
  const loaded = [], queued = [], fetched = [];
  const models = Object.fromEntries([...new Set(catalog.models.map(model => model.folder))].map(folder => [folder,catalog.models.filter(model => model.folder === folder).map(model => model.filename)]));
  const app = {
    canvas:{draw() {}}, registerExtension() {},
    graph:{get _nodes() { return current.nodes; },serialize() { return copy(current); }},
    async loadGraphData(data) { current = copy(data || stock); loaded.push(copy(current)); },
    async graphToPrompt() {
      const face = nodeAt(current,1125)?.widgets_values?.[0] || '';
      const scene = nodeAt(current,1126)?.widgets_values?.[0] || '';
      const negative = nodeAt(current,1127)?.widgets_values?.[0] || '';
      const primary = nodeAt(current,1122)?.widgets_values?.[0];
      const output = {
        'text:face':{class_type:'DF_Text_Box',inputs:{Text:face}},
        'text:scene':{class_type:'DF_Text_Box',inputs:{Text:scene}},
        'text:negative':{class_type:'DF_Text_Box',inputs:{Text:negative}},
        'expanded:face':{class_type:'DonutText',inputs:{text:['text:face',0]}},
        'expanded:scene':{class_type:'DonutText',inputs:{text:['text:scene',0]}},
        'condition':{class_type:'DonutPromptConditioning',inputs:{face:['expanded:face',0],scene:['expanded:scene',0],negative:['text:negative',0]}},
        'sample':{class_type:'DonutSampler',inputs:{positive:['condition',3],negative:['condition',5],seed:nodeAt(current,1137)?.widgets_values?.[0] || 42}},
      };
      if (primary) output.model={class_type:'UNETLoader',inputs:{unet_name:primary}};
      return {output,workflow:copy(current)};
    },
  };
  const api = {
    async fetchApi(route) {
      fetched.push(route);
      const value = route.startsWith('/dmc/workflow') ? {revision:0,workflow:null}
        : route === '/object_info' ? Object.fromEntries(manifest.required_nodes.map(node=>[node,{}])) : models[route.slice('/models/'.length)];
      return {ok:true,json:async()=>copy(value)};
    },
    async queuePrompt(...args) { queued.push(copy(args)); return {prompt_id:'synthetic-prompt-id'}; },
  };
  const context = vm.createContext({
    URL,JSON,Map,Set,Date,Promise,performance:{now:()=>0},
    console:{log() {},warn() {},error() {}}, document,app,LiteGraph:{NODE_TITLE_HEIGHT:30,registered_node_types:{DonutWorkflowPanel:function () {}}},
    location:{origin:'http://synthetic.test',host:'synthetic.test',pathname:'/api/create/studio/synthetic-session/',href:'http://synthetic.test/api/create/studio/synthetic-session/'},
    __DMC_CREATE__:{prefix:'/api/create/studio/synthetic-session/',sessionId:'synthetic-session',backendKey:backend,profile:'workflow'},
    __fixtureApi:api, __fixtureStock:stock,
    localStorage:{getItem:key=>store.get(key) || null,setItem:(key,value)=>store.set(key,String(value))},
    setInterval(fn, delay) { const timer={fn,delay,active:true}; timers.push(timer); return timer; },clearInterval(timer) { timer.active=false; },
    setTimeout(fn, delay) { const timer={fn,delay,active:true}; timers.push(timer); return timer; },requestAnimationFrame() {},
    addEventListener(name,fn) { events.set(name,fn); },
    fetch:async url=>({ok:new URL(url).pathname.endsWith('/workflow.json'),json:async()=>copy(preset)}),
  });
  context.window=context;
  const imports = new Map();
  async function importModule(specifier) {
    if (imports.has(specifier)) return imports.get(specifier);
    const file = new URL(specifier).pathname;
    const exported = file.endsWith('/app.js') ? 'export const app=globalThis.app;' : file.endsWith('/api.js') ? 'export const api=globalThis.__fixtureApi;' : 'export const defaultGraph=globalThis.__fixtureStock;';
    const module = new vm.SourceTextModule(exported,{context});
    await module.link(() => { throw new Error('Unexpected fixture import'); });
    await module.evaluate(); imports.set(specifier,module); return module;
  }
  const script = new vm.Script(source,{filename:'donut-create.js',importModuleDynamically:importModule});
  script.runInContext(context);
  async function settle() { for(let i=0;i<12;i++) await new Promise(resolve=>setImmediate(resolve)); }
  async function boot() {
    await settle();
    const waiter = timers.find(timer=>timer.delay===200 && timer.active);
    if (waiter) await waiter.fn();
    await settle();
  }
  function editPrompt(value) {
    const panel=nodeAt(current,1140);
    const control=panel.properties.donut_app_controls.groups[0].controls.find(control=>control.path.at(-1)===1126);
    assert.equal(control.widget,'Text');
    nodeAt(current,control.path.at(-1)).widgets_values[0]=value;
  }
  return {app,api,loaded,queued,fetched,models,store,context,events,boot,settle,editPrompt,
    current:()=>copy(current),notice:()=>elements.get('donut-create-status')?.querySelector('.message')?.textContent};
}

// AC: @donut-create-plugin ac-workflow-state
test('empty first studio loads genuine v5, scopes API and queues edited panel prompt',async()=>{
  const env=fixture(); await env.boot();
  assert.equal(env.current().nodes.length,25);
  assert.equal(env.api.api_base,'/api/create/studio/synthetic-session');
  env.editPrompt('A blue glass vase in afternoon light.');
  const prompt=await env.app.graphToPrompt();
  await env.api.queuePrompt(0,prompt);
  assert.equal(env.queued.length,1);
  assert.equal(env.queued[0][1].output['text:scene'].inputs.Text,'A blue glass vase in afternoon light.');
  assert.equal(nodeAt(env.queued[0][1].workflow,1126).widgets_values[0],'A blue glass vase in afternoon light.');
  assert.equal(nodeAt(JSON.parse(env.store.get(draftKey('synthetic-device-a'))),1126).widgets_values[0],'A blue glass vase in afternoon light.');
  assert.equal(env.queued[0][1].workflow.extra.dmc_output_destination,'preview');
  assert.equal(env.current().extra?.dmc_output_destination,undefined);
});

// AC: @donut-create-plugin ac-save-gallery
test('advanced queue uses the selected output destination without editing the saved graph',async()=>{
  const env=fixture(); await env.boot();
  const replies=[];
  const parent={postMessage(value){replies.push(value);}};
  env.context.parent=parent;
  env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',
    sessionId:'synthetic-session',requestId:'destination',action:'snapshot',payload:{outputDestination:'comfy'}}});
  await env.settle();
  assert.equal(replies.length,1);
  assert.equal(replies[0].error,undefined);
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.queued[0][1].workflow.extra.dmc_output_destination,'comfy');
  assert.equal(env.current().extra?.dmc_output_destination,undefined);
  assert.equal(JSON.parse(env.store.get(draftKey('synthetic-device-a'))).extra?.dmc_output_destination,undefined);
});

// AC: @donut-create-plugin ac-workflow-state
test('late bootstrap replaces stock graph with v5 and restores the backend draft on reopen',async()=>{
  const store=new Map(); const first=fixture({initial:stock,store}); await first.boot();
  assert.equal(first.current().nodes.length,25);
  first.editPrompt('A synthetic copper kettle on a shelf.');
  first.events.get('pagehide')();
  const reopened=fixture({initial:stock,store}); await reopened.boot();
  assert.equal(nodeAt(reopened.current(),1126).widgets_values[0],'A synthetic copper kettle on a shelf.');
});

// AC: @donut-create-plugin ac-workflow-state
test('meaningful existing workflow survives initial restore and future explicit loads',async()=>{
  const working=copy(preset); nodeAt(working,1126).widgets_values[0]='Synthetic working draft';
  const env=fixture({initial:working}); await env.boot();
  assert.equal(env.loaded.length,0);
  assert.equal(nodeAt(env.current(),1126).widgets_values[0],'Synthetic working draft');
  const uploaded=copy(preset); nodeAt(uploaded,1126).widgets_values[0]='Synthetic uploaded workflow';
  await env.app.loadGraphData(uploaded); env.events.get('pagehide')();
  assert.equal(nodeAt(JSON.parse(env.store.get(draftKey('synthetic-device-a'))),1126).widgets_values[0],'Synthetic uploaded workflow');
});

// AC: @donut-create-plugin ac-managed-setup
test('active missing model prevents actual queue request and produces visible error',async()=>{
  const env=fixture(); await env.boot();
  env.models.diffusion_models=[];
  await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/Missing models/);
  assert.equal(env.queued.length,0);
  assert.match(env.notice(),/Missing models/);
});

// AC: @donut-create-plugin ac-workflow-state
test('distinct device/backend keys isolate drafts even with identical managed origins',async()=>{
  const store=new Map(); const first=fixture({store,backend:'device-a-managed'}); await first.boot();
  first.editPrompt('Synthetic device A draft'); first.events.get('pagehide')();
  const other=fixture({initial:stock,store,backend:'device-b-managed'}); await other.boot();
  assert.notEqual(nodeAt(other.current(),1126).widgets_values[0],'Synthetic device A draft');
  const reopened=fixture({initial:stock,store,backend:'device-a-managed'}); await reopened.boot();
  assert.equal(nodeAt(reopened.current(),1126).widgets_values[0],'Synthetic device A draft');
});

// AC: @donut-create-plugin ac-mobile-shared-workspace
// AC: @donut-create-plugin ac-workflow-state
test('storage-only remote run does not advertise a changed workflow',async()=>{
 const env=fixture();await env.boot();const unchanged=env.current();unchanged.extra={...unchanged.extra,dmc_output_destination:'preview'};
 const fetch=env.api.fetchApi;env.api.fetchApi=async route=>route.startsWith('/dmc/workflow')?{ok:true,json:async()=>({revision:1,workflow:unchanged})}:fetch(route);
 const replies=[];const parent={postMessage:v=>replies.push(v)};env.context.parent=parent;
 env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'check',action:'snapshot',payload:{outputDestination:'preview'}}});await env.settle();
 assert.equal(replies[0].snapshot.latestAvailable,false);assert.doesNotMatch(env.notice()||'',/newer run/);
});

// AC: @donut-create-plugin ac-mobile-shared-workspace
// AC: @donut-create-plugin ac-workflow-state
test('storage-only remote run preserves an unsaved local draft without notice',async()=>{
 const env=fixture();await env.boot();const unchanged=env.current();unchanged.extra={...unchanged.extra,dmc_output_destination:'comfy'};env.editPrompt('Keep my unsaved synthetic violet vase.');
 const fetch=env.api.fetchApi;env.api.fetchApi=async route=>route.startsWith('/dmc/workflow')?{ok:true,json:async()=>({revision:1,workflow:unchanged})}:fetch(route);
 const replies=[];const parent={postMessage:v=>replies.push(v)};env.context.parent=parent;
 env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'check',action:'snapshot',payload:{outputDestination:'preview'}}});await env.settle();
 assert.equal(replies[0].snapshot.latestAvailable,false);assert.doesNotMatch(env.notice()||'',/newer run/);assert.equal(nodeAt(env.current(),1126).widgets_values[0],'Keep my unsaved synthetic violet vase.');
});
// AC: @donut-create-plugin ac-mobile-shared-workspace
// AC: @donut-create-plugin ac-workflow-state
test('changed external prompt still shows notice and preserves the local draft',async()=>{
 const env=fixture();await env.boot();const changed=env.current();nodeAt(changed,1126).widgets_values[0]='Different external synthetic red vase.';changed.extra={...changed.extra,dmc_output_destination:'preview'};env.editPrompt('Keep my unsaved synthetic violet vase.');
 const fetch=env.api.fetchApi;env.api.fetchApi=async route=>route.startsWith('/dmc/workflow')?{ok:true,json:async()=>({revision:1,workflow:changed})}:fetch(route);
 const replies=[];const parent={postMessage:v=>replies.push(v)};env.context.parent=parent;
 env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'check',action:'snapshot',payload:{outputDestination:'preview'}}});await env.settle();
 assert.equal(replies[0].snapshot.latestAvailable,true);assert.match(env.notice()||'',/newer run/);assert.equal(nodeAt(env.current(),1126).widgets_values[0],'Keep my unsaved synthetic violet vase.');
});


test('queue validation uses the already compiled prompt once', async () => {
  const env = fixture(); await env.boot();
  const compiled = await env.app.graphToPrompt();
  env.fetched.length = 0;
  env.app.graphToPrompt = async () => { throw new Error('Unexpected recompilation'); };
  await env.api.queuePrompt(0, compiled);
  assert.equal(env.queued.length, 1);
  assert.equal(env.fetched.filter(route => route === '/object_info').length, 1);
});
