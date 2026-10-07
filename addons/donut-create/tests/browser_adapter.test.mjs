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

function fixture({initial = {nodes:[],links:[]}, store = new Map(), backend = 'synthetic-device-a', livePrompts = false, failFace = false, failFaceUpgrade = false} = {}) {
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
  function liveNodes() {
    if (!livePrompts) return current.nodes;
    function build(graph) {
      return graph.nodes.map(node => {
        const subgraph = current.definitions?.subgraphs?.find(item => item.id === node.type);
        if (subgraph) return { ...node, subgraph: { _nodes: build(subgraph) } };
        if (node.type !== 'DF_Text_Box') return node;
        return { ...node, widgets: [{ name: 'Text',
          get value() { return node.widgets_values[0]; },
          set value(value) { node.widgets_values[0] = value; },
          callback() { if (node.properties?.dmc_prompt_role === 'face' && failFace) throw new Error('Synthetic face callback failure'); },
        }] };
      });
    }
    return build(current);
  }
  const app = {
    canvas:{draw() {}}, registerExtension() {},
    graph:{get _nodes() { return liveNodes(); },serialize() { return copy(current); }},
    async loadGraphData(data) { current = copy(data || stock); loaded.push(copy(current)); if (failFaceUpgrade && nodeAt(current,1180)) return false; },
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
    fetch:async url=>{ fetched.push(String(url)); return {ok:new URL(url).pathname.endsWith('/workflow.json'),json:async()=>copy(preset)}; },
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

// AC: @create-managed-prompt-enhancer ac-apply-routing
test('enhancement patches both prompts atomically, rejects stale fields, and never queues', async () => {
  const env = fixture({livePrompts:true}); await env.boot();
  const replies = [], parent = {postMessage:value=>replies.push(value)};
  env.context.parent = parent;
  async function action(action, payload) {
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'enhancer-'+replies.length,action,payload}});
    await env.settle(); return replies.at(-1);
  }
  const original = (await action('snapshot')).snapshot.fields;
  const expected = {prompt:original.prompt.value, facePrompt:original.facePrompt.value};
  const fields = {prompt:'A synthetic owl gazes left.',facePrompt:'The owl gazes left.'};
  const changed = await action('enhance-prompts', {fields,expected});
  assert.equal(changed.ok,true);
  assert.equal(changed.snapshot.fields.prompt.value,fields.prompt);
  assert.equal(changed.snapshot.fields.facePrompt.value,fields.facePrompt);
  const stale = await action('enhance-prompts', {fields:{prompt:'Wrong scene',facePrompt:''},expected});
  assert.equal(stale.ok,false);
  assert.match(stale.error,/prompts changed/);
  assert.equal(nodeAt(env.current(),1126).widgets_values[0],fields.prompt);
  assert.equal(env.queued.length,0);
});

// AC: @create-managed-prompt-enhancer ac-apply-routing
test('failed face mutation rolls back both fields', async () => {
  const env = fixture({livePrompts:true,failFace:true}); await env.boot();
  const replies = [], parent = {postMessage:value=>replies.push(value)};
  env.context.parent = parent;
  async function action(action,payload) {
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'failure-'+replies.length,action,payload}});
    await env.settle(); return replies.at(-1);
  }
  const original = (await action('snapshot')).snapshot.fields;
  const response = await action('enhance-prompts',{fields:{prompt:'Synthetic new scene',facePrompt:'Synthetic new face'},expected:{prompt:original.prompt.value,facePrompt:original.facePrompt.value}});
  assert.equal(response.ok,false);
  assert.equal(nodeAt(env.current(),1126).widgets_values[0],original.prompt.value);
  assert.equal(nodeAt(env.current(),1178).widgets_values[0],original.facePrompt.value);
  assert.equal(env.queued.length,0);
});

// Equivalent public v5 structure before the face-only branch was introduced.
function legacyFaceRecipe() {
  const workflow = copy(preset);
  delete workflow.extra.donut_workflow.controls_revision;
  const sources = workflow.definitions.subgraphs.find(graph => graph.id === nodeAt(workflow,1138).type);
  const body = workflow.definitions.subgraphs.find(graph => graph.id === nodeAt(workflow,1014).type);
  const removed = new Set([1178,1179,1180]);
  sources.nodes = sources.nodes.filter(node => !removed.has(node.id));
  body.nodes = body.nodes.filter(node => !removed.has(node.id));
  sources.links = sources.links.filter(edge => !removed.has(edge.origin_id) && !removed.has(edge.target_id));
  body.links = body.links.filter(edge => !removed.has(edge.target_id));
  for (const edge of body.links.filter(edge => edge.origin_id === 1180)) {
    edge.origin_id = 996; edge.origin_slot = edge.type === 'STRING' ? 1 : 4;
    nodeAt(workflow,996).outputs[edge.origin_slot].links.push(edge.id);
  }
  const removedLinks = new Set([1115413,1115414,1115415,1115417,1115418,1115419,1115420,1115421,1115422,1115423,1115424,1115425]);
  for (const graph of [sources,body]) {
    for (const node of graph.nodes) for (const output of node.outputs || []) output.links = (output.links || []).filter(id => !removedLinks.has(id));
    for (const port of [...graph.inputs,...graph.outputs]) port.linkIds = (port.linkIds || []).filter(id => !removedLinks.has(id));
  }
  sources.outputs = sources.outputs.filter(port => port.name !== 'face_instruction');
  const faceSlot = body.inputs.findIndex(port => port.name === 'face_instruction');
  body.inputs.splice(faceSlot,1);
  for (const edge of body.links) if (edge.origin_id === -10 && edge.origin_slot > faceSlot) edge.origin_slot--;
  nodeAt(workflow,1138).outputs = nodeAt(workflow,1138).outputs.filter(port => port.name !== 'face_instruction');
  nodeAt(workflow,1014).inputs = nodeAt(workflow,1014).inputs.filter(port => port.name !== 'face_instruction');
  workflow.links = workflow.links.filter(edge => edge[0] !== 1115416);
  for (const group of nodeAt(workflow,1140).properties.donut_app_controls.groups) group.controls = group.controls.filter(control => control.path.at(-1) !== 1178);
  nodeAt(workflow,1126).widgets_values[0] = 'A synthetic owl on a branch.';
  nodeAt(workflow,1125).widgets_values[0] = 'Soft blue light, ink illustration.';
  nodeAt(workflow,984).mode = 0;
  return workflow;
}

function faceInstruction(workflow) {
  const body = workflow.definitions.subgraphs.find(graph => graph.id === nodeAt(workflow,1014).type);
  const sources = workflow.definitions.subgraphs.find(graph => graph.id === nodeAt(workflow,1138).type);
  const target = body.links.find(edge => edge.target_id === 984 && nodeAt(workflow,984).inputs[edge.target_slot].name === 'edit_prompt');
  assert.equal(target.origin_id,1180);
  const positive = body.links.find(edge => edge.target_id === 1118 && nodeAt(workflow,1118).inputs[edge.target_slot].name === 'conditioning_in_2');
  assert.equal(positive.origin_id,1180); assert.equal(positive.origin_slot,3);
  const context = body.links.find(edge => edge.target_id === 1180 && edge.target_slot === 1);
  assert.equal(context.origin_id,996); assert.equal(context.origin_slot,0);
  const face = body.links.find(edge => edge.target_id === 1180 && edge.target_slot === 2);
  assert.equal(body.inputs[face.origin_slot].name,'face_instruction');
  const root = workflow.links.find(edge => edge[3] === 1014 && nodeAt(workflow,1014).inputs[edge[4]].name === 'face_instruction');
  assert.equal(root[1],1138);
  assert.equal(nodeAt(workflow,1138).outputs[root[2]].name,'face_instruction');
  const output = sources.outputs.find(port => port.name === 'face_instruction');
  const expansion = sources.links.find(edge => edge.id === output.linkIds[0]);
  assert.equal(expansion.origin_id,1179);
  const text = sources.links.find(edge => edge.target_id === 1179 && edge.target_slot === 2);
  return nodeAt(workflow,text.origin_id).widgets_values[0];
}

// AC: @create-managed-prompt-enhancer ac-apply-routing
// AC: @donut-create-plugin ac-workflow-state
test('legacy face detail gains an editable, wired face prompt without replacing the recipe or generating',async()=>{
  const working = legacyFaceRecipe(), before = copy(working);
  nodeAt(working,1137).widgets_values[0] = 8675309;
  const env = fixture({initial:working,livePrompts:true}); await env.boot();
  assert.ok(nodeAt(env.current(),1178),'Face prompt must be added to the existing recipe');
  assert.equal(faceInstruction(env.current()),'');
  for (const id of [1126,1125,1137,1014,984]) assert.deepEqual(nodeAt(env.current(),id).widgets_values,nodeAt(working,id).widgets_values);
  assert.equal(nodeAt(env.current(),984).mode,0);
  assert.equal(env.current().nodes.length,before.nodes.length);
  assert.equal(env.queued.length,0);
  const replies = [], parent = {postMessage:value=>replies.push(value)}; env.context.parent = parent;
  async function action(action,payload) {
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'legacy-'+replies.length,action,payload}});
    await env.settle(); return replies.at(-1);
  }
  const snapshot = (await action('snapshot')).snapshot;
  assert.equal(snapshot.fields.facePrompt.available,true);
  assert.equal(snapshot.fields.faceDetail.available,true);
  assert.equal(snapshot.fields.faceDetail.value,true);
  const response = await action('enhance-prompts',{fields:{prompt:'A synthetic owl gazes left.',facePrompt:'The owl gazes left.'},expected:{prompt:snapshot.fields.prompt.value,facePrompt:''}});
  assert.equal(response.ok,true);
  assert.equal(faceInstruction(env.current()),'The owl gazes left.');
  assert.equal(nodeAt(env.current(),1125).widgets_values[0],nodeAt(working,1125).widgets_values[0]);
  const reopened = fixture({initial:env.current(),livePrompts:true}); await reopened.boot();
  assert.equal(reopened.loaded.length,0,'Upgrade must be idempotent');
  assert.equal(faceInstruction(reopened.current()),'The owl gazes left.');
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('explicitly loading an older recipe also upgrades only its missing face path',async()=>{
  const env = fixture({livePrompts:true}); await env.boot();
  const working = legacyFaceRecipe();
  await env.app.loadGraphData(working);
  assert.equal(faceInstruction(env.current()),'');
  assert.equal(nodeAt(env.current(),1126).widgets_values[0],nodeAt(working,1126).widgets_values[0]);
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('custom face wiring is preserved instead of being silently replaced',async()=>{
  const working = legacyFaceRecipe();
  const body = working.definitions.subgraphs.find(graph => graph.id === nodeAt(working,1014).type);
  body.links.find(edge => edge.target_id === 984 && edge.target_slot === 11).origin_id = 777;
  const env = fixture({initial:working}); await env.boot();
  assert.deepEqual(env.current(),working);
  assert.equal(env.loaded.length,0);
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('failed face-path load restores the complete original recipe',async()=>{
  const working = legacyFaceRecipe();
  const env = fixture({initial:working,failFaceUpgrade:true}); await env.boot();
  assert.deepEqual(env.current(),working);
  assert.match(env.notice(),/face prompt could not be added/);
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('face upgrade allocates fresh links and preserves unrelated custom nodes and settings',async()=>{
  const working = legacyFaceRecipe();
  working.nodes.push({id:8000,type:'SyntheticCustomNode',widgets_values:['Keep this setting'],inputs:[],outputs:[]});
  working.last_link_id = 2000000;
  const env = fixture({initial:working}); await env.boot();
  const upgraded = env.current();
  assert.deepEqual(nodeAt(upgraded,8000),nodeAt(working,8000));
  const graphs = [upgraded,...upgraded.definitions.subgraphs];
  const oldLinks = new Set([working,...working.definitions.subgraphs].flatMap(graph => graph.links.map(edge => Array.isArray(edge) ? edge[0] : edge.id)));
  const added = graphs.flatMap(graph => graph.links.map(edge => Array.isArray(edge) ? edge[0] : edge.id)).filter(id => !oldLinks.has(id));
  assert.equal(new Set(added).size,added.length);
  assert.ok(added.every(id => id > 2000000));
  assert.equal(faceInstruction(upgraded),'');
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
// AC: @create-managed-prompt-enhancer ac-apply-routing
test('legacy face repair recognizes executable wiring when release metadata is absent',async()=>{
  const working = legacyFaceRecipe();
  delete working.extra.donut_workflow;
  const env = fixture({initial:working,livePrompts:true}); await env.boot();
  assert.equal(faceInstruction(env.current()),'');
  assert.equal(nodeAt(env.current(),1126).widgets_values[0],nodeAt(working,1126).widgets_values[0]);
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('face repair follows control paths and connections in a renumbered copy of v5',async()=>{
  const working = legacyFaceRecipe();
  const graphs = [working,...working.definitions.subgraphs];
  const remap = id => id >= 0 ? id + 5000 : id;
  for (const graph of graphs) {
    for (const node of graph.nodes) {
      node.id = remap(node.id);
      for (const group of node.properties?.donut_app_controls?.groups || [])
        for (const control of group.controls || []) control.path = control.path.map(remap);
    }
    for (const edge of graph.links) {
      if (Array.isArray(edge)) { edge[1] = remap(edge[1]); edge[3] = remap(edge[3]); }
      else { edge.origin_id = remap(edge.origin_id); edge.target_id = remap(edge.target_id); }
    }
  }
  const subgraphIds = new Map(working.definitions.subgraphs.map(graph => [graph.id,'synthetic-copy-'+graph.id]));
  for (const graph of graphs) {
    if (subgraphIds.has(graph.id)) graph.id = subgraphIds.get(graph.id);
    for (const node of graph.nodes) if (subgraphIds.has(node.type)) node.type = subgraphIds.get(node.type);
  }
  const env = fixture({initial:working,livePrompts:true}); await env.boot();
  const upgraded = env.current();
  const face = upgraded.definitions.subgraphs.flatMap(g=>g.nodes).find(n=>n.properties?.dmc_prompt_role==='face' && n.type==='DF_Text_Box');
  assert.ok(face,'Copied workflows must gain their own face text field');
  const detail = nodeAt(upgraded,5984);
  const body = upgraded.definitions.subgraphs.find(g=>g.nodes.includes(detail));
  const edge = body.links.find(e=>e.target_id===5984 && detail.inputs[e.target_slot].name==='edit_prompt');
  assert.equal(nodeAt(upgraded,edge.origin_id).properties.dmc_prompt_role,'face');
  assert.equal(nodeAt(upgraded,6126).widgets_values[0],nodeAt(working,6126).widgets_values[0]);
  const replies = [], parent = {postMessage:value=>replies.push(value)}; env.context.parent = parent;
  async function action(action,payload) {
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'copy-'+replies.length,action,payload}});
    await env.settle(); return replies.at(-1);
  }
  const snapshot = (await action('snapshot')).snapshot;
  assert.equal(snapshot.fields.facePrompt.available,true);
  assert.equal(snapshot.fields.faceDetail.available,true);
  assert.equal(snapshot.fields.faceDetail.value,true);
  const response = await action('enhance-prompts',{fields:{prompt:'A synthetic owl.',facePrompt:'The owl gazes left.'},expected:{prompt:snapshot.fields.prompt.value,facePrompt:''}});
  assert.equal(response.ok,true);
  assert.equal(nodeAt(env.current(),face.id).widgets_values[0],'The owl gazes left.');
  const reopened = fixture({initial:env.current(),livePrompts:true}); await reopened.boot();
  assert.equal(reopened.loaded.length,0);
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('face repair allocates new node IDs when a custom node already occupies the old reserved face ID',async()=>{
  const working = legacyFaceRecipe();
  working.nodes.push({id:1178,type:'SyntheticCustomNode',widgets_values:['Keep this setting'],inputs:[],outputs:[]});
  const env = fixture({initial:working,livePrompts:true}); await env.boot();
  const upgraded = env.current();
  assert.deepEqual(nodeAt(upgraded,1178),nodeAt(working,1178));
  const face = upgraded.definitions.subgraphs.flatMap(graph=>graph.nodes).find(node=>node.type==='DF_Text_Box' && node.properties?.dmc_prompt_role==='face');
  assert.ok(face);
  assert.notEqual(face.id,1178);
  assert.equal(new Set([upgraded,...upgraded.definitions.subgraphs].flatMap(graph=>graph.nodes.map(node=>node.id))).size,
    [upgraded,...upgraded.definitions.subgraphs].reduce((count,graph)=>count+graph.nodes.length,0));
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-basic-studio
test('explicit preset reset fetches the selected bundled preset and exposes both prompt controls',async()=>{
  const working = legacyFaceRecipe();
  const env = fixture({initial:working,livePrompts:true}); await env.boot();
  const replies=[],parent={postMessage:value=>replies.push(value)};env.context.parent=parent;
  env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'connect-reset',action:'snapshot'}});
  await env.settle();
  env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'reset',action:'load-preset',payload:{preset:'base'}}});
  await env.settle();
  assert.equal(replies.at(-1).ok,true);
  assert.ok(env.fetched.some(url=>url.endsWith('/workflow.json?preset=base')));
  assert.equal(replies.at(-1).snapshot.fields.facePrompt.available,true);
  assert.equal(faceInstruction(env.current()),'');
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-workflow-state
test('failed explicit preset reset rolls back the entire working draft without generating',async()=>{
  const working = copy(preset); nodeAt(working,1126).widgets_values[0]='A synthetic saved owl.';
  const env = fixture({initial:working,livePrompts:true,failFaceUpgrade:true}); await env.boot();
  const replies=[],parent={postMessage:value=>replies.push(value)};env.context.parent=parent;
  async function action(action,payload) {
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'rollback-'+replies.length,action,payload}});
    await env.settle();return replies.at(-1);
  }
  await action('snapshot');
  const reply = await action('load-preset',{preset:'base'});
  assert.equal(reply.ok,false);
  assert.match(reply.error,/setup workflow could not be loaded/);
  assert.deepEqual(env.current(),working);
  assert.equal(env.queued.length,0);
});
