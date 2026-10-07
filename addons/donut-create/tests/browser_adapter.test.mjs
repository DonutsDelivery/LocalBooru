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

const compatibilityPresets = ['Off', 'Custom', 'Bypass 2', 'Bypass 3', 'Rebalance', 'Enhancer', 'Rebalance + Enhancer',
  'Rebalance + Bypass 2', 'Rebalance + Bypass 3', 'Balanced', 'Balanced + raw NAG', 'Balanced + Enhancer', 'UncensorFix'];
const fusionWidgets = ['compatibility_preset', 'tap_method', 'tap_profile', 'per_layer_weights', 'tap_strength',
  'tap_formula', 'tap_normalization', 'projector_method', 'projector_profile', 'projector_layer_weights',
  'projector_strength', 'projector_formula', 'projector_normalization', 'fusion_method', 'fusion_strength',
  'ui_mode', 'uncensorfix_controls', 'uncensorfix_target', 'uncensorfix_strength', 'nag_match_taps',
  'nag_raw_negative', 'nag_text_energy_compensation', 'nag_batch_txtfusion'];

function fixture({initial = {nodes:[],links:[]}, store = new Map(), backend = 'synthetic-device-a', livePrompts = false, liveFusion = false, failPreset = false, failFace = false, failFaceUpgrade = false} = {}) {
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
  let cachedLive;
  const loaded = [], queued = [], fetched = [];
  const models = Object.fromEntries([...new Set(catalog.models.map(model => model.folder))].map(folder => [folder,catalog.models.filter(model => model.folder === folder).map(model => model.filename)]));
  function liveNodes() {
    if (!livePrompts && !liveFusion) return current.nodes;
    if (liveFusion && cachedLive) return cachedLive;
    function build(graph) {
      return graph.nodes.map(node => {
        const subgraph = current.definitions?.subgraphs?.find(item => item.id === node.type);
        const live = {...node};
        if (subgraph) live.subgraph = {_nodes:build(subgraph)};
        if (liveFusion && [1014,1118].includes(node.id)) {
          const names = node.id === 1118 ? fusionWidgets : node.inputs.filter(input=>input.widget).map(input=>input.name);
          const item = name => live.widgets.find(widget=>widget.name===name);
          live.widgets = names.map((name,index)=>({name,
            get value() { return node.widgets_values[index]; },
            set value(value) { node.widgets_values[index]=value; },
            options:name==='compatibility_preset'?{values:compatibilityPresets}
              :name==='uncensorfix_controls'?{values:['Fusion only','Fusion + UncensorFix weights','UncensorFix weights only']}
              :{min:0,max:10,step:0.01},
            callback(value) {
              if (name==='compatibility_preset') {
                const inner = node.id===1118 ? live : live.subgraph._nodes.find(child=>child.id===1118);
                const control = name=>inner.widgets.find(widget=>widget.name===name);
                control('tap_strength').value = 1;
                control('tap_method').value = 'Synthetic preset '+value;
                control('uncensorfix_controls').value = value==='UncensorFix'?'Fusion + UncensorFix weights':'Fusion only';
                control('uncensorfix_strength').value = 1;
                // Preset extensions schedule sibling updates, as well as doing synchronous work.
                queueMicrotask(()=>{control('uncensorfix_strength').value=1;});
                if (node.id===1118 && failPreset) throw new Error('Synthetic preset callback failure');
              } else if (['uncensorfix_controls','uncensorfix_strength'].includes(name)) {
                item('compatibility_preset').value='Custom';
              }
            },
          }));
          return live;
        }
        if (subgraph) return live;
        if (node.type === 'DonutEditStudio') return {...node, widgets: node.properties.donut_widget_order.map((name,index)=>({name,
          get value() {return node.widgets_values[index];},set value(value) {node.widgets_values[index]=value;}}))};
        if (node.type !== 'DF_Text_Box') return node;
        return { ...node, widgets: [{ name: 'Text',
          get value() { return node.widgets_values[0]; },
          set value(value) { node.widgets_values[0] = value; },
          callback() { if (node.properties?.dmc_prompt_role === 'face' && failFace) throw new Error('Synthetic face callback failure'); },
        }] };
      });
    }
    const result=build(current);
    if (liveFusion) cachedLive=result;
    return result;
  }
  const app = {
    canvas:{draw() {}}, registerExtension() {},
    graph:{get extra() { return current.extra; }, set extra(value) { current.extra=value; },get _nodes() { return liveNodes(); },serialize() { return copy(current); }},
    async loadGraphData(data) { current = copy(data || stock); cachedLive=null; loaded.push(copy(current)); if (failFaceUpgrade && nodeAt(current,1180)) return false; },
    async graphToPrompt() {
      const face = nodeAt(current,1178)?.widgets_values?.[0] || '';
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
        'sample':{class_type:'DonutSampler',inputs:{positive:['condition',3],negative:['condition',5],seed:nodeAt(current,1137)?.widgets_values?.[0] ?? 42}},
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
    URL,JSON,Map,Set,Date,Promise,AbortController,performance:{now:()=>0},
    console:{log() {},warn() {},error() {}}, document,app,LiteGraph:{NODE_TITLE_HEIGHT:30,registered_node_types:{DonutWorkflowPanel:function () {}}},
    location:{origin:'http://synthetic.test',host:'synthetic.test',pathname:'/api/create/studio/synthetic-session/',href:'http://synthetic.test/api/create/studio/synthetic-session/'},
    __DMC_CREATE__:{prefix:'/api/create/studio/synthetic-session/',sessionId:'synthetic-session',backendKey:backend,profile:'workflow'},
    __fixtureApi:api, __fixtureStock:stock, __fixtureTimers:timers,
    localStorage:{getItem:key=>store.get(key) || null,setItem:(key,value)=>store.set(key,String(value))},
    setInterval(fn, delay) { const timer={fn,delay,active:true}; timers.push(timer); return timer; },clearInterval(timer) { timer.active=false; },
    setTimeout(fn, delay) { const timer={fn,delay,active:true}; timers.push(timer); return timer; },clearTimeout(timer) { timer.active=false; },requestAnimationFrame() {},
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
    editSeed:value=>{nodeAt(current,1137).widgets_values[0]=value;},
    editFace:value=>{nodeAt(current,1178).widgets_values[0]=value;},
    editFaceDetail:value=>{nodeAt(current,984).mode=value?0:4;},
    editMode:value=>{nodeAt(current,881).widgets_values[0]=value;nodeAt(current,881).widgets_values[1]=value?'donutref:'+'a'.repeat(64):'';},
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
  assert.equal(target.origin_slot,1);
  assert.equal(positive.origin_id,1180); assert.equal(positive.origin_slot,4);
  const context = body.links.find(edge => edge.target_id === 1180 && edge.target_slot === 2);
  assert.equal(context.origin_id,996); assert.equal(context.origin_slot,0);
  const face = body.links.find(edge => edge.target_id === 1180 && edge.target_slot === 1);
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

async function fusionBridge(options = {}) {
  const working=copy(preset), fusion=nodeAt(working,1118);
  fusion.widgets_values[fusionWidgets.indexOf('uncensorfix_controls')]=options.composition || 'Fusion only';
  fusion.widgets_values[fusionWidgets.indexOf('uncensorfix_strength')]=2.37;
  const env=fixture({initial:working,liveFusion:true,...options}); await env.boot();
  const replies=[],parent={postMessage:value=>replies.push(value)};env.context.parent=parent;
  const action=async(action,payload)=>{
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'fusion-'+replies.length,action,payload}});
    await env.settle();return replies.at(-1);
  };
  await action('snapshot');
  return {env,action};
}

// AC: @donut-create-plugin ac-basic-advanced-sync
// AC: @donut-create-plugin ac-basic-model-effects
test('every compatibility preset preserves raw decensor composition and weight through callbacks and microtasks',async()=>{
  for (const composition of ['Fusion only','Fusion + UncensorFix weights','UncensorFix weights only']) {
    const {env,action}=await fusionBridge({composition});
    for (const compatibilityPreset of compatibilityPresets) {
      const reply=await action('patch',{compatibilityPreset});
      assert.equal(reply.ok,true,reply.error);
      const fusion=nodeAt(env.current(),1118), value=name=>fusion.widgets_values[fusionWidgets.indexOf(name)];
      assert.equal(value('uncensorfix_controls'),composition,compatibilityPreset);
      assert.equal(value('uncensorfix_strength'),2.37,compatibilityPreset);
      assert.equal(value('compatibility_preset'),compatibilityPreset);
      assert.equal(reply.snapshot.fields.compatibilityPreset.value,compatibilityPreset);
      assert.equal(reply.snapshot.fields.decensor.value,composition!=='Fusion only');
      assert.equal(reply.snapshot.fields.decensorWeight.value,2.37);
      assert.equal(value('tap_method'),'Synthetic preset '+compatibilityPreset);
      if (!['Off','Custom'].includes(compatibilityPreset)) assert.equal(reply.snapshot.fields.tapStrength.value,1);
    }
    assert.equal(env.queued.length,0);
  }
});

// AC: @donut-create-plugin ac-basic-advanced-sync
test('explicit decensor and weight edits win within the compatibility patch independently of payload order',async()=>{
  for (const decensor of [true,false]) {
    const {env,action}=await fusionBridge({composition:decensor?'Fusion only':'Fusion + UncensorFix weights'});
    for (const patch of [
      {compatibilityPreset:'Balanced',decensor},
      {decensorWeight:3.14,compatibilityPreset:'UncensorFix'},
      {decensorWeight:0.61,decensor,compatibilityPreset:'Rebalance',tapStrength:0.73},
    ]) {
      const reply=await action('patch',patch);
      assert.equal(reply.ok,true,reply.error);
      assert.equal(reply.snapshot.fields.decensor.value,decensor);
      assert.equal(reply.snapshot.fields.decensorWeight.value,patch.decensorWeight??2.37);
      if (patch.tapStrength!==undefined) assert.equal(reply.snapshot.fields.tapStrength.value,patch.tapStrength);
    }
    assert.equal(env.queued.length,0);
  }
});

// AC: @donut-create-plugin ac-workflow-state
test('preset callback failure rolls back the entire workflow and saved draft after its microtasks',async()=>{
  const {env,action}=await fusionBridge({composition:'Fusion + UncensorFix weights',failPreset:true});
  const before=env.current(),saved=env.store.get(draftKey('synthetic-device-a'));
  const reply=await action('patch',{compatibilityPreset:'Bypass 3',decensor:false,decensorWeight:0.42});
  assert.equal(reply.ok,false);
  assert.match(reply.error,/Synthetic preset callback failure/);
  assert.deepEqual(env.current(),before);
  assert.equal(env.store.get(draftKey('synthetic-device-a')),saved);
  assert.equal(env.queued.length,0);
});

// AC: @donut-create-plugin ac-basic-advanced-sync
test('Advanced preset callbacks preserve Decensor in both UI modes and after workflow recreation',async()=>{
  for (const composition of ['Fusion only','Fusion + UncensorFix weights','UncensorFix weights only']) {
    const {env,action}=await fusionBridge({composition});
    for (const uiMode of ['Simple','Advanced']) {
      const inner=()=>env.app.graph._nodes.find(node=>node.id===1014).subgraph._nodes.find(node=>node.id===1118);
      const widget=name=>inner().widgets.find(widget=>widget.name===name);
      widget('ui_mode').value=uiMode;
      for (const name of compatibilityPresets) {
        widget('compatibility_preset').value=name;
        widget('compatibility_preset').callback(name);
        await env.settle();
        assert.equal(widget('uncensorfix_controls').value,composition);
        assert.equal(widget('uncensorfix_strength').value,2.37);
        assert.equal(widget('compatibility_preset').value,name);
        assert.equal((await action('snapshot')).snapshot.fields.decensorWeight.value,2.37);
      }
      await env.app.loadGraphData(env.current());
      widget('compatibility_preset').value='Balanced';
      widget('compatibility_preset').callback('Balanced');
      await env.settle();
      assert.equal(widget('uncensorfix_controls').value,composition);
      assert.equal(widget('uncensorfix_strength').value,2.37);
      assert.equal(widget('compatibility_preset').value,'Balanced');
    }
  }
});

// AC: @donut-create-plugin ac-basic-advanced-sync
test('explicit Advanced Decensor edits after selecting a preset win over queued preset work',async()=>{
  const {env}=await fusionBridge();
  const inner=env.app.graph._nodes.find(node=>node.id===1014).subgraph._nodes.find(node=>node.id===1118);
  const widget=name=>inner.widgets.find(widget=>widget.name===name);
  widget('compatibility_preset').value='Rebalance';
  widget('compatibility_preset').callback('Rebalance');
  widget('uncensorfix_controls').value='Fusion + UncensorFix weights';
  widget('uncensorfix_controls').callback(widget('uncensorfix_controls').value);
  widget('uncensorfix_strength').value=0.42;
  widget('uncensorfix_strength').callback(0.42);
  await env.settle();
  assert.equal(widget('uncensorfix_controls').value,'Fusion + UncensorFix weights');
  assert.equal(widget('uncensorfix_strength').value,0.42);
});

async function enhancerFixture(options = {}) {
  const env=fixture({livePrompts:true, ...options}); await env.boot();
  const replies=[], parent={postMessage:value=>replies.push(value)}; env.context.parent=parent;
  env.action=async (action,payload)=>{
    env.events.get('message')({source:parent,origin:'http://parent.test',data:{channel:'donut-create-basic-v1',sessionId:'synthetic-session',requestId:'run-'+replies.length,action,payload}});
    await env.settle(); return replies.at(-1);
  };
  env.requests=[]; env.cancels=[];
  const fetch=env.api.fetchApi;
  env.api.fetchApi=async (route,options)=>{
    if (route==='/dmc/enhancer/cancel') {env.cancels.push(JSON.parse(options.body)); return {ok:true,json:async()=>({})};}
    if (route!=='/dmc/enhancer') return fetch(route,options);
    let job;
    if (options?.method==='POST') {
      const request=JSON.parse(options.body); env.requests.push(request);
      if (env.onEnhance) await env.onEnhance(request);
      job={id:'owned-'+env.requests.length,kind:'enhance',model:request.model,seed:request.seed,state:'complete',
        result:{main_prompt:'Scene for '+request.seed,face_prompt:'Face for '+request.seed}};
      if (env.jobChange) job=env.jobChange(job);
      env.job=job;
    }
    return {ok:true,json:async()=>({supported:true,busy:false,models:[{id:'qwen',ready:true}],job:env.job})};
  };
  await env.action('snapshot');
  return env;
}

// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('per-run enhancer configuration persists without queuing and rejects malformed configuration',async()=>{
  const env=await enhancerFixture();
  assert.equal((await env.action('snapshot')).snapshot.enhancer.enabled,false);
  assert.equal((await env.action('enhancer-config',{enabled:true,model:'qwen'})).snapshot.mutationChanged,true);
  assert.deepEqual(env.current().extra.dmc_prompt_enhancer,{enabled:true,model:'qwen'});
  assert.equal(env.queued.length,0);
  const invalid=await env.action('enhancer-config',{enabled:'true',model:'qwen'});
  assert.equal(invalid.ok,false);
  assert.equal((await env.action('enhancer-config',{enabled:true,model:'qwen'})).snapshot.mutationChanged,false);
});

// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('Basic, Edit, Run Instant and Advanced each enhance original briefs with compiled execution seed',async()=>{
  const env=await enhancerFixture();
  await env.action('enhancer-config',{enabled:true,model:'qwen'});
  env.editPrompt('Original synthetic scene');env.editFace('Original synthetic face');
  const before=env.current();
  env.app.queuePrompt=async ()=>env.api.queuePrompt(0,await env.app.graphToPrompt());
  // Basic, Edit and Run Instant share the host generate bridge; Advanced invokes the API directly.
  for (const mode of ['Basic','Edit','Run Instant','Advanced']) {
    env.editMode(mode==='Edit');
    if (mode==='Advanced') await env.api.queuePrompt(0,await env.app.graphToPrompt());
    else assert.equal((await env.action('generate')).ok,true,mode);
  }
  env.editMode(false);
  assert.equal(env.requests.length,4);
  assert.ok(env.requests.every(request=>request.face_brief==='Original synthetic face'));
  assert.ok(env.requests.every(request=>request.image_brief==='Original synthetic scene'));
  assert.deepEqual(env.current(),before);
  for (let i=0;i<4;i++) {
    const compiled=env.queued[i][1], request=env.requests[i];
    assert.equal(compiled.output.sample.inputs.seed,request.seed);
    assert.equal(compiled.output['text:scene'].inputs.Text,'Scene for '+request.seed);
    assert.equal(compiled.output['text:face'].inputs.Text,'Face for '+request.seed);
    assert.equal(nodeAt(compiled.workflow,1126).widgets_values[0],'Original synthetic scene');
    assert.equal(compiled.workflow.extra.dmc_prompt_enhancement.seed,request.seed);
  }
  assert.equal((await env.action('snapshot')).snapshot.enhancer.lastRun.main_prompt,'Scene for '+env.requests[0].seed);
  const changed=await env.app.graphToPrompt(); changed.output.sample.inputs.seed=0;
  // Simulate an authoritative compiled seed that differs from the displayed seed, as custom beforeQueued hooks can do.
  const compile=env.app.graphToPrompt;
  env.app.graphToPrompt=async ()=>{const value=await compile();value.output.sample.inputs.seed=0;return value;};
  await env.api.queuePrompt(0,changed);
  assert.equal(env.requests.at(-1).seed,0);
  assert.equal(env.queued.at(-1)[1].output.sample.inputs.seed,0);
  assert.notEqual(env.queued.at(-1)[1].output['text:scene'].inputs.Text,env.queued[0][1].output['text:scene'].inputs.Text);
  assert.deepEqual(env.current(),before);
});

// AC: @create-managed-prompt-enhancer ac-per-run-seed
// AC: @create-managed-prompt-enhancer ac-owned-private
test('per-run failures, malformed output and compilation errors restore source and submit nothing',async()=>{
  for (const failure of ['malformed','compile','seed','unavailable']) {
    const env=await enhancerFixture(); await env.action('enhancer-config',{enabled:true,model:'qwen'});
    const compiled=await env.app.graphToPrompt(), before=env.current();
    if (failure==='malformed') env.jobChange=job=>({...job,result:{main_prompt:'',face_prompt:''}});
    if (failure==='compile') env.app.graphToPrompt=async ()=>{throw new Error('Synthetic compilation failure');};
    if (failure==='seed') compiled.output.sample.inputs.seed=['custom',0];
    if (failure==='unavailable') env.api.fetchApi=async route=>route==='/dmc/enhancer'?{ok:true,json:async()=>({models:[{id:'qwen',ready:false}]})}:{ok:true,json:async()=>Object.fromEntries(manifest.required_nodes.map(node=>[node,{}]))};
    await assert.rejects(env.api.queuePrompt(0,compiled));
    assert.equal(env.queued.length,0,failure); assert.deepEqual(env.current(),before,failure);
  }
});

// AC: @create-managed-prompt-enhancer ac-per-run-seed
// AC: @create-managed-prompt-enhancer ac-owned-private
test('cancellation, concurrent calls and edits during enhancement cannot queue stale briefs',async()=>{
  for (const mode of ['cancel','edit','concurrent']) {
    const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
    let release;env.onEnhance=()=>new Promise(resolve=>{release=resolve;});
    const compiled=await env.app.graphToPrompt();
    const pending=env.api.queuePrompt(0,compiled);const rejected=assert.rejects(pending);
    await env.settle();
    if (mode==='cancel') assert.equal((await env.action('cancel-generation')).ok,true);
    else if (mode==='edit') env.editPrompt('New authored scene');
    else {await assert.rejects(env.api.queuePrompt(0,compiled),/already|progress|busy/i);await env.action('cancel-generation');}
    release();await rejected;await env.settle();
    assert.equal(env.queued.length,0);
    if (mode==='edit') assert.equal(nodeAt(env.current(),1126).widgets_values[0],'New authored scene');
    else assert.deepEqual(env.cancels,[{id:'owned-1'}]);
  }
});


// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('pinned beforeQueued/afterQueued seed timing and SeedPlan links use the executed seed on every run',async()=>{
  const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
  const compile=env.app.graphToPrompt;
  env.app.graphToPrompt=async ()=>{
    const p=await compile();
    p.output.shared={class_type:'SeedNode',inputs:{seed:p.output.sample.inputs.seed}};
    p.output.plan={class_type:'DonutSeedPlan',inputs:{text_seed:['shared',0],sampler_seed:['shared',0],filename_seed:['shared',0]}};
    p.output.sample.inputs.seed=['plan',1]; return p;
  };
  let nextSeed=0;
  env.app.queuePrompt=async ()=>{
    env.editSeed(nextSeed); // pinned frontend beforeQueued runs before graphToPrompt
    const result=await env.api.queuePrompt(0,await env.app.graphToPrompt());
    env.editSeed(++nextSeed); // pinned afterQueued affects only the next run
    return result;
  };
  for (let i=0;i<3;i++) assert.equal((await env.action('generate')).ok,true);
  assert.deepEqual(env.requests.map(request=>request.seed),[0,1,2]);
  assert.deepEqual(env.queued.map(args=>args[1].output.shared.inputs.seed),[0,1,2]);
  assert.deepEqual(env.queued.map(args=>args[1].workflow.extra.dmc_prompt_enhancement.seed),[0,1,2]);
  assert.equal(nodeAt(env.current(),1137).widgets_values[0],3);
  assert.ok(env.requests.every(request=>request.image_brief===nodeAt(env.current(),1126).widgets_values[0]));
});

// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('saved per-run metadata reopens with original briefs and never compounds generated output',async()=>{
  const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  const saved=copy(env.queued[0][1].workflow);
  await env.app.loadGraphData(saved);
  assert.deepEqual(copy((await env.action('snapshot')).snapshot.enhancer.lastRun),saved.extra.dmc_prompt_enhancement);
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.requests[1].image_brief,env.requests[0].image_brief);
  assert.equal(env.requests[1].face_brief,env.requests[0].face_brief);
  assert.equal(env.requests[1].seed,env.requests[0].seed);
});

// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('execution changes on recompilation and cancellation during capability checks fail closed',async()=>{
  for (const mode of ['compile-seed','validation-cancel','pagehide','timeout']) {
    const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
    const compiled=await env.app.graphToPrompt();
    let release;
    if (mode==='compile-seed') {
      const compile=env.app.graphToPrompt;
      env.app.graphToPrompt=async ()=>{const p=await compile();p.output.sample.inputs.seed++;return p;};
    } else if (mode==='validation-cancel') {
      const fetch=env.api.fetchApi;
      env.api.fetchApi=async (route,options)=>{if(route==='/object_info') await new Promise(resolve=>{release=resolve;});return fetch(route,options);};
    } else env.onEnhance=()=>new Promise(resolve=>{release=resolve;});
    const pending=env.api.queuePrompt(0,compiled), rejected=assert.rejects(pending);
    await env.settle();
    if (mode==='validation-cancel') await env.action('cancel-generation');
    if (mode==='pagehide') env.events.get('pagehide')();
    if (mode==='timeout') env.context.__fixtureTimers.find(timer=>timer.delay===9*60*1000).fn();
    release?.();await rejected;await env.settle();assert.equal(env.queued.length,0,mode);
    if (mode==='validation-cancel') assert.equal(env.requests.length,0);
  }
});


// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('cancel during prompt compilation restores temporary text immediately and discards its late result',async()=>{
  const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
  const compiled=await env.app.graphToPrompt(), before=env.current();let release;
  env.app.graphToPrompt=()=>new Promise(resolve=>{release=resolve;});
  const pending=env.api.queuePrompt(0,compiled), rejected=assert.rejects(pending);
  await env.settle();assert.notEqual(nodeAt(env.current(),1126).widgets_values[0],nodeAt(before,1126).widgets_values[0]);
  await env.action('cancel-generation');await rejected;
  assert.deepEqual(env.current(),before);release(compiled);await env.settle();assert.equal(env.queued.length,0);
});


// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('polls only its own seeded job and accepts a deliberately blank generated face prompt',async()=>{
  const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
  env.jobChange=job=>({...job,state:'running',result:{...job.result,face_prompt:''}});
  const pending=env.api.queuePrompt(0,await env.app.graphToPrompt());await env.settle();
  assert.equal(env.queued.length,0);env.job.state='complete';
  env.context.__fixtureTimers.find(timer=>timer.delay===500 && timer.active).fn();
  await pending;assert.equal(env.queued[0][1].output['text:face'].inputs.Text,'');
});

// AC: @create-managed-prompt-enhancer ac-per-run-seed
// AC: @create-managed-prompt-enhancer ac-owned-private
test('unbound custom controls and wrong job identity/model/seed never submit image work',async()=>{
  for (const mode of ['unbound','identity','model','seed','error']) {
    const env=await enhancerFixture({livePrompts:mode!=='unbound'});await env.action('enhancer-config',{enabled:true,model:'qwen'});
    if (mode==='identity') env.jobChange=job=>({...job,kind:'install'});
    if (mode==='model') env.jobChange=job=>({...job,model:'minicpm'});
    if (mode==='seed') env.jobChange=job=>({...job,seed:job.seed+1});
    if (mode==='error') env.jobChange=job=>({...job,state:'error'});
    await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()));assert.equal(env.queued.length,0,mode);
    if (mode==='unbound') assert.equal(env.requests.length,0);
  }
});


// AC: @create-managed-prompt-enhancer ac-per-run-seed
test('identical scene and face briefs with identical generated prompts queue without consuming binding evidence',async()=>{
  const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
  env.editPrompt('The same synthetic subject brief.');env.editFace('The same synthetic subject brief.');
  const before=env.current();
  env.jobChange=job=>({...job,result:{main_prompt:'The same generated subject description.',face_prompt:'The same generated subject description.'}});
  for (let run=0;run<2;run++) await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.queued.length,2);
  for (const [index,args] of env.queued.entries()) {
    const compiled=args[1];
    assert.equal(compiled.output['text:scene'].inputs.Text,'The same generated subject description.');
    assert.equal(compiled.output['text:face'].inputs.Text,'The same generated subject description.');
    assert.equal(compiled.output.sample.inputs.seed,env.requests[index].seed);
    assert.equal(compiled.workflow.extra.dmc_prompt_enhancement.main_prompt,'The same generated subject description.');
    assert.equal(compiled.workflow.extra.dmc_prompt_enhancement.face_prompt,'The same generated subject description.');
    assert.equal(nodeAt(compiled.workflow,1126).widgets_values[0],'The same synthetic subject brief.');
    assert.equal(nodeAt(compiled.workflow,1178).widgets_values[0],'The same synthetic subject brief.');
    assert.equal(env.requests[index].image_brief,'The same synthetic subject brief.');
    assert.equal(env.requests[index].face_brief,'The same synthetic subject brief.');
  }
  assert.deepEqual(env.current(),before);
  assert.deepEqual(JSON.parse(env.store.get(draftKey('synthetic-device-a'))),before);
});


// Execute the relevant serialized graph edges with pinned DonutNodes text/
// conditioning semantics (2806ebe12caada993e6305e3f441ce61ab22a11c
// donut_prompt.py); synthetic CLIP records the actual encoded text.
function executeFaceText(workflow, wholeImage = false) {
  const controls=workflow.nodes.flatMap(node=>(node.properties?.donut_app_controls?.groups||[]).flatMap(group=>group.controls||[]));
  const path=controls.find(control=>control.title==='Face detail').path;
  const rootNode=nodeAt(workflow,path[0]);
  const body=workflow.definitions.subgraphs.find(graph=>graph.id===rootNode.type);
  const detail=body.nodes.find(node=>node.id===path[1]);
  function output(graph,parent,id,slot) {
    if(id===-10) {
      const parentGraph=[workflow,...workflow.definitions.subgraphs].find(candidate=>candidate.nodes.includes(parent));
      const edge=parentGraph.links.find(edge=>Array.isArray(edge)?edge[3]===parent.id && edge[4]===slot:edge.target_id===parent.id && edge.target_slot===slot);
      if(!edge) return ''; // Unconnected optional text ports are empty.
      return output(parentGraph,parentGraph===workflow?null:workflow.nodes.find(node=>node.type===parentGraph.id),Array.isArray(edge)?edge[1]:edge.origin_id,Array.isArray(edge)?edge[2]:edge.origin_slot);
    }
    const node=graph.nodes.find(node=>node.id===id);assert.ok(node,'Connected source node must exist');
    const nested=workflow.definitions.subgraphs.find(child=>child.id===node.type);
    if(nested) {
      const edge=nested.links.find(edge=>edge.id===nested.outputs[slot].linkIds[0]);
      return output(nested,node,edge.origin_id,edge.origin_slot);
    }
    const input=name=>{
      const index=node.inputs.findIndex(port=>port.name===name), port=node.inputs[index];
      const edge=graph.links.find(edge=>edge.id===port?.link);
      return edge?output(graph,parent,edge.origin_id,edge.origin_slot):'';
    };
    if(node.type==='DF_Text_Box') return node.widgets_values[0];
    if(node.type==='DonutText') return input('prefix')+input('text')+input('suffix');
    if(node.type==='DonutPromptConditioning') {
      let face=input('face'),scene=input('scene');
      const names=Object.keys(node.widgets_values_named||{}),rows=JSON.parse(node.widgets_values[names.indexOf('prompt_sets_json')]||'[]');
      const index=Number(node.widgets_values[names.indexOf('prompt_set_index')]||1),position=rows.length?Math.max(0,index-1)%(rows.length+1):0;
      if(position){face=rows[position-1].face??face;scene=rows[position-1].scene??scene;}
      const full=face+(node.widgets_values[1]||'')+scene;
      return [full,face,'',{encoded:full},{encoded:face}][slot];
    }
    if(node.type==='DonutKrea2FusionControl') return input('conditioning_in_'+slot);
    throw new Error('Unexpected text/conditioning source '+node.type);
  }
  const value=name=>{
    const port=detail.inputs.find(input=>input.name===name),edge=body.links.find(edge=>edge.id===port.link);
    return output(body,rootNode,edge.origin_id,edge.origin_slot);
  };
  if(wholeImage) {
    const branch=body.links.find(edge=>edge.id===detail.inputs.find(port=>port.name==='edit_prompt').link).origin_id;
    const main=body.nodes.find(node=>node.type==='DonutPromptConditioning' && node.id!==branch);
    return output(body,rootNode,main.id,0);
  }
  return {edit:value('edit_prompt'),positive:value('positive').encoded};
}
function brokenFaceRecipe() {
  const workflow=copy(preset),body=workflow.definitions.subgraphs.find(graph=>graph.nodes.some(node=>node.id===1180));
  const conditioning=nodeAt(workflow,1180);
  const input=body.links.find(edge=>edge.target_id===1180 && edge.origin_id===996);
  const face=body.links.find(edge=>edge.target_id===1180 && edge.origin_id===-10 && body.inputs[edge.origin_slot].name==='face_instruction');
  input.target_slot=1;face.target_slot=2;
  conditioning.inputs[1].link=input.id;conditioning.inputs[2].link=face.id;
  for(const [target,oldSlot] of [[984,0],[1118,3]]) {
    const edge=body.links.find(edge=>edge.origin_id===1180 && edge.target_id===target);
    for(const port of conditioning.outputs) port.links=(port.links||[]).filter(id=>id!==edge.id);
    edge.origin_slot=oldSlot;conditioning.outputs[oldSlot].links.push(edge.id);
  }
  return workflow;
}
function copiedFaceRecipe(workflow) {
  const copyRecipe=copy(workflow),remap=id=>id>=0?id+5000:id;
  for(const graph of [copyRecipe,...copyRecipe.definitions.subgraphs]) {
    for(const node of graph.nodes) {
      node.id=remap(node.id);
      for(const group of node.properties?.donut_app_controls?.groups||[])
        for(const control of group.controls||[]) control.path=control.path.map(remap);
    }
    for(const edge of graph.links) {
      if(Array.isArray(edge)){edge[1]=remap(edge[1]);edge[3]=remap(edge[3]);}
      else {edge.origin_id=remap(edge.origin_id);edge.target_id=remap(edge.target_id);}
    }
  }
  const remapLink=id=>id==null?id:id+7000000;
  for(const graph of [copyRecipe,...copyRecipe.definitions.subgraphs]) {
    for(const edge of graph.links) {if(Array.isArray(edge)) edge[0]=remapLink(edge[0]);else edge.id=remapLink(edge.id);}
    for(const node of graph.nodes) {
      for(const port of node.inputs||[]) if('link' in port) port.link=remapLink(port.link);
      for(const port of node.outputs||[]) if(port.links) port.links=port.links.map(remapLink);
    }
    for(const port of [...(graph.inputs||[]),...(graph.outputs||[])]) if(port.linkIds) port.linkIds=port.linkIds.map(remapLink);
  }
  const ids=new Map(copyRecipe.definitions.subgraphs.map(graph=>[graph.id,'synthetic-copy-'+graph.id]));
  for(const graph of [copyRecipe,...copyRecipe.definitions.subgraphs]) {
    if(ids.has(graph.id)) graph.id=ids.get(graph.id);
    for(const node of graph.nodes) if(ids.has(node.type)) node.type=ids.get(node.type);
  }
  return copyRecipe;
}
function syntheticFaceBriefs(workflow,face) {
  nodeAt(workflow,1126).widgets_values[0]='WHOLE SCENE: a boat on a lake';
  nodeAt(workflow,1125).widgets_values[0]='WHOLE STYLE: watercolor and golden light';
  if(nodeAt(workflow,1178)) nodeAt(workflow,1178).widgets_values[0]=face;
  nodeAt(workflow,984).mode=0;
  return workflow;
}

// AC: @donut-create-plugin ac-face-prompt-routing
test('both bundled presets execute only their separate face text through edit and positive conditioning',()=>{
  for(const original of [preset,JSON.parse(fs.readFileSync(path.join(root,'assets/base-workflow.json'),'utf8'))]) {
    for(const face of ['FACE ONLY: freckles and a calm expression','']) {
      const workflow=syntheticFaceBriefs(copy(original),face);
      assert.deepEqual(executeFaceText(workflow),{edit:face,positive:face});
      assert.equal(executeFaceText(workflow,true),'WHOLE STYLE: watercolor and golden lightWHOLE SCENE: a boat on a lake');
    }
  }
});

// AC: @donut-create-plugin ac-face-prompt-routing
// AC: @donut-create-plugin ac-workflow-state
test('recognized saved broken face branches repair atomically, preserve settings, and are idempotent including copied IDs',async()=>{
  for(const copied of [false,true]) {
    for(const face of ['FACE ONLY: blue eyes','']) {
      let workflow=syntheticFaceBriefs(brokenFaceRecipe(),face);
      nodeAt(workflow,1180).widgets_values[2]=123456;nodeAt(workflow,1180).widgets_values[3]=false;
      if(copied) workflow=copiedFaceRecipe(workflow);
      const before=copy(workflow),env=fixture({initial:workflow,livePrompts:true});await env.boot();
      assert.deepEqual(executeFaceText(env.current()),{edit:face,positive:face});
      for(const id of [1126,1125,1178,1137,1180,984,1014]) assert.deepEqual(nodeAt(env.current(),id+(copied?5000:0)).widgets_values,nodeAt(before,id+(copied?5000:0)).widgets_values);
      assert.equal(env.queued.length,0);
      const reopened=fixture({initial:env.current(),livePrompts:true});await reopened.boot();
      assert.deepEqual(reopened.current(),env.current());assert.equal(reopened.loaded.length,0);
    }
  }
});

// AC: @donut-create-plugin ac-face-prompt-routing
test('legacy missing branches upgrade to executable face-only routes without falling back to whole scene/style',async()=>{
  const env=fixture({initial:legacyFaceRecipe(),livePrompts:true});await env.boot();
  assert.deepEqual(executeFaceText(env.current()),{edit:'',positive:''});
  env.editFace('FACE ONLY: green eyes');
  assert.deepEqual(executeFaceText(env.current()),{edit:'FACE ONLY: green eyes',positive:'FACE ONLY: green eyes'});
});

// AC: @donut-create-plugin ac-face-prompt-routing
test('custom face text or conditioning wiring is preserved when a branch resembles the broken preset',async()=>{
  for(const target of ['edit_prompt','conditioning_in_2']) {
    const workflow=syntheticFaceBriefs(brokenFaceRecipe(),'FACE ONLY: preserve custom wiring');
    const body=workflow.definitions.subgraphs.find(graph=>graph.nodes.some(node=>node.id===1180));
    const edge=body.links.find(edge=>edge.origin_id===1180 && nodeAt(workflow,edge.target_id).inputs[edge.target_slot].name===target);
    nodeAt(workflow,1180).outputs[edge.origin_slot].links=nodeAt(workflow,1180).outputs[edge.origin_slot].links.filter(id=>id!==edge.id);
    edge.origin_id=996;nodeAt(workflow,996).outputs[edge.origin_slot].links.push(edge.id);
    const before=copy(workflow),env=fixture({initial:workflow,livePrompts:true});await env.boot();
    assert.deepEqual(env.current(),before);assert.equal(env.loaded.length,0);assert.equal(env.queued.length,0);
  }
});


// AC: @donut-create-plugin ac-face-prompt-routing
test('new legacy face branches do not inherit selected whole-image prompt sets',async()=>{
  const workflow=legacyFaceRecipe(),main=nodeAt(workflow,996),before=copy(main.widgets_values);
  const names=Object.keys(main.widgets_values_named),sets=JSON.stringify([{face:'WHOLE STYLE SET',scene:'WHOLE SCENE SET'}]);
  main.widgets_values[names.indexOf('prompt_sets_json')]=sets;main.widgets_values[names.indexOf('prompt_set_index')]=2;
  main.widgets_values_named.prompt_sets_json=sets;main.widgets_values_named.prompt_set_index=2;
  const authored=copy(main.widgets_values),env=fixture({initial:workflow,livePrompts:true});await env.boot();
  assert.deepEqual(executeFaceText(env.current()),{edit:'',positive:''});
  env.editFace('FACE ONLY: smile');assert.deepEqual(executeFaceText(env.current()),{edit:'FACE ONLY: smile',positive:'FACE ONLY: smile'});
  assert.deepEqual(nodeAt(env.current(),996).widgets_values,authored);
  assert.notDeepEqual(authored,before);
  assert.equal(executeFaceText(env.current(),true),'WHOLE STYLE SETWHOLE SCENE SET');
});

// AC: @donut-create-plugin ac-workflow-state
test('failed recognized face-route repair restores the complete broken saved recipe without generating',async()=>{
  const workflow=syntheticFaceBriefs(brokenFaceRecipe(),'FACE ONLY: preserve draft'),before=copy(workflow);
  const env=fixture({initial:workflow,livePrompts:true,failFaceUpgrade:true});await env.boot();
  assert.deepEqual(env.current(),before);assert.equal(env.queued.length,0);assert.match(env.notice(),/face prompt could not be added/);
});


const twoFacePlan=[
  {character:'Left traveler',position:'left',face_prompt:'Blue eyes, freckles, relaxed smile.'},
  {character:'Right traveler',position:'right',face_prompt:'Brown eyes, clean jawline, serious expression.'},
];
async function multiFaceFixture() {
  const env=await enhancerFixture();await env.action('enhancer-config',{enabled:true,model:'qwen'});
  const compile=env.app.graphToPrompt;
  env.app.graphToPrompt=async ()=>{
    const value=await compile();
    value.output['1138:1178']=copy(value.output['text:face']);
    value.output['face:expanded']={class_type:'DonutText',inputs:{text:['1138:1178',0]}};
    value.output['face:condition']={class_type:'DonutPromptConditioning',inputs:{face:['face:expanded',0],scene:['text:scene',0]}};
    value.output['face:fusion']={class_type:'DonutKrea2FusionControl',inputs:{conditioning_in_2:['face:condition',4]}};
    value.output['1014:984']={class_type:'DonutFaceDetailer',inputs:{edit_prompt:['face:condition',1],positive:['face:fusion',2],edit_mode:false,max_faces:5,seed:value.output.sample.inputs.seed}};
    return value;
  };
  const fetch=env.api.fetchApi;
  env.api.fetchApi=async (route,options)=>{
    const response=await fetch(route,options);
    if(route==='/object_info') {
      const value=await response.json();
      if(!env.missingMultiNode)value.DMCFaceDetailer={input:{required:{face_prompts_json:['STRING',{}]},optional:{require_single_face:['BOOLEAN',{default:false}]}}};
      return {...response,json:async()=>value};
    }
    return response;
  };
  env.jobChange=job=>({...job,result:{main_prompt:job.result.main_prompt,face_prompt:'',face_prompts:copy(twoFacePlan)}});
  return env;
}

// AC: @create-multiple-face-prompts ac-character-prompts
// AC: @create-multiple-face-prompts ac-single-face-compatible
test('multi-face enhancement submits separate ordered records only in the compiled detailer and run metadata',async()=>{
  const env=await multiFaceFixture();env.editFace('Left traveler: blue eyes. Right traveler: brown eyes.');
  const before=env.current();
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  const compiled=env.queued[0][1],detail=compiled.output['1014:984'];
  assert.equal(detail.class_type,'DMCFaceDetailer');
  assert.equal(detail.inputs.require_single_face,undefined);
  assert.deepEqual(JSON.parse(detail.inputs.face_prompts_json),twoFacePlan);
  assert.deepEqual(detail.inputs.positive,['face:fusion',2]);
  assert.equal(detail.inputs.seed,env.requests[0].seed);
  assert.deepEqual(compiled.workflow.extra.dmc_prompt_enhancement.face_prompts,twoFacePlan);
  assert.equal(compiled.output['text:face'].inputs.Text,'');
  assert.deepEqual(env.current(),before);
  assert.equal(nodeAt(compiled.workflow,984).type,'DonutFaceDetailer');
  assert.deepEqual(copy((await env.action('snapshot')).snapshot.enhancer.lastRun.face_prompts),twoFacePlan);
  await env.app.loadGraphData(compiled.workflow);
  assert.deepEqual(copy((await env.action('snapshot')).snapshot.enhancer.lastRun.face_prompts),twoFacePlan);
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.requests[1].face_brief,env.requests[0].face_brief);
  assert.deepEqual(JSON.parse(env.queued[1][1].output['1014:984'].inputs.face_prompts_json),twoFacePlan);
});

// AC: @create-multiple-face-prompts ac-single-face-compatible
test('legacy two-key results preserve the detailer and new single-face results require exactly one detected face',async()=>{
  for(const result of [{main_prompt:'One scene',face_prompt:'One face'}, {main_prompt:'One scene',face_prompt:'One face',face_prompts:[]}]) {
    const env=await multiFaceFixture();env.jobChange=job=>({...job,result});env.missingMultiNode=!Object.hasOwn(result,'face_prompts');
    await env.api.queuePrompt(0,await env.app.graphToPrompt());
    const strictSingle=Object.hasOwn(result,'face_prompts'),detail=env.queued[0][1].output['1014:984'];
    assert.equal(detail.class_type,strictSingle?'DMCFaceDetailer':'DonutFaceDetailer');
    assert.equal(detail.inputs.face_prompts_json,strictSingle?'[]':undefined);
    assert.equal(detail.inputs.require_single_face,strictSingle?true:undefined);
    assert.equal(env.queued[0][1].output['text:face'].inputs.Text,'One face');
  }
});

// AC: @create-multiple-face-prompts ac-character-prompts
// AC: @create-multiple-face-prompts ac-mismatch
test('malformed and spatially ambiguous character records cannot submit a shared-face fallback',async()=>{
  const invalid=[
    [twoFacePlan[0]], [...twoFacePlan].reverse(),
    [{...twoFacePlan[0],position:'left'},{...twoFacePlan[1],position:'left'}],
    [{...twoFacePlan[0],position:'gazing left'},twoFacePlan[1]],
    [{...twoFacePlan[0],position:'1st'},twoFacePlan[1]],
    [{...twoFacePlan[0],character:''},twoFacePlan[1]],
    [twoFacePlan[0],{...twoFacePlan[1],character:twoFacePlan[0].character}],
    [twoFacePlan[0],{...twoFacePlan[1],face_prompt:''}],
    [twoFacePlan[0],{...twoFacePlan[1],extra:true}],
    [twoFacePlan[0],{...twoFacePlan[1],character:'x'.repeat(201)}],
    [twoFacePlan[0],{...twoFacePlan[1],face_prompt:'x'.repeat(8001)}],
    Array.from({length:17},(_,index)=>({...twoFacePlan[0],character:'Character '+index,position:index+'th'})),
  ];
  for(const records of invalid) {
    const env=await multiFaceFixture(),before=env.current();env.jobChange=job=>({...job,result:{...job.result,face_prompt:'',face_prompts:records}});
    await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()));assert.equal(env.queued.length,0);assert.deepEqual(env.current(),before);
  }
});

// AC: @create-multiple-face-prompts ac-mismatch
test('missing per-face node capability and custom shared-face branches fail before image submission',async()=>{
  for(const mode of ['missing','wrong-text','wrong-positive','multiple-detailers','no-detailer','edit','wrong-detail-instance','wrong-face-instance','wildcard','hook','max-faces','no-schema']) {
    const env=await multiFaceFixture();
    if(mode==='missing')env.missingMultiNode=true;
    if(mode==='no-schema'){const fetch=env.api.fetchApi;env.api.fetchApi=async (route,options)=>{const response=await fetch(route,options);if(route!=='/object_info')return response;const value=await response.json();value.DMCFaceDetailer={};return {...response,json:async()=>value};};}
    const compile=env.app.graphToPrompt;
    env.app.graphToPrompt=async ()=>{
      const value=await compile(),detail=value.output['1014:984'];
      if(mode==='wrong-text')detail.inputs.edit_prompt=['condition',0];
      if(mode==='wrong-positive')detail.inputs.positive=['condition',3];
      if(mode==='multiple-detailers')value.output.extra=copy(detail);
      if(mode==='no-detailer')delete value.output['1014:984'];
      if(mode==='edit')detail.inputs.edit_mode=true;
      if(mode==='wrong-detail-instance'){value.output['9999:984']=detail;delete value.output['1014:984'];}
      if(mode==='wrong-face-instance'){value.output['9999:1178']=value.output['1138:1178'];value.output['face:expanded'].inputs.text=['9999:1178',0];}
      if(mode==='wildcard')detail.inputs.wildcard='A shared face wildcard';
      if(mode==='hook')detail.inputs.detailer_hook=['custom-hook',0];
      if(mode==='max-faces')detail.inputs.max_faces=1;
      return value;
    };
    await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/face|DMCFaceDetailer|conditioning|edit/i);
    assert.equal(env.queued.length,0,mode);
  }
});

// AC: @create-multiple-face-prompts ac-single-face-compatible
// AC: @create-multiple-face-prompts ac-mismatch
test('multi-face records support ordered ordinal positions and never erase ambiguous supplied face instructions',async()=>{
  const env=await multiFaceFixture();
  env.jobChange=job=>({...job,result:{main_prompt:'Scene',face_prompt:'',face_prompts:twoFacePlan.map((entry,index)=>({...entry,position:index?'2nd':'1st'}))}});
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.deepEqual(JSON.parse(env.queued[0][1].output['1014:984'].inputs.face_prompts_json).map(entry=>entry.position),['1st','2nd']);
  env.editFace('Left traveler has blue eyes; right traveler has brown eyes.');
  env.jobChange=job=>({...job,result:{main_prompt:'Scene',face_prompt:'',face_prompts:[]}});
  await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/face|ambiguous/i);
  assert.equal(env.queued.length,1);
});

// AC: @create-multiple-face-prompts ac-mismatch
test('cancel and stale edits while per-face capabilities load cannot queue the multi-face run',async()=>{
  for(const mode of ['cancel','stale']) {
    const env=await multiFaceFixture(),fetch=env.api.fetchApi;let checks=0,release;
    env.api.fetchApi=async (route,options)=>{
      if(route==='/object_info' && ++checks===2)await new Promise(resolve=>{release=resolve;});
      return fetch(route,options);
    };
    const pending=env.api.queuePrompt(0,await env.app.graphToPrompt()),rejected=assert.rejects(pending);
    await env.settle();
    if(mode==='cancel')await env.action('cancel-generation');else env.editPrompt('New synthetic scene');
    release();await rejected;assert.equal(env.queued.length,0);
  }
});


// AC: @create-multiple-face-prompts ac-single-face-compatible
test('disabled known face detail keeps ordered metadata without requiring the multi-face runtime, and re-enabling routes each face',async()=>{
  const env=await multiFaceFixture(),compile=env.app.graphToPrompt;
  let enabled=false;env.editFaceDetail(enabled);env.missingMultiNode=true;
  env.app.graphToPrompt=async ()=>{const value=await compile();if(!enabled)delete value.output['1014:984'];return value;};
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.queued.length,1);assert.equal(env.queued[0][1].output['1014:984'],undefined);
  assert.deepEqual(env.queued[0][1].workflow.extra.dmc_prompt_enhancement.face_prompts,twoFacePlan);
  assert.deepEqual(copy((await env.action('snapshot')).snapshot.enhancer.lastRun.face_prompts),twoFacePlan);
  enabled=true;env.editFaceDetail(enabled);env.missingMultiNode=false;
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.queued[1][1].output['1014:984'].class_type,'DMCFaceDetailer');
  assert.deepEqual(JSON.parse(env.queued[1][1].output['1014:984'].inputs.face_prompts_json),twoFacePlan);
  assert.equal(env.requests[1].image_brief,env.requests[0].image_brief);
});


// AC: @create-multiple-face-prompts ac-mismatch
test('an authored edit after per-face capability verification still cancels submission',async()=>{
  const env=await multiFaceFixture(),fetch=env.api.fetchApi;
  env.api.fetchApi=async (route,options)=>{
    const response=await fetch(route,options);if(route!=='/object_info')return response;
    const value=await response.json(),required={};
    Object.defineProperty(required,'face_prompts_json',{get(){queueMicrotask(()=>env.editPrompt('Latest authored scene'));return ['STRING',{}];}});
    value.DMCFaceDetailer={input:{required}};return {...response,json:async()=>value};
  };
  await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/workflow changed/i);
  assert.equal(env.queued.length,0);assert.equal(nodeAt(env.current(),1126).widgets_values[0],'Latest authored scene');
});


// AC: @create-multiple-face-prompts ac-mismatch
// AC: @create-multiple-face-prompts ac-single-face-compatible
test('strict single-face execution requires the runtime Boolean guard capability and preserves reference/edit options',async()=>{
  for(const mode of ['supported','missing-guard','wrong-type']) {
    const env=await multiFaceFixture(),fetch=env.api.fetchApi,compile=env.app.graphToPrompt;
    env.jobChange=job=>({...job,result:{main_prompt:'One character',face_prompt:'Blue eyes and freckles',face_prompts:[]}});
    env.app.graphToPrompt=async ()=>{const value=await compile();Object.assign(value.output['1014:984'].inputs,{
      edit_mode:true,face_reference:['reference',0],face_reference_b:['reference-b',0],detailer_hook:['hook',0],wildcard:'A single face wildcard'});return value;};
    env.api.fetchApi=async (route,options)=>{
      const response=await fetch(route,options);if(route!=='/object_info')return response;
      const value=await response.json();
      if(mode==='missing-guard')delete value.DMCFaceDetailer.input.optional.require_single_face;
      if(mode==='wrong-type')value.DMCFaceDetailer.input.optional.require_single_face=['STRING',{}];
      return {...response,json:async()=>value};
    };
    if(mode==='supported') {
      const before=env.current(),compiled=await env.app.graphToPrompt();await env.api.queuePrompt(0,compiled);
      const detail=env.queued[0][1].output['1014:984'];assert.equal(detail.class_type,'DMCFaceDetailer');
      assert.deepEqual(detail.inputs,{...compiled.output['1014:984'].inputs,face_prompts_json:'[]',require_single_face:true});
      assert.deepEqual(env.current(),before);
    } else {await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/single.face|guard|repair/i);assert.equal(env.queued.length,0);}
  }
});

// AC: @create-multiple-face-prompts ac-single-face-compatible
test('disabled known face detail retains new single-face metadata without requiring a runtime guard',async()=>{
  const env=await multiFaceFixture(),compile=env.app.graphToPrompt;env.editFaceDetail(false);env.missingMultiNode=true;
  env.jobChange=job=>({...job,result:{main_prompt:'One character',face_prompt:'Blue eyes and freckles',face_prompts:[]}});
  env.app.graphToPrompt=async ()=>{const value=await compile();delete value.output['1014:984'];return value;};
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  assert.equal(env.queued.length,1);assert.equal(env.queued[0][1].output['1014:984'],undefined);
  assert.equal(env.queued[0][1].workflow.extra.dmc_prompt_enhancement.face_prompt,'Blue eyes and freckles');
});

const familyFacePlan=[
  {character:'Upper left',position:{left_of:['Upper middle'],above:['Lower left']},face_prompt:'Freckles and blue eyes.'},
  {character:'Upper middle',position:{left_of:['Upper right'],above:['Lower middle']},face_prompt:'Brown eyes and a smile.'},
  {character:'Upper right',position:{left_of:[],above:['Lower right']},face_prompt:'Green eyes and dimples.'},
  {character:'Lower left',position:{left_of:['Lower middle'],above:[]},face_prompt:'Round cheeks and a calm expression.'},
  {character:'Lower middle',position:{left_of:['Lower right'],above:[]},face_prompt:'Hazel eyes and a serious expression.'},
  {character:'Lower right',position:{left_of:[],above:[]},face_prompt:'Grey eyes and an open smile.'},
];
const raggedFacePlan=[
  {character:'Top',position:{left_of:[],above:['Bottom left','Bottom right']},face_prompt:'Blue eyes.'},
  {character:'Bottom right',position:{left_of:[],above:[]},face_prompt:'Green eyes.'},
  {character:'Bottom left',position:{left_of:['Bottom right'],above:[]},face_prompt:'Brown eyes.'},
];

// AC: @create-multiple-face-prompts ac-character-prompts
test('relative face plans retain two-axis and ragged layouts in compiled routing and metadata regardless of array order',async()=>{
  for(const plan of [copy(familyFacePlan).reverse(),raggedFacePlan,[{character:'Top',position:{left_of:[],above:['Bottom']},face_prompt:'Blue eyes.'},{character:'Bottom',position:{left_of:[],above:[]},face_prompt:'Brown eyes.'}]]) {
    const env=await multiFaceFixture(),before=env.current(),compile=env.app.graphToPrompt;
    env.app.graphToPrompt=async()=>{const compiled=await compile();compiled.output['1014:984'].inputs.max_faces=16;return compiled;};
    env.jobChange=job=>({...job,result:{main_prompt:'Placed family scene',face_prompt:'',face_prompts:copy(plan)}});
    const originalSeed=(await env.app.graphToPrompt()).output.sample.inputs.seed;
    await env.api.queuePrompt(0,await env.app.graphToPrompt());
    const submitted=env.queued[0][1];
    assert.deepEqual(JSON.parse(submitted.output['1014:984'].inputs.face_prompts_json),plan);
    assert.equal(submitted.output['1014:984'].class_type,'DMCFaceDetailer');
    assert.equal(submitted.output['1014:984'].inputs.seed,originalSeed);
    assert.equal(env.requests[0].seed,originalSeed);
    assert.deepEqual(submitted.workflow.extra.dmc_prompt_enhancement.face_prompts,plan);
    assert.deepEqual(env.current(),before);
    await env.app.loadGraphData(submitted.workflow);
    assert.deepEqual(copy((await env.action('snapshot')).snapshot.enhancer.lastRun.face_prompts),plan);
    await env.api.queuePrompt(0,await env.app.graphToPrompt());
    assert.equal(env.requests[1].image_brief,env.requests[0].image_brief);
    assert.deepEqual(JSON.parse(env.queued[1][1].output['1014:984'].inputs.face_prompts_json),plan);
  }
});

// AC: @create-multiple-face-prompts ac-mismatch
test('invalid relative face schemas and axis cycles fail before compiled face submission',async()=>{
  const cases=[
    position=>{position.left_of=['Upper left'];},
    position=>{position.left_of=['Unknown character'];},
    position=>{position.left_of=['upper middle'];},
    position=>{position.left_of=['Upper middle','Upper middle'];},
    position=>{position.left_of=Array(16).fill('Upper middle');},
    position=>{position.left_of='Upper middle';},
    position=>{position.above=null;},
    position=>{position.above=[1];},
    position=>{delete position.above;},
    position=>{position.row=1;},
  ].map(change=>{const plan=copy(familyFacePlan);change(plan[0].position);return plan;});
  const horizontal=copy(familyFacePlan);horizontal[2].position.left_of=['Upper left'];cases.push(horizontal);
  const vertical=copy(familyFacePlan);vertical[3].position.above=['Upper left'];cases.push(vertical);
  const mixed=copy(familyFacePlan);mixed[0].position='left';cases.push(mixed);
  const duplicate=copy(familyFacePlan);duplicate[1].character='UPPER LEFT';cases.push(duplicate);
  for(const plan of cases) {
    const env=await multiFaceFixture(),before=env.current();
    env.jobChange=job=>({...job,result:{main_prompt:'Family scene',face_prompt:'',face_prompts:plan}});
    await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/incomplete|ambiguous/i);
    assert.equal(env.queued.length,0);assert.deepEqual(env.current(),before);
  }
});

// AC: @create-multiple-face-prompts ac-mismatch
test('tampered cyclic run metadata is not exposed or reused as an executable face plan',async()=>{
  const env=await multiFaceFixture(),compile=env.app.graphToPrompt;
  env.app.graphToPrompt=async()=>{const compiled=await compile();compiled.output['1014:984'].inputs.max_faces=16;return compiled;};
  env.jobChange=job=>({...job,result:{main_prompt:'Family scene',face_prompt:'',face_prompts:copy(familyFacePlan)}});
  await env.api.queuePrompt(0,await env.app.graphToPrompt());
  const saved=copy(env.queued[0][1].workflow),bad=saved.extra.dmc_prompt_enhancement.face_prompts;
  bad[2].position.left_of=['Upper left'];
  await env.app.loadGraphData(saved);
  assert.equal((await env.action('snapshot')).snapshot.enhancer.lastRun,undefined);
  const before=env.current();
  env.jobChange=job=>({...job,result:{main_prompt:'Tampered family scene',face_prompt:'',face_prompts:bad}});
  await assert.rejects(env.api.queuePrompt(0,await env.app.graphToPrompt()),/incomplete|ambiguous/i);
  assert.equal(env.queued.length,1);assert.deepEqual(env.current(),before);
});
