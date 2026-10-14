"use client";
import {useEffect,useState} from "react";
import HeartScene from "./HeartScene";
import DoctorGuide from "./DoctorGuide";
type Vessel="LAD"|"LCX"|"RCA";
type Prediction={probabilities:Record<string,number>;cad_status:string;imputed_from_reference:string[];explanation:{feature:string;value:string|number;contribution:number}[];reference_probability:number;explanation_method:string};
type Schema={features:Record<string,string|number>;choices:Record<string,(string|number)[]>};
const base=[["Age","Age","years"],["BP","Systolic blood pressure","mmHg"],["LDL","LDL cholesterol","mg/dL"],["EF-TTE","Ejection fraction","%"],["BMI","Body mass index","kg/m²"],["Sex","Sex",""]];
const vesselNames={LAD:"Left anterior descending",LCX:"Left circumflex",RCA:"Right coronary artery"};
const api=process.env.NEXT_PUBLIC_API_BASE_URL||"http://localhost:8000";
export default function Home(){
 const [active,setActive]=useState<Vessel>("LAD"),[patient,setPatient]=useState<Record<string,string>>({}),[schema,setSchema]=useState<Schema|null>(null),[prediction,setPrediction]=useState<Prediction|null>(null),[busy,setBusy]=useState(false),[error,setError]=useState("");
 useEffect(()=>{fetch(api+"/schema").then(r=>r.json()).then(setSchema).catch(()=>{})},[]);
 function update(key:string,value:string){setPatient(p=>({...p,[key]:value}));setPrediction(null)}
 function field(key:string,label=key,unit=""){
  const choices=schema?.choices[key]||(key==="Sex"?["Male","Fmale"]:null);
  return <label key={key}>{label}<span>{unit||"Clinical measurement"}</span>{choices?<select value={patient[key]||""} onChange={e=>update(key,e.target.value)}><option value="">Not provided</option>{choices.map(v=><option key={v} value={v}>{v==="Fmale"?"Female":String(v)}</option>)}</select>:<input type="number" step="any" placeholder="Enter value" value={patient[key]||""} onChange={e=>update(key,e.target.value)} />}</label>
 }
 async function predict(e:React.FormEvent){
  e.preventDefault();setError("");setPrediction(null);
  const features:Record<string,string|number>={};
  Object.entries(patient).forEach(([k,v])=>{if(v.trim())features[k]=typeof schema?.features[k]==="string"||k==="Sex"?v:Number(v)});
  if(!Object.keys(features).length){setError("Add at least one measurement to continue.");return}
  setBusy(true);
  try{const r=await fetch(api+"/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({features}),signal:AbortSignal.timeout(30000)});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==="string"?data.detail:"Please check your measurements.");setPrediction(data)}
  catch(e){setError(e instanceof Error&& !e.message.includes("fetch")?e.message:"Vascularis is temporarily unavailable. Please try again.")}finally{setBusy(false)}
 }
 return <main className="appShell">
  <header className="topbar"><a className="brand" href="#top"><span className="brandMark">V</span><span>vascularis<small>CARDIAC INSIGHTS</small></span></a><nav><a href="#overview">Your heart</a><a href="#details">Your details</a><a href="#evidence">About your insights</a></nav><span className="online"><i/> Your heart, in focus</span></header>
  <section className="welcome" id="top"><div><div className="eyebrow">A CLEARER VIEW OF YOUR HEART</div><h1>Know more.<br/><em>See the bigger picture.</em></h1><p>Your measurements. Clearer insights. Explore coronary estimates with Vascularis and a detailed, interactive view of the heart.</p></div><div className="datasetBadge"><span className="badgeIcon">✦</span><div><small>MEET YOUR HEART COMPANION</small><strong>Vascularis AI</strong><span>Understand. Explore. Ask.</span></div></div></section>
  <section className="heroGrid" id="overview">
   <article className="panel anatomyPanel" id="map"><div className="panelHead"><div><div><small>EXPLORE THE ANATOMY</small><h2>Your heart, in detail</h2></div></div><span className="viewTag">INTERACTIVE 3D</span></div>
    <div className="anatomyStage"><HeartScene active={active} onSelect={setActive} probabilities={prediction?.probabilities}/><div className="vesselSelector">{(["LAD","LCX","RCA"] as Vessel[]).map(v=><button key={v} aria-pressed={active===v} onClick={()=>setActive(v)} style={{borderColor:prediction?(prediction.probabilities[v]>=.5?"#ff657e":"#5ce1c2"):undefined}}>{v} {prediction?Math.round(prediction.probabilities[v]*100)+"%":""}</button>)}</div><div className="mapHint">Drag to rotate · scroll or pinch to zoom</div></div>
    <div className="vesselDetail"><small>SELECTED VESSEL</small><h3>{vesselNames[active]}</h3><p>{prediction?<><b>{Math.round(prediction.probabilities[active]*100)}%</b> estimated probability of stenosis in this vessel.</>:"Add your measurements to explore this vessel’s estimate."}</p><span>Locator dots represent vessel-level estimates on generic anatomy; they do not identify a lesion location.</span>{prediction&&<div className="riskLegend"><i/> Below 50% <i/> 50% and above · display threshold</div>}</div>
   </article>
   <aside className="panel insightPanel" id="details"><div className="panelHead"><div><div><small>LET’S START WITH YOU</small><h2>Your clinical details</h2></div></div></div>
    <form className="patientForm" onSubmit={predict}><p className="formIntro">Use measurements from your clinical records. Any fields left blank use reference values, which can substantially affect the estimate.</p><div className="inputGrid">{base.map(([key,label,unit])=>field(key,label,unit))}</div>
     <details className="advancedFields"><summary>Add more clinical details <span>ECG, symptoms & laboratory results</span></summary><div className="inputGrid">{schema?Object.keys(schema.features).filter(k=>!base.some(b=>b[0]===k)).map(k=>field(k)):<p>Additional fields will appear when Vascularis connects.</p>}</div></details>
     <button className="predictButton" disabled={busy}>{busy?"Vascularis is assessing…":"Explore my insights"} <span>→</span></button>{error&&<p className="formError" role="alert">{error}</p>}
    </form>
    {prediction&&<div className="predictionResult" aria-live="polite"><div className="resultHead"><b>Your Vascularis insights</b><span>{prediction.cad_status} · threshold 50%</span></div><div className="probabilityGrid">{Object.entries(prediction.probabilities).map(([v,p])=><div key={v} className={v===active?"probSelected":""}><span>{v}</span><b>{(p*100).toFixed(1)}%</b><i><em style={{width:p*100+"%"}}/></i></div>)}</div><p>{prediction.imputed_from_reference.length} missing fields use reference values. These are classification estimates, not a diagnosis or a calibrated future-event risk.</p><details className="imputedDetails"><summary>Review reference-filled fields</summary><div>{prediction.imputed_from_reference.join(" · ")}</div></details>
     <div className="explanation"><h3>What influenced this estimate?</h3><p>How your entered measurements shifted the CAD estimate from the reference profile.</p>{prediction.explanation.slice(0,8).map(x=><div className="factor" key={x.feature}><div><b>{x.feature}</b><span>{String(x.value)} · {x.contribution>=0?"higher":"lower"} estimate</span></div><div className="factorTrack"><i style={{width:Math.max(2,Math.abs(x.contribution)/Math.max(.01,...prediction.explanation.map(a=>Math.abs(a.contribution)))*100)+"%",background:x.contribution>=0?"#ff8197":"#67d9c0"}}/></div><small>{x.contribution>0?"+":""}{x.contribution.toFixed(3)} log-odds</small></div>)}<details><summary>How this explanation works</summary><p>{prediction.explanation_method} Reference estimate: {(prediction.reference_probability*100).toFixed(1)}%. Missing fields have zero change relative to that reference. This explains CAD; vessel-specific explanations are not yet available.</p></details></div>
    </div>}
   </aside>
  </section>
  <details className="evidenceSection productEvidence" id="evidence"><summary>About your insights <span>How Vascularis is evaluated</span></summary><p>Vascularis uses clinical records to estimate overall coronary artery disease and stenosis in three major vessels. Current serving pipelines were evaluated on a cohort of 303 records with five-fold validation; performance can differ in other populations.</p><div className="scoreCards">{[["CAD","87.13%","0.9303"],["LAD","79.54%","0.8460"],["LCX","67.66%","0.7204"],["RCA","68.98%","0.7430"]].map(([v,acc,auc])=><div className="scoreCard" key={v}><span className="scoreTop"><b>{v}</b><small>Vascularis AI</small></span><strong>{acc}</strong><span className="scoreBottom">Accuracy · AUC {auc}</span></div>)}</div><p>Separate ten-fold experiments reached higher accuracy for some outcomes. Those experimental results do not describe the saved pipelines currently serving this screen.</p><p>Anatomy asset: <a href="https://grab3d.com/models/anatomy/human-heart-cutaway/" target="_blank" rel="noreferrer">Human Heart Cutaway · Grab3D</a>, CC0. The source discloses AI-assisted mesh generation; it is a detailed educational illustration, not a medically validated reconstruction.</p></details>
  <footer className="safetyBar"><span className="safetyIcon">✚</span><div><b>For education and decision support only</b><span>Vascularis does not replace clinical assessment or formal diagnostic imaging.</span></div></footer>
  <DoctorGuide hasResult={!!prediction}/>
 </main>
}
