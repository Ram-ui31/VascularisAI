"use client";
import { useEffect, useRef, useState } from "react";
import type { Mesh, MeshStandardMaterial } from "three";
type Vessel = "LAD" | "LCX" | "RCA";
export default function HeartScene({probabilities,active,onSelect}:{probabilities?:Record<string,number>;active:Vessel;onSelect:(v:Vessel)=>void}) {
  const host=useRef<HTMLDivElement>(null);
  const controls=useRef<{reset:()=>void;zoom:(factor:number)=>void}|null>(null);
  const markers=useRef<Mesh[]>([]);
  const latest=useRef({probabilities,active,onSelect});
  const [status,setStatus]=useState("Loading detailed anatomy…");
  useEffect(()=>{latest.current={probabilities,active,onSelect}},[probabilities,active,onSelect]);
  useEffect(()=>{
    let disposed=false, release:(()=>void)|undefined;
    async function setup(){
      const T=await import("three");
      const {GLTFLoader}=await import("three/addons/loaders/GLTFLoader.js");
      const {OrbitControls}=await import("three/addons/controls/OrbitControls.js");
      if(disposed||!host.current)return;
      const container=host.current;
      const renderer=new T.WebGLRenderer({antialias:true,alpha:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.05;container.appendChild(renderer.domElement);
      const scene=new T.Scene(), camera=new T.PerspectiveCamera(35,1,.1,100);camera.position.set(0,.15,11);
      scene.add(new T.HemisphereLight(0xffffff,0x29314b,2));
      const key=new T.DirectionalLight(0xfff1e6,3);key.position.set(-4,5,6);scene.add(key);
      const fill=new T.DirectionalLight(0xa6bfff,1.3);fill.position.set(5,1,-2);scene.add(fill);
      const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.enablePan=false;orbit.minDistance=5;orbit.maxDistance=18;orbit.target.set(0,0,0);orbit.update();orbit.saveState();
      controls.current={reset:()=>orbit.reset(),zoom:(factor)=>{camera.position.multiplyScalar(factor);camera.position.clampLength(5,18);orbit.update()}};
      const modelRoot=new T.Group();scene.add(modelRoot);
      const ray=new T.Raycaster();let start=[0,0];
      const down=(e:PointerEvent)=>{start=[e.clientX,e.clientY]};
      const pick=(e:PointerEvent)=>{if(Math.hypot(e.clientX-start[0],e.clientY-start[1])>5)return;const r=renderer.domElement.getBoundingClientRect();ray.setFromCamera(new T.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);const hit=ray.intersectObjects(markers.current)[0];if(hit)latest.current.onSelect(hit.object.userData.vessel)};
      renderer.domElement.addEventListener("pointerdown",down);renderer.domElement.addEventListener("pointerup",pick);
      const resize=()=>{renderer.setSize(container.clientWidth,container.clientHeight);camera.aspect=container.clientWidth/container.clientHeight;camera.updateProjectionMatrix()};const observer=new ResizeObserver(resize);observer.observe(container);resize();
      let frame=0;const draw=()=>{if(disposed)return;frame=requestAnimationFrame(draw);orbit.update();markers.current.forEach(m=>{const v=m.userData.vessel as Vessel,p=latest.current.probabilities?.[v];const material=m.material as MeshStandardMaterial;material.color.set(p===undefined?"#b4a9ff":p>=.5?"#ff657e":"#5ce1c2");material.emissive.copy(material.color);material.emissiveIntensity=latest.current.active===v?.7:.22;m.scale.setScalar(latest.current.active===v?1.35:1)});renderer.render(scene,camera)};draw();
      release=()=>{cancelAnimationFrame(frame);observer.disconnect();orbit.dispose();renderer.dispose();renderer.domElement.remove();scene.traverse(o=>{const m=o as Mesh;m.geometry?.dispose();const materials=Array.isArray(m.material)?m.material:[m.material];materials.forEach(mat=>mat?.dispose())});markers.current=[]};
      try{
        const gltf=await new GLTFLoader().loadAsync("/anatomy/heart.glb");
        if(disposed)return;
        const box=new T.Box3().setFromObject(gltf.scene),size=box.getSize(new T.Vector3()),center=box.getCenter(new T.Vector3());
        const scale=5.3/Math.max(size.x,size.y,size.z);gltf.scene.position.copy(center).multiplyScalar(-scale);gltf.scene.scale.setScalar(scale);modelRoot.add(gltf.scene);
        // Vessel-level locator nodes on generic anatomy, not inferred lesions.
        const locations:Record<Vessel,number[]>={LAD:[-.18,-.35,1.15],LCX:[1.12,.38,.55],RCA:[-1.15,.16,.8]};
        markers.current=Object.entries(locations).map(([v,p])=>{const marker=new T.Mesh(new T.SphereGeometry(.1,20,16),new T.MeshStandardMaterial());marker.position.set(p[0],p[1],p[2]);marker.userData.vessel=v;modelRoot.add(marker);return marker});setStatus("");
      }catch{if(!disposed)setStatus("Anatomy could not load. Reload to try again.")}
    }
    setup().catch(()=>setStatus("3D rendering is unavailable in this browser."));return()=>{disposed=true;release?.()};
  },[]);
  return <><div ref={host} className="heartCanvas" aria-label="Detailed anatomical heart: drag to rotate, scroll or pinch to zoom"/>{status&&<div className="meshStatus" role="status">{status}</div>}<div className="meshControls"><button onClick={()=>controls.current?.zoom(.85)} aria-label="Zoom in">+</button><button onClick={()=>controls.current?.zoom(1.15)} aria-label="Zoom out">−</button><button onClick={()=>controls.current?.reset()}>Reset view</button></div></>;
}
