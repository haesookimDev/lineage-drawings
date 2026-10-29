export type Port={point:number[];side:string;score:number;status:string};
type Junction={point:number[];type:string;score:number;connected:boolean};
type Path={from_id:string;to_id:string;points:number[][];from_bbox?:number[];to_bbox?:number[]};
export type Recognition={recognition_algorithm?:string;recognition_is_advisory?:boolean;port_count?:number;paths?:Path[];junctions?:Junction[];lines?:{points:number[][];style:string;score:number}[]};
type Shape={id:string;bbox:number[];status:string;properties:Record<string,unknown>};
export function pointsString(points:number[][]){return points.map(p=>p.map(x=>x*1000).join(',')).join(' ')}
const sameBox=(a:number[]|undefined,b:number[])=>!!a&&a.every((v,i)=>Math.abs(v-b[i])<1e-7);
export function observedPath(recognition:Recognition|undefined,from:Shape,to:Shape){
 const p=recognition?.paths?.find(p=>(p.from_id===from.id&&p.to_id===to.id)||(p.from_id===to.id&&p.to_id===from.id));
 if(!p)return null;
 const forward=p.from_id===from.id;
 if(!sameBox(forward?p.from_bbox:p.to_bbox,from.bbox)||!sameBox(forward?p.to_bbox:p.from_bbox,to.bbox))return null;
 return forward?p.points:[...p.points].reverse();
}
const junctionNames:Record<string,string>={tee:'T 접속 후보',dot:'점 표시 접속 후보',plain_cross:'비접속 교차 후보',inferred_junction:'점 표시 없는 추정 접속 · 확인 필요'};
export function TopologyOverlay({recognition,elements}:{recognition:Recognition|undefined;elements:Shape[]}){
 return <g className="topology-overlay" aria-label="선·포트·접속점 인식 후보">
  {recognition?.lines?.map((l,i)=><polyline key={'l'+i} points={pointsString(l.points)} className={'recognized-line '+l.style}><title>{l.style==='dashed'?'점선':'실선'} 후보 · 검토 필요</title></polyline>)}
  {elements.filter(e=>e.status!=='rejected'&&sameBox(e.properties.topology_bbox as number[]|undefined,e.bbox)).flatMap(e=>(e.properties.ports as Port[]|undefined||[]).map((p,i)=><circle key={e.id+i} cx={p.point[0]*1000} cy={p.point[1]*1000} r={2.5} className="recognized-port"><title>포트 후보 · {p.side} · 점수 {p.score.toFixed(2)} · 검토 필요</title></circle>))}
  {recognition?.junctions?.map((j,i)=><rect key={'j'+i} x={j.point[0]*1000-2.5} y={j.point[1]*1000-2.5} width={5} height={5} className={'recognized-junction '+j.type}><title>{junctionNames[j.type]||j.type} · 점수 {j.score.toFixed(2)}</title></rect>)}
 </g>
}
export function PortsInfo({element}:{element:Shape}){
 const ports=element.properties.ports as Port[]|undefined;
 if(!ports)return null;
 const stale=!sameBox(element.properties.topology_bbox as number[]|undefined,element.bbox);
 return <details className="ports-info"><summary>포트 인식 후보 {ports.length}개</summary><p>추출 당시의 관찰 근거입니다. 요소 승인은 포트·접속점의 승인을 뜻하지 않습니다.</p>{stale&&<p>요소 영역이 변경되었습니다. 기존 포트와 경로 표시는 숨겼습니다.</p>}<ul>{ports.map((p,i)=><li key={i}>{p.side} · ({p.point.map(v=>v.toFixed(3)).join(', ')}) · 점수 {p.score.toFixed(2)}</li>)}</ul><small>점수는 보정된 정확도가 아닙니다.</small></details>
}

export function ReviewedTopologyOverlay({review}:{review:any}){return <g className="topology-overlay" aria-label={'저장된 교정 버전 '+review.version}>{review.payload.ports.filter((p:any)=>!p.stale&&p.status!=='rejected').map((p:any)=><circle key={p.id} cx={p.point[0]*1000} cy={p.point[1]*1000} r={3} fill={p.status==='approved'?'#147b59':'#176db1'}><title>교정 포트 · {p.side} · {p.status}</title></circle>)}{review.payload.junctions.filter((p:any)=>p.status!=='rejected').map((p:any)=><rect key={p.id} x={p.point[0]*1000-3} y={p.point[1]*1000-3} width={6} height={6} fill={p.kind==='plain_cross'?'#8846b8':'#c18525'}><title>교정 접점 · {p.kind} · {p.status}</title></rect>)}</g>}
