"""Batch seam-aware ERP stylization."""
import argparse, hashlib, json, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence
import torch
from PIL import Image
from telestyleimage_inference import ImageStyleInference
from telestylepanorama_inference import DEFAULT_PROMPT, stylize_panorama
IMAGE_SUFFIXES={".jpg",".jpeg",".png",".webp"}

def discover_images(directory: Path)->list[Path]:
    if not directory.is_dir(): raise FileNotFoundError(f"Image directory does not exist: {directory}")
    images=sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),key=lambda p:p.name.casefold())
    if not images: raise ValueError(f"No supported images found in {directory}.")
    stems={}
    for path in images:
        key=path.stem.casefold()
        if key in stems: raise ValueError(f"Duplicate image stem in {directory}: {stems[key].name}, {path.name}")
        stems[key]=path
    return images

def select_styles(style_paths: Sequence[Path], requested_stems: Sequence[str]|None)->list[Path]:
    if requested_stems is None: return list(style_paths)
    names=[name.casefold() for name in requested_stems]
    if len(names)!=len(set(names)): raise ValueError("--styles cannot contain duplicate style names.")
    available={path.stem.casefold() for path in style_paths}
    missing=sorted(set(names)-available)
    if missing: raise ValueError(f"Unknown style name(s): {', '.join(missing)}")
    return [path for path in style_paths if path.stem.casefold() in set(names)]

def output_path_for(output_dir:Path,style_path:Path,content_path:Path)->Path:
    return output_dir/style_path.stem/f"{content_path.stem}.png"

def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def style_metadata(path:Path)->dict:
    return {"style":str(path),"style_name":path.name,"style_size_bytes":path.stat().st_size,"style_sha256":sha256_file(path)}

def make_manifest_path(output_dir:Path,style_paths:Sequence[Path],run_id:str|None=None)->Path:
    run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label=re.sub(r"[^A-Za-z0-9._-]+","_","-".join(path.stem for path in style_paths)).strip("._-") or "styles"
    return output_dir/"manifests"/f"{run_id}-{label}.jsonl"

def _stylize_call(engine,content,style,args):
    return stylize_panorama(engine,content,style,args.prompt,args.seed,args.steps,args.margin_px,args.blend_px,args.enable_polar_fusion,args.polar_rotation_degrees,args.polar_blend_start_degrees,args.polar_blend_end_degrees,args.polar_fusion_steps,args.polar_fusion_strength,args.polar_lowpass_radius_latent)

def run_batch(engine,content_paths,style_paths,output_dir,manifest_path,args,stylize_fn:Callable=_stylize_call)->list[dict]:
    output_dir.mkdir(parents=True,exist_ok=True); manifest_path.parent.mkdir(parents=True,exist_ok=True)
    metadata={path:style_metadata(path) for path in style_paths}; records=[]
    with manifest_path.open("x",encoding="utf-8") as manifest:
        for style_path in style_paths:
            for content_path in content_paths:
                target=output_path_for(output_dir,style_path,content_path); started=time.perf_counter()
                record={"content":str(content_path),"output":str(target),"seed":args.seed,**metadata[style_path]}
                try:
                    with Image.open(content_path) as image: content=image.convert("RGB")
                    with Image.open(style_path) as image: style=image.convert("RGB")
                    result,margin,working_size=stylize_fn(engine,content,style,args)
                    target.parent.mkdir(parents=True,exist_ok=True); result.save(target)
                    record.update({"status":"success","margin_px":margin,"model_canvas":list(working_size)})
                    print(f"Saved {target}")
                except Exception as error:
                    record.update({"status":"failed","error":f"{type(error).__name__}: {error}"})
                    print(f"Failed {content_path.name} x {style_path.name}: {record['error']}")
                record["seconds"]=round(time.perf_counter()-started,3); manifest.write(json.dumps(record,ensure_ascii=False)+"\n"); manifest.flush(); records.append(record)
    return records

def parse_args():
    parser=argparse.ArgumentParser(description="Batch seam-aware TeleStyle ERP stylization")
    parser.add_argument("--content-dir",type=Path,default=Path("inputs/zind_panos")); parser.add_argument("--style-dir",type=Path,default=Path("inputs/styles"))
    parser.add_argument("--styles",nargs="+",metavar="STYLE"); parser.add_argument("--output-dir",type=Path,default=Path("qwen_style_output/batch"))
    parser.add_argument("--prompt",default=DEFAULT_PROMPT); parser.add_argument("--seed",type=int,default=123); parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--margin-px",type=int,default=256); parser.add_argument("--blend-px",type=int,default=96)
    parser.add_argument("--enable-polar-fusion",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--polar-rotation-degrees",type=float,default=90.0); parser.add_argument("--polar-blend-start-degrees",type=float,default=45.0); parser.add_argument("--polar-blend-end-degrees",type=float,default=75.0)
    parser.add_argument("--polar-fusion-steps",type=int,default=2); parser.add_argument("--polar-fusion-strength",type=float,default=1.0); parser.add_argument("--polar-lowpass-radius-latent",type=int,default=8)
    return parser.parse_args()

def main():
    args=parse_args(); contents=discover_images(args.content_dir); styles=select_styles(discover_images(args.style_dir),args.styles); manifest=make_manifest_path(args.output_dir,styles)
    print(f"Generating {len(contents)*len(styles)} pairs")
    with torch.no_grad(): records=run_batch(ImageStyleInference(),contents,styles,args.output_dir,manifest,args)
    failed=sum(record["status"]=="failed" for record in records); print(f"Completed {len(records)-failed}/{len(records)} pairs | manifest={manifest}")
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
