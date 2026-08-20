"""Download ZInD sample_tour/000 panoramas from GitHub."""
import argparse,base64,json
from pathlib import Path
from urllib.request import Request,urlopen
API_URL="https://api.github.com/repos/zillow/zind/contents/sample_tour/000/panos"
def request_json(url):
    with urlopen(Request(url,headers={"Accept":"application/vnd.github+json","User-Agent":"TeleStyle-ZInD-downloader"})) as response: return json.load(response)
def download_panos(output_dir:Path,overwrite=False,api_url=API_URL):
    listing=request_json(api_url)
    if not isinstance(listing,list): raise ValueError("Expected a directory listing from the GitHub Contents API.")
    output_dir.mkdir(parents=True,exist_ok=True); downloaded=[]
    for item in sorted(listing,key=lambda entry:entry.get("name","").casefold()):
        name=item.get("name","")
        if item.get("type")!="file" or Path(name).suffix.lower() not in {".jpg",".jpeg"}: continue
        target=output_dir/name
        if target.exists() and not overwrite: print(f"Keeping {target}"); continue
        encoded=item.get("content")
        if encoded is None: encoded=request_json(item["url"]).get("content")
        if encoded is None: raise ValueError(f"GitHub did not provide content for {name}.")
        target.write_bytes(base64.b64decode(encoded)); downloaded.append(target); print(f"Downloaded {target}")
    return downloaded
def main():
    parser=argparse.ArgumentParser(description="Download ZInD sample_tour/000 panorama JPEGs"); parser.add_argument("--output-dir",type=Path,default=Path("inputs/zind_panos")); parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(); download_panos(args.output_dir,args.overwrite)
if __name__=="__main__": main()
