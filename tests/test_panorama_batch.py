import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from telestylepanorama_batch import discover_images,make_manifest_path,output_path_for,run_batch,select_styles
class BatchTests(unittest.TestCase):
 def image(self,path,color): Image.new("RGB",(8,4),color).save(path)
 def test_discovery_selection_and_batch(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp); contents=root/"contents"; styles=root/"styles"; output=root/"output"; contents.mkdir(); styles.mkdir()
   first,second,style=contents/"first.png",contents/"second.png",styles/"style1.png"; self.image(first,(255,0,0)); self.image(second,(0,0,255)); self.image(style,(0,255,0))
   self.assertEqual([p.name for p in discover_images(contents)],["first.png","second.png"]); self.assertEqual(select_styles([style],["STYLE1"]),[style])
   def fake(engine,content,style,args):
    if content.getpixel((0,0))==(0,0,255): raise RuntimeError("synthetic")
    return content,16,(16,32)
   manifest=make_manifest_path(output,[style],"test"); records=run_batch(object(),[first,second],[style],output,manifest,SimpleNamespace(seed=123),fake)
   self.assertEqual([r["status"] for r in records],["success","failed"]); self.assertTrue(output_path_for(output,style,first).is_file())
   self.assertEqual([json.loads(line) for line in manifest.read_text().splitlines()],records)
if __name__=="__main__": unittest.main()
