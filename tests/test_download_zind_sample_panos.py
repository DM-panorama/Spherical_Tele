import base64,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from download_zind_sample_panos import download_panos
class DownloadTests(unittest.TestCase):
 def test_download_and_overwrite(self):
  listing=[{"name":"b.jpg","type":"file","content":base64.b64encode(b"b").decode()},{"name":"a.jpg","type":"file","content":base64.b64encode(b"a").decode()}]
  with tempfile.TemporaryDirectory() as temp,patch("download_zind_sample_panos.request_json",return_value=listing):
   output=Path(temp); self.assertEqual([p.name for p in download_panos(output)],["a.jpg","b.jpg"]); (output/"a.jpg").write_bytes(b"old"); self.assertEqual(download_panos(output),[]); download_panos(output,True); self.assertEqual((output/"a.jpg").read_bytes(),b"a")
if __name__=="__main__": unittest.main()
