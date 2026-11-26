"""Mapped sources preserve pixels, close on failures, and refuse inadequate space."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import tempfile,sys,json
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Resources/Tools'))
from scan_memory import MappedScanStore,MappedArrayStore,render_tile_size,GiB
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
a=np.arange(40*32*3,dtype=np.uint8).reshape(40,32,3);paths=[]
for k in range(2):
 p=root/f'{k}.png';Image.fromarray(a+k).save(p);paths.append(str(p))
calls=[]
def loader(path,scale):
 calls.append(path)
 with Image.open(path) as image:return np.array(image)
with patch('scan_memory.shutil.disk_usage',return_value=SimpleNamespace(free=20*GiB)):
 with MappedScanStore(paths,[0,1],loader,scratch=root) as store:
  assert len(calls)==2
  assert np.array_equal(store[0],a) and np.array_equal(store[1],a+1)
  assert not store[0].flags.writeable
  store.drop_pages();assert np.array_equal(store[0],a)
  assert sorted(p.name for p in root.iterdir() if p.name!='RESULTAAT.json')==['0.png','1.png']
 assert not store.maps and not store.arrays
 def failing(path,scale):
  if path==paths[1]:raise RuntimeError('intentional')
  return loader(path,scale)
 failed=MappedScanStore(paths,[0,1],failing,scratch=root)
 try:
  with failed:raise AssertionError('Expected failure')
 except RuntimeError:pass
 assert not failed.maps and not failed.arrays
with patch('scan_memory.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
 try:MappedScanStore(paths,[0],loader,scratch=root);raise AssertionError('Expected space check')
 except ValueError as error:assert 'disk space' in str(error)
with patch('scan_memory.physical_memory',return_value=8*GiB):assert render_tile_size()==1024
with patch('scan_memory.physical_memory',return_value=16*GiB):assert render_tile_size()==2048
# Total source pixels are not simultaneous decoder memory. Five A4 scans at
# 1200 dpi exceed 700 Mpx but fit the actual sequential decoder and disk guards.
class Header:
 width,height=10176,14016
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def close(self):pass
with patch('scan_memory.Image.open',return_value=Header()), \
     patch('scan_memory.physical_memory',return_value=8*GiB), \
     patch('scan_memory.shutil.disk_usage',return_value=SimpleNamespace(free=100*GiB)):
 large=MappedScanStore(['source']*5,range(5),loader,scratch=root)
 assert len(large.shapes)==5
 Header.width=Header.height=50000
 try:MappedScanStore(['source'],[0],loader,scratch=root);raise AssertionError('Expected decoder memory check')
 except ValueError as error:assert 'too large to open safely' in str(error)
 Header.width,Header.height=10176,14016
with patch('scan_memory.Image.open',return_value=Header()), \
     patch('scan_memory.physical_memory',return_value=8*GiB), \
     patch('scan_memory.shutil.disk_usage',return_value=SimpleNamespace(free=2*GiB)):
 try:MappedScanStore(['source']*5,range(5),loader,scratch=root);raise AssertionError('Expected real scratch-space check')
 except ValueError as error:assert 'disk space' in str(error)
# Removing a total-native-pixel cap must not silently cap the memory estimate
# and then decode an arbitrarily large number of previews.
import scan_stitch
with patch('PIL.Image.open',return_value=Header()), \
     patch.object(sys,'argv',['scan_stitch','unused.png']+['source']*100+['--toon-format','vinyl']), \
     patch('scan_stitch.load',side_effect=AssertionError('Must refuse before decoding')), \
     patch.dict('os.environ',{'SLEEVE_MEMORY_SLOTS':'72'}):
 try:scan_stitch.main();raise AssertionError('Expected preparation budget check')
 except ValueError as error:assert 'available memory' in str(error)
with MappedArrayStore(4096,scratch=root) as work:
 value=np.arange(90,dtype=np.float32).reshape(10,9)
 stored=work.put('image',value)
 assert np.array_equal(stored,value)
 zeros=work.empty('other',(10,9));assert not zeros.any()
 zeros[:]=value*2;work.drop_pages();assert np.array_equal(zeros,value*2)
 del zeros
 work.keep_only({'image'});assert set(work.maps)=={'image'}
 del stored
 work.discard('image');assert not work.maps
 assert sorted(p.name for p in root.iterdir() if p.name!='RESULTAAT.json')==['0.png','1.png']
assert not work.maps and not work.arrays
with patch('scan_memory.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
 try:MappedArrayStore(4096,scratch=root);raise AssertionError('Expected space check')
 except ValueError as error:assert 'disk space' in str(error)
checks={'work_buffers_preserve_pixels_and_cleanup':True,'source_pixels_exact' :True,'decode_once_per_source':True,'scratch_is_anonymous_and_closed':True,'failure_closes_scratch':True,'insufficient_space_rejected':True,'tile_size_matches_memory_budget':True,'large_group_checked_against_real_resources':True,'oversized_single_source_rejected':True}
(root/'RESULTAAT.json').write_text(json.dumps(checks,indent=2));print(checks)
