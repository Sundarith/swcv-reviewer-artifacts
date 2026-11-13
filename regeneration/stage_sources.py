"""Materialize the recorded source layout; does not execute experiments."""
import argparse,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--asset-root',type=Path,required=True);a=p.parse_args();root=a.workspace.resolve()
for item in json.loads((H/'source_manifest.json').read_text()):
 src=H/item['public_path'];assert hashlib.sha256(src.read_bytes()).hexdigest()==item['public_sha256'];dest=root/item['original_path'];assert not dest.exists(),str(dest);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(src.read_text().replace('__EXPERIMENT_WORKSPACE__',str(root)).replace('__LOCAL_ASSET_ROOT__',str(a.asset_root.resolve())))
print('Sources staged. No model loaded, inference run or gold labels scored.')
