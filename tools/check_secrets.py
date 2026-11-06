"""Scan all tracked working-tree files before push; never print matched values.
Source /tmp/.s/env.sh first so exact live credentials are checked too.
"""
import gzip
import os
from pathlib import Path
import re
import subprocess
import sys
paths = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
patterns = [rb'\bgh[pousr]_[A-Za-z0-9]{20,}', rb'\bgithub_pat_[A-Za-z0-9_]{30,}', rb'\bKGAT_[A-Za-z0-9_-]{20,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
secrets = [os.environ[k].encode() for k in ('GH_TOKEN', 'KAGGLE_API_TOKEN', 'GEMINI_API_KEY') if os.environ.get(k)]
failed=[]
for name in filter(None,paths):
 p=Path(name)
 if not p.is_file(): continue
 data=p.read_bytes()
 if p.suffix=='.gz': data=gzip.decompress(data)
 if any(s in data for s in secrets) or any(re.search(pattern,data) for pattern in patterns): failed.append(name)
if failed:
 print('FAIL: possible credentials in these paths only:', *failed, sep='\n'); sys.exit(1)
print(f'PASS: {sum(bool(p) for p in paths)} tracked paths; pattern checks; {len(secrets)} live credential values checked. Not a full history audit.')
