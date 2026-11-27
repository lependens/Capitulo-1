"""Read-only batch assessment. Station codes are candidates until verified."""
from pathlib import Path
from collections import Counter
from datetime import date
import json
import unicodedata
import re
import sys
from andalucia_stage import parse_file, HEADER

def main(root, output):
    root=Path(root); result=[]
    aliases={'AL':'AL','HU':'H','JA':'J','MA':'MA','SE':'SE'}
    for f in sorted(root.glob('*/*.txt')):
        if f.name in {'server.txt','unionsimple.txt'}: continue
        with f.open(encoding='utf-8-sig') as stream: first=stream.readline()
        m=HEADER.match(first)
        if not m:
            result.append({'file':str(f.relative_to(root)),'status':'unidentified_or_other_format'}); continue
        label=unicodedata.normalize('NFKD',m.group(1)).encode('ascii','ignore').decode().upper()
        p=re.fullmatch(r'([A-Z]+)(\d+)',label)
        if not p or p[1] not in aliases:
            result.append({'file':str(f.relative_to(root)),'header':label,'status':'unknown_header'}); continue
        station=aliases[p[1]]+p[2]
        try:
            data=parse_file(f,station,2000,header_label=label,max_date=date(2020,6,5))
            result.append({'file':str(f.relative_to(root)),'header':label,'candidate_station':station,'status':'candidate_unverified','sha256':data['source_sha256'],'candidate_rows':len(data['rows']),'exceptions':dict(Counter(e['reason'] for e in data['exceptions']))})
        except (UnicodeError,ValueError) as e:
            result.append({'file':str(f.relative_to(root)),'header':label,'status':'parse_failure','error':str(e)})
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('files',len(result),'statuses',dict(Counter(r['status'] for r in result)))
    print('candidate_rows',sum(r.get('candidate_rows',0) for r in result))
    print('exceptions',dict(sum((Counter(r.get('exceptions',{})) for r in result),Counter())))

if __name__=='__main__': main(sys.argv[1],sys.argv[2])

