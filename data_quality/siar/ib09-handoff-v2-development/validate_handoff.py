"""Validate the versioned IB09 DEVELOPMENT exception, without building a dataset."""
import hashlib, importlib.util, json
from pathlib import Path

SCHEMA_SHA256='6bf3d8bccdcf2dc4360b9ba3d5e304b57d3c5acac3402d3ce637fddac9835185'
class HandoffError(ValueError): pass

def require(condition, message):
    if not condition: raise HandoffError(message)

def schema_check(value,schema,path='$'):
    if 'const' in schema:
        expected=schema['const']; require(type(value) is type(expected) and value==expected,'schema mismatch: '+path)
    if schema.get('type')=='object':
        require(isinstance(value,dict) and set(value)==set(schema['required']),'schema keys: '+path)
        for k,v in schema['properties'].items(): schema_check(value[k],v,path+'.'+k)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(handoff, root, *, for_admission=False, scope=None, official_evidence_contradiction=None):
    root=Path(root); here=Path(__file__).resolve().parent
    schema_bytes=(here/'schema.json').read_bytes().replace(b'\r\n',b'\n')
    require(hashlib.sha256(schema_bytes).hexdigest()==SCHEMA_SHA256,'schema hash mismatch')
    schema_check(handoff,json.loads(schema_bytes))
    prev=handoff['supersedes']; prev_path=root/prev['path']; require(sha(prev_path)==prev['sha256'],'superseded handoff hash mismatch')
    decision=handoff['admission']['decision_ref']; require(sha(root/decision['path'])==decision['sha256'],'decision hash mismatch')
    # Reuse the preserved PENDING validator to check raw, catalogue, ledger,
    # POST, prototype, duplicates, outside row and gap reconciliation.
    prior_module_path=root/'data_quality/siar/ib09-handoff-v2/validate_handoff.py'
    require(hashlib.sha256(prior_module_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() == '82ab7358c935bae6ffe5e466570e97972a370c9c9aa87c86344cb313424fc1bd', 'prior validator hash mismatch')
    spec=importlib.util.spec_from_file_location('ib09_pending_validator',prior_module_path)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    prior.validate_handoff(json.loads(prev_path.read_text(encoding='utf-8')),root)
    if for_admission:
        require(scope=='DEVELOPMENT_ONLY','scope not authorized')
        require(official_evidence_contradiction is False,'contradiction present or evidence review unspecified')
    return {'valid':True,'scope':'DEVELOPMENT_ONLY','admission':'APPROVED','official_period_verification':'UNRESOLVED','dataset_row_binding':'CONSUMER_REQUIRED','campaign_launch_authorized':False}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--for-admission',action='store_true');p.add_argument('--scope');p.add_argument('--official-evidence-reviewed-no-contradiction',action='store_true');a=p.parse_args()
    h=json.loads((Path(__file__).resolve().parent/'handoff.json').read_text(encoding='utf-8'))
    print(json.dumps(validate(h,a.root,for_admission=a.for_admission,scope=a.scope,official_evidence_contradiction=False if a.official_evidence_reviewed_no_contradiction else None),sort_keys=True))
if __name__=='__main__': main()
