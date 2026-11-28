import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent))
from emissions import summarize
from backup_status import classify

def cap(t,issue,status='complete'):
 return {'capture':{'started_at_utc':t,'status':status},'source':{'issue_time_raw':issue,'product':'daily'},'municipality':{'id':'07040'}}
a=summarize([cap('1','e1'),cap('2','e1'),cap('3','e2'),cap('4',None,'partial')])
assert [r['repeated_emission'] for r in a]==[False,True,False,None]
r='run_abc'; events=[{'MESSAGE':'backup verified: '+r,'__REALTIME_TIMESTAMP':'1'},{'MESSAGE':'backup pending: '+r,'__REALTIME_TIMESTAMP':'2'}]
assert classify([r,'run_new'],events)[r]['state']=='pending'
assert classify([r,'run_new'],events)['run_new']['state']=='unknown'
print('synthetic emission and backup checks: PASS')
