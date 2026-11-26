import unittest
from unittest.mock import patch
from aemet_forecast_collector.client import AemetClient
from aemet_forecast_collector.transport import HttpResponse
class RetryTests(unittest.TestCase):
 def run_case(self, answers):
  calls=[]
  def fetch(*a): calls.append(a); return answers[min(len(calls)-1,len(answers)-1)]
  with patch('time.sleep') as sleep:
   result=AemetClient(None,fetcher=fetch)._request('https://example.invalid',{})
  return result,calls,sleep
 def test_429_waits_once(self):
  result,calls,sleep=self.run_case([HttpResponse(429,{},b''),HttpResponse(200,{},b'ok')])
  self.assertEqual(result.status_code,200); self.assertEqual(len(calls),2); sleep.assert_called_once_with(120)
 def test_retry_after_respected(self):
  _,calls,sleep=self.run_case([HttpResponse(429,{'Retry-After':'300'},b''),HttpResponse(200,{},b'ok')])
  self.assertEqual(len(calls),2); sleep.assert_called_once_with(300)
 def test_too_long_no_retry(self):
  result,calls,sleep=self.run_case([HttpResponse(429,{'Retry-After':'900'},b'')])
  self.assertEqual(result.status_code,429); self.assertEqual(len(calls),1); sleep.assert_not_called()
 def test_bad_header_no_retry(self):
  result,calls,sleep=self.run_case([HttpResponse(429,{'Retry-After':'bad'},b'')])
  self.assertEqual(len(calls),1); sleep.assert_not_called()
 def test_timeout_and_client_error(self):
  _,calls,sleep=self.run_case([HttpResponse(None,{},b'', 'timeout'),HttpResponse(200,{},b'ok')]); self.assertEqual(len(calls),2); sleep.assert_called_once_with(30)
  _,calls,sleep=self.run_case([HttpResponse(401,{},b'')]); self.assertEqual(len(calls),1); sleep.assert_not_called()
if __name__=='__main__': unittest.main()
