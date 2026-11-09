"""Independent service measurements across user arrivals and departures."""

import unittest

from evaluate import evaluate_service


class ServiceEvaluationTest(unittest.TestCase):
    def test_pair_keeps_history_when_a_third_user_arrives(self):
        requests = [
            {'user':'a','weight':2.,'submitted_s':0.,'finished_s':10.},
            {'user':'b','weight':1.,'submitted_s':0.,'finished_s':10.},
            {'user':'c','weight':1.,'submitted_s':1.5,'finished_s':10.},
        ]
        work = {'user':'a','query_tokens':1,'context_tokens':8,'prompt_tokens':8}
        batches = [{'completed_at':t,'work':[work]} for t in (1.,2.)]
        service,gap = evaluate_service(requests,batches,{(1,8,8):6.},0.)
        self.assertEqual(service,{'a':12.,'b':0.,'c':0.})
        self.assertEqual(gap,6.)

    def test_returning_pair_starts_a_new_interval(self):
        requests = [
            {'user':'a','weight':1.,'submitted_s':0.,'finished_s':1.5},
            {'user':'a','weight':1.,'submitted_s':2.5,'finished_s':10.},
            {'user':'b','weight':1.,'submitted_s':0.,'finished_s':10.},
        ]
        work = {'user':'a','query_tokens':1,'context_tokens':8,'prompt_tokens':8}
        batches = [
            {'completed_at':1.,'work':[work]},
            {'completed_at':3.,'work':[work]},
        ]
        service,gap = evaluate_service(requests,batches,{(1,8,8):6.},0.)
        self.assertEqual(service['a'],12.)
        self.assertEqual(gap,6.)


if __name__ == '__main__':
    unittest.main()
