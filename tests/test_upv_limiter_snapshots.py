from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_limiter_snapshots import summarize


class LimiterSummary(unittest.TestCase):
    def fixture(self):
        def edge(a,b,donor,raw,pair,accepted):
            return dict(id=a+'--'+b, **{'from':a, 'to':b}, donor=donor,
                raw_mass_kg=raw, pair_request_kg=pair, accepted_mass_kg=accepted,
                mix_request_each_way_kg=2., mix_accepted_each_way_kg=1.)
        return dict(window_s=.01, nodes=[dict(id='slit', pressure_scale=.4, mixing_scale=.5)],
            edges=[edge('outside','slit','outside',10.,4.,1.), edge('slit','far','slit',8.,2.,1.),
                   edge('elsewhere','far','elsewhere',100.,100.,100.)])

    def test_stages_are_distinct_and_boundary_is_local(self):
        row=summarize(self.fixture(), {'slit'})
        self.assertEqual(row['incoming']['raw_kg_s'], 1000.)
        self.assertEqual(row['incoming']['after_pair_kg_s'], 400.)
        self.assertEqual(row['incoming']['accepted_kg_s'], 100.)
        self.assertEqual(row['incoming']['pair_acceptance'], .4)
        self.assertEqual(row['incoming']['node_acceptance'], .25)
        self.assertEqual(row['outgoing']['raw_kg_s'], 800.)
        self.assertEqual(row['mixing_budget_acceptance'], .5)
        self.assertEqual(len(row['top_pressure_bottlenecks']), 2)

    def test_no_flow_has_no_misleading_zero_ratio(self):
        result=self.fixture();result['edges']=[]
        row=summarize(result, {'slit'})
        self.assertIsNone(row['incoming']['pair_acceptance'])
        self.assertIsNone(row['mixing_budget_acceptance'])


if __name__=='__main__':unittest.main()
