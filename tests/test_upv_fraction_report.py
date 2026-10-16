import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_upv_fraction import witness_gas_budgets


class ContactResolvedReport(unittest.TestCase):
    def test_pad_number_is_not_battery_number(self):
        scene=dict(ports=[dict(id='ISS1:+x',cell_id='ISS1'),dict(id='ISS3:+x',cell_id='ISS3')],
            measurement_probes=[dict(id='108',neighbor_port_id='ISS1:+x'),dict(id='101',neighbor_port_id='ISS3:+x')],
            gas_network=dict(compartments=[dict(id='left',local_equilibrium=dict(cell_id='ISS1')),
                                          dict(id='right',local_equilibrium=dict(cell_id='ISS3'))],
                             channels=[dict(id='left-feed',**{'from':'source','to':'left'}),
                                       dict(id='right-feed',**{'from':'source','to':'right'})]))
        run=dict(channels={'left-feed':dict(delivered_gas_energy_j=[11]),'right-feed':dict(delivered_gas_energy_j=[33])},
                 gas={'left':dict(released_energy_j=[1]),'right':dict(released_energy_j=[3])})
        b=witness_gas_budgets(scene,run)
        self.assertEqual(b['108']['material_cell_id'],'ISS1')
        self.assertEqual(b['108']['net_advected_energy_j'],11)
        self.assertEqual(b['108']['gas_reaction_j'],1)
        self.assertEqual(b['101']['material_cell_id'],'ISS3')
        self.assertEqual(b['101']['net_advected_energy_j'],33)


if __name__=='__main__':unittest.main()
