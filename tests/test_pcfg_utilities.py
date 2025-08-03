import unittest
from pcfg_utilities.pcfg_utilities import characteristic_matrix, derivational_entropy, mean_length
from nltk.grammar import PCFG

class TestPCFGUtilities(unittest.TestCase):
    def setUp(self):
        self.grammar = PCFG.fromstring("""
            S -> NP VP [1.0]
            NP -> 'John' [0.5] | 'Mary' [0.5]
            VP -> V NP [1.0]
            V -> 'sees' [1.0]
        """)

    def test_characteristic_matrix(self):
        M, nts = characteristic_matrix(self.grammar)
        self.assertEqual(M.shape[0], len(nts))

    def test_derivational_entropy(self):
        ent = derivational_entropy(self.grammar)
        self.assertIn('NP', {str(nt) for nt in ent.keys()})

    def test_mean_length(self):
        ml = mean_length(self.grammar)
        self.assertTrue(all(val > 0 for val in ml.values()))

if __name__ == "__main__":
    unittest.main()

