import mchmm._mc as mc
import mchmm._hmm as hmm
import numpy as np
import unittest


class TestMC(unittest.TestCase):

    seq = 'ABCABCABCACBCBA'

    def test_tfm(self):
        '''Checking transition frequency matrix'''
        result = np.array([[0,3,1],
                           [1,0,4],
                           [3,2,0]])

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.all(a.observed_matrix == result))

    def test_tpm(self):
        '''Checking transition probability matrix'''
        tfm = np.array([[0,3,1], [1,0,4], [3,2,0]])
        result = tfm / tfm.sum(axis=1)[:, None]

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.all(a.observed_p_matrix == result))




if __name__ == '__main__':
    unittest.main()
