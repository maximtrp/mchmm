import mchmm._mc as mc
import mchmm._hmm as hmm
import numpy as np
import unittest


class TestMC(unittest.TestCase):

    seq = 'ABCABCABCACBCBA'

    # Markov chains tests
    def test_tfm(self):
        '''Checking transition frequency matrix'''
        result = np.array([[0,3,1],
                           [1,0,4],
                           [3,2,0]])

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.all(a.observed_matrix == result))

    def test_tpm(self):
        '''Checking transition probability matrix'''
        tfm = np.array([[0,3,1], [1,0,4], [3,2,0]], dtype=np.float)
        result = tfm / tfm.sum(axis=1)[:, None]

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.all(a.observed_p_matrix == result))

    def test_mcsim(self):
        '''Checking simulation process'''
        a = mc.MarkovChain().from_data(self.seq)
        si, sn = a.simulate(15, start=0, ret='both', seed=11)
        si2 = a.simulate(15, start=0, ret='indices', seed=11)
        sn2 = a.simulate(15, start=0, ret='states', seed=11)

        self.assertTrue(np.all(si == si2) and np.all(sn == sn2))

    # HMM tests


if __name__ == '__main__':
    unittest.main()
