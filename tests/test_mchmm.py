import mchmm._mc as mc
import mchmm._hmm as hmm
import numpy as np
import unittest


class TestMC(unittest.TestCase):

    seq = 'CCCAAAAAACCCCCAACCDDCCBBBBBCCCBBCCCCCAAAAACCAAACCDCCDDCAAAAAAC'

    def test_tfm(self):
        '''Checking transition frequency matrix correctness'''
        result = np.array([[17,0,5,0],
                           [0,5,2,0],
                           [5,2,17,3],
                           [0,0,3,2]])

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.all(a.observed_matrix == result))




if __name__ == '__main__':
    unittest.main()
