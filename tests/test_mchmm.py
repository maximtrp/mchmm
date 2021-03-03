import mchmm._mc as mc
import mchmm._hmm as hmm
import numpy as np
import unittest
from graphviz import Digraph


class TestMC(unittest.TestCase):

    seq = 'ABCABCABCACBCBA'

    # Markov chains tests
    def test_tfm(self):
        """Checking transition frequency matrix."""
        result = np.array([[0, 3, 1],
                           [1, 0, 4],
                           [3, 2, 0]])

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.allclose(a.observed_matrix, result))

    def test_prf(self):
        """Checking prob_to_freq method."""
        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.allclose(a.prob_to_freq_matrix(), a.observed_matrix))

    def test_tpm(self):
        """Checking transition probability matrix."""
        tfm = np.array([
            [0, 3, 1],
            [1, 0, 4],
            [3, 2, 0]
        ], dtype=float)
        result = tfm / tfm.sum(axis=1)[:, None]

        a = mc.MarkovChain().from_data(self.seq)
        self.assertTrue(np.allclose(a.observed_p_matrix, result))

    def test_nom(self):
        """Checking raising a matrix to power N."""
        a = mc.MarkovChain().from_data(self.seq)
        a2 = a.n_order_matrix()
        self.assertTrue(np.allclose(a2, a.observed_p_matrix @ a.observed_p_matrix))

    def test_mcsim(self):
        """Checking simulation process."""
        a = mc.MarkovChain().from_data(self.seq)
        seed = np.random.randint(0, 15, 15)
        si, sn = a.simulate(15, start=0, ret='both', seed=seed)
        si2 = a.simulate(15, tf=a.observed_matrix, start=0, ret='indices', seed=seed)
        sn2 = a.simulate(15, start=0, states=['A', 'B', 'C'], ret='states', seed=seed)
        si3, sn3 = a.simulate(15, start='A', ret='both', seed=seed)
        si4, sn4 = a.simulate(15, start='B', ret='both', seed=seed)
        si5, sn5 = a.simulate(15, ret='both', seed=seed)
        
        self.assertTrue(np.allclose(si, si2) and np.all(sn == sn2))
        self.assertTrue(np.allclose(si, si3) and np.all(sn == sn3))
        self.assertTrue(np.allclose(si4, si5) and np.all(sn4 == sn5))

    def test_mcgraph(self):
        """Checking graphing process for MC."""
        a = mc.MarkovChain().from_data(self.seq)
        graph = a.graph_make()
        self.assertTrue(isinstance(graph, Digraph))

    # HMM tests
    def test_epm(self):
        """Checking emission probability matrix."""
        result = np.array([[0.21052632, 0.21052632, 0.47368421, 0.10526316],
                           [0.57142857, 0., 0., 0.42857143]])

        obs_seq = 'AGACTGCATATATAAGGGGCAGGCTG'
        sts_seq = '00000000111111100000000000'
        a = hmm.HiddenMarkovModel().from_seq(obs_seq, sts_seq)
        self.assertTrue(np.allclose(a.ep, result))

    def test_vit(self):
        """Checking Viterbi."""

        obs_seq = 'AGACTGCATATATAAGGGGCAGGCTG'
        sts_seq = '00000000111111100000000000'
        a = hmm.HiddenMarkovModel().from_seq(obs_seq, sts_seq)
        b = a.viterbi(obs_seq)
        self.assertTrue(
            isinstance(b[0], np.ndarray) and isinstance(b[1], np.ndarray)
        )

    def test_bw(self):
        """Checking Baum-Welch."""

        obs_seq = 'AGACTGCATATATAAGGGGCAGGCTG'
        a = hmm.HiddenMarkovModel().from_baum_welch(obs_seq, states=['0', '1'])
        self.assertTrue(
            isinstance(a, hmm.HiddenMarkovModel)
        )

    def test_hmmgraph(self):
        """Checking graphing process for HMM."""
        obs_seq = 'AGACTGCATATATAAGGGGCAGGCTG'
        sts_seq = '00000000111111100000000000'
        a = hmm.HiddenMarkovModel().from_seq(obs_seq, sts_seq)
        graph = a.graph_make()
        self.assertTrue(isinstance(graph, Digraph))


if __name__ == '__main__':
    unittest.main()
