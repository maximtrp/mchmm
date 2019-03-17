import itertools as iter
import numpy as np
import re
import scipy.stats as ss


class HiddenMarkovModel:

    def __init__(self, obs_seq=None, states=None, tp=None, ep=None, pi=None):
        '''Hidden Markov model.

        Parameters
        ----------
        obs_seq : str, array_like or numpy ndarray
            Observations sequence (of size T).

        states : array_like or numpy ndarray
            States list (of size K).

        tp : array_like or numpy ndarray
            Transition matrix of size K × K which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : array_like or numpy ndarray
            Emission matrix of size K × N which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space O = {o_1, o_2, ..., o_N}.

        pi : array_like or numpy ndarray
            Initial probabilities array (of size K).

        '''

        self.obs_seq = np.array(obs_seq)
        self.states = np.array(states)
        self.tp = np.array(tp)
        self.ep = np.array(ep)
        self.pi = np.array(pi)

    def _transition_matrix(self, seq, states=None):
        '''Calculate a transition probability matrix.

        Parameters
        ----------
        seq : str or array_like
            A string or an array-like object exposing the array interface and
            containing strings or ints.

        states : numpy ndarray
            Array containing a list of states.

        Returns
        -------
        matrix : numpy ndarray
            Transition frequency matrix.

        '''

        seql = np.array(list(seq))
        if not states:
            states = np.unique(seql)
        matrix = np.zeros((len(states), len(states)))

        for x, y in iter.product(range(len(states)), repeat=2):
            xid = np.argwhere(seql == states[x]).flatten()
            yid = xid + 1
            yid = yid[yid < len(seql)]
            s = np.count_nonzero(seql[yid] == states[y])
            matrix[x, y] = s

        return matrix


    def _emission_matrix(self, obs_seq, states_seq):
        '''Calculate an emission probability matrix.

        Parameters
        ----------
        obs_seq : str or array_like
            Sequence of observations (of size T).
            Observation space = {o_1, o_2, ..., o_N}.

        states_seq : str or array_like
            Sequence of states (of size T).
            State space = {s_1, s_2, ..., s_K}.

        Returns
        -------
        result : numpy ndarray
            Emission probability matrix of size K × N.

        '''

        _os = np.array(list(obs_seq))
        _ss = np.array(list(states_seq))

        obs_space = np.unique(_os)
        states_space = np.unique(_ss)
        k = states_space.size
        n = obs_space.size

        ef = np.zeros((k, n))

        for i in range(k):
            for j in range(n):
                o = _os[_ss == states_space[i]]
                ef[i,j] = np.count_nonzero(o == obs_space[j])

        ep = ef / ef.sum(axis=1)
        return ep

    def from_data(self, obs_seq, states_seq, pi=None, pi_seed=None):

        self.tp = self._transition_matrix(obs_seq)
        self.ep = self._emission_matrix(obs_seq, states_seq)
        self.states = np.unique(list(states_seq))

        if not pi:
            self.pi = ss.uniform().rvs(size=self.states.size, random_state=pi_seed)

        return self


    def viterbi(self, obs_seq, states=None, tp=None, ep=None, pi=None,):
        '''Viterbi algorithm.

        Parameters
        ----------

        '''

        if not states:
            states = self.states

        if not tp:
            tp = self.tp

        if not ep:
            ep = self.ep

        if not pi:
            pi = self.pi

        obs_seq = np.array(list(obs_seq))
        T = obs_seq.size

        def s(i):
            return np.asscalar(np.argwhere(states == obs_seq[i]).flatten())

        t1[:, 0] = pi * ep[:, s(0)]
        t1[:, 0] /= t1[:, 0].sum()
        t2[:, 0] = 0

        for i in range(1, T):
            t1[:, i] = np.max(t1[:, i-1] * tp * ep[:, s(i)], axis=1)
            t2[:, i] = np.argmax(t1[:, i-1] * tp * ep[:, s(i)], axis=1)
            t1[:, i] /= t1[:, i].sum()

        z = np.argmax(t1, axis=0)
        x = states[z]

        for i in reversed(range(1, T)):
            z[i-1] = t2[z[i], i]
            x[i-1] = states[z[i-1]]

        return x, z
