__all__ = ['HiddenMarkovModel']

import numpy as np
import scipy.stats as ss
from itertools import product
from graphviz import Digraph


class HiddenMarkovModel:

    def __init__(
        self, observations=None, states=None, tp=None, ep=None, pi=None
    ):
        '''Hidden Markov model.

        Parameters
        ----------
        observations : array_like or numpy.ndarray
            Observations space (of size N).

        states : array_like
            List of states (of size K).

        tp : array_like or numpy.ndarray
            Transition matrix of size K × K which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : array_like or numpy.ndarray
            Emission matrix of size K × N which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space O = {o_1, o_2, ..., o_N}.

        pi : array_like or numpy.ndarray
            Initial state probabilities array (of size K).

        '''

        self.observations = np.array(observations)
        self.states = np.array(states)
        self.tp = np.array(tp)
        self.ep = np.array(ep)
        self.pi = np.array(pi)

    def _transition_matrix(self, seq=None, states=None):
        '''Calculate a transition probability matrix which stores transition
        probability of transiting from state i to state j.

        Parameters
        ----------
        seq : str or array_like
            Sequence of states.

        states : array_like
            List of unique states.

        Returns
        -------
        matrix : numpy.ndarray
            Transition frequency matrix.

        '''

        seql = self.states_seq if seq is None else np.array(list(seq))
        if states is None:
            states = self.states
        T = len(seql)
        K = len(states)

        matrix = np.zeros((K, K))

        for x, y in product(range(K), repeat=2):
            xid = np.argwhere(seql == states[x]).flatten()
            yid = xid + 1
            yid = yid[yid < T]
            s = np.count_nonzero(seql[yid] == states[y])
            matrix[x, y] = s

        matrix /= matrix.sum(axis=1)[:, None]
        return matrix

    def _emission_matrix(
        self, obs_seq=None, states_seq=None, obs=None, states=None
    ):
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
        ep : numpy.ndarray
            Emission probability matrix of size K × N.

        '''

        _os = self.obs_seq if obs_seq is None else np.array(list(obs_seq))
        _ss = self.states_seq if states_seq is None else np.array(
            list(states_seq))

        obs_space = np.unique(_os) if obs is None else np.sort(
            np.array(list(obs)))
        states_space = np.unique(_ss) if states is None else np.sort(
            np.array(list(states)))
        k = states_space.size
        n = obs_space.size

        ef = np.zeros((k, n))

        for i in range(k):
            for j in range(n):
                o = _os[_ss == states_space[i]]
                ef[i, j] = np.count_nonzero(o == obs_space[j])

        ep = ef / ef.sum(axis=1)[:, None]
        return ep

    def from_seq(self, obs_seq, states_seq, pi=None, end=None, seed=None):
        '''Analyze sequences of observations and states.

        Parameters
        ----------
        obs_seq : str or array_like
            Sequence of observations (of size T).
            Observation space = {o_1, o_2, ..., o_N}.

        states_seq : str or array_like
            Sequence of states (of size T).
            State space = {s_1, s_2, ..., s_K}.

        pi : None, array_like or numpy.ndarray, optional
            Initial state probabilities array (of size K). If None, array is
            sampled from a uniform distribution.

        pi_seed : int, optional
            Random state used to draw random variates. Passed to
            `scipy.stats.uniform` method.

        Returns
        -------
        model : object
            Hidden Markov model learned from the given data.
        '''

        self.obs_seq = np.array(list(obs_seq))
        self.observations = np.unique(self.obs_seq)
        self.states_seq = np.array(list(states_seq))
        self.states = np.unique(self.states_seq)
        self.tp = self._transition_matrix(self.states_seq, self.states)
        self.ep = self._emission_matrix(self.obs_seq, self.states_seq)

        if pi is None:
            self.pi = ss.uniform().rvs(
                size=self.states.size, random_state=seed
            )
            self.pi /= self.pi.sum()

        if end is None:
            self.end = ss.uniform().rvs(
                size=self.states.size, random_state=seed
            )
            self.end /= self.end.sum()

        return self

    def viterbi(
        self, obs_seq, obs=None, states=None, tp=None, ep=None, pi=None
    ):
        '''Viterbi algorithm.

        Parameters
        ----------
        obs_seq : array_like
            Sequence of observations.

        obs : array_like, optional
            Observations space (of size N).

        states : array_like, optional
            List of states (of size K).

        tp : array_like or numpy.ndarray, optional
            Transition matrix (of size K × K) which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : array_like or numpy.ndarray, optional
            Emission matrix (of size K × N) which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space, O = {o_1, o_2, ..., o_N}.

        pi : array_like or numpy.ndarray, optional
            Initial probabilities array (of size K).

        Returns
        -------
        x : numpy.ndarray
            Sequence of states.

        z : numpy.ndarray
            Sequence of state indices.
        '''

        if states is None:
            states = self.states

        if tp is None:
            tp = self.tp

        if ep is None:
            ep = self.ep

        if pi is None:
            pi = self.pi
        else:
            pi = np.array(pi)

        obs_seq = np.array(list(obs_seq))

        if obs is None:
            obs = np.unique(obs_seq)

        T = len(obs_seq)
        K = len(states)

        def s(i):
            return np.argwhere(obs == obs_seq[i]).flatten().item()

        t1 = np.zeros((K, T))
        t2 = np.zeros((K, T))

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

    def from_baum_welch(
        self, obs_seq, states, thres=0.001, obs=None,
        tp=None, ep=None, pi=None, end=None
    ):
        '''Baum-Welch algorithm.

        Parameters
        ----------
        obs_seq : array_like
            Sequence of observations.

        states : array_like, optional
            List of states (of size K).

        thres : float
            Convergence threshold. Kullback-Leibler divergence value below
            which model training is stopped.

        obs : array_like, optional
            Observations space (of size N).

        tp : array_like or numpy.ndarray, optional
            Transition matrix (of size K × K) which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : array_like or numpy.ndarray, optional
            Emission matrix (of size K × N) which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space, O = {o_1, o_2, ..., o_N}.

        pi : array_like or numpy.ndarray, optional
            Initial probabilities array (of size K).

        Returns
        -------
        model : object
            Hidden Markov model trained using Baum-Welch algorithm.
        '''

        obs_seq = np.array(list(obs_seq))

        if obs is None:
            obs = np.unique(obs_seq)

        K = len(states)
        N = len(obs)

        if tp is None:
            tp = np.random.random((K, K))
            tp /= tp.sum(axis=1)[:, None]

        if ep is None:
            ep = np.random.random((K, N))
            ep /= ep.sum(axis=1)[:, None]

        if pi:
            pi = np.array(pi)
        else:
            pi = np.random.random(K)
            pi /= pi.sum()

        T = len(obs_seq)
        K = len(states)

        def s(i):
            return np.argwhere(obs == obs_seq[i]).flatten().item()

        alpha = np.zeros((T, K))
        beta = np.zeros((T, K))
        running = True

        log = {
            'tp': [], 'ep': [], 'pi': []
        }

        while running:
            alpha[0] = pi * ep[:, s(0)]
            alpha[0] /= alpha[0].sum()

            for i in range(1, T):
                alpha[i] = np.sum(alpha[i-1] * tp, axis=1) * ep[:, s(i)]
                alpha[i] /= alpha[i].sum()

            beta[T-1] = 1
            beta[T-1] /= beta[T-1].sum()

            for i in reversed(range(T-1)):
                beta[i] = np.sum(
                    beta[i+1] * tp * ep[:, s(i+1)],
                    axis=1
                )  # i + 1
                beta[i] /= beta[i].sum()

            ksi = np.zeros((T, K, K))
            gamma = np.zeros((T, K))

            for i in range(T-1):
                ksi[i] = alpha[i] * tp * beta[i+1] * ep[:, s(i+1)]
                ksi[i] /= ksi[i].sum()

                gamma[i] = alpha[i] * beta[i]
                gamma[i] /= gamma[i].sum()

            _pi = gamma[1]
            _tp = np.sum(ksi[:-1], axis=0) / gamma[:-1].sum(axis=0)
            _tp /= _tp.sum(axis=1)[:, None]
            _ep = np.zeros((K, N))

            for n, ob in enumerate(obs):
                _ep[:, n] = gamma[
                    np.argwhere(obs_seq == ob).ravel(), :
                ].sum(axis=0) / gamma.sum(axis=0)

            tp_entropy = ss.entropy(tp.ravel(), _tp.ravel())
            ep_entropy = ss.entropy(ep.ravel(), _ep.ravel())
            pi_entropy = ss.entropy(pi, _pi)

            log['tp'].append(tp_entropy)
            log['ep'].append(ep_entropy)
            log['pi'].append(pi_entropy)

            if tp_entropy < thres and\
                    ep_entropy < thres and\
                    pi_entropy < thres:
                running = False

            ep = _ep.copy()
            tp = _tp.copy()
            pi = _pi.copy()

            if not running:
                break

        model = self.__class__(
            observations=obs, states=states, tp=tp, ep=ep, pi=pi
        )
        model.obs_seq = obs_seq
        model.log = log
        return model

    def graph_make(self, *args, **kwargs):
        '''Make a directed graph of a Hidden Markov model using `graphviz`.

        Parameters
        ----------
        args : optional
            Passed to the underlying `graphviz.Digraph` method.

        kwargs : optional
            Passed to the underlying `graphviz.Digraph` method.

        Returns
        -------
        graph : graphviz.dot.Digraph
            Digraph object with its own methods.

        Note
        ----
        `graphviz.dot.Digraph.render` method should be used to output a file.
        '''

        self.graph = Digraph(*args, **kwargs)

        self.subgraph_states = Digraph(
            name="states",
            node_attr=[
                ("shape", "rect"), ("style", "filled"), ("fillcolor", "gray")
            ]
        )

        self.subgraph_obs = Digraph(
            name="obs", node_attr=[("shape", "circle")]
        )

        states_ids = range(len(self.states))
        obs_ids = range(
            len(self.states), len(self.observations) + len(self.states)
        )

        states_edges = product(states_ids, states_ids)
        obs_edges = product(states_ids, obs_ids)

        for edge in states_edges:
            v1 = edge[0]
            v2 = edge[1]
            s1 = self.states[v1]
            s2 = self.states[v2]
            p = str(np.round(self.tp[v1, v2], 2))
            self.subgraph_states.edge(s1, s2, label=p, weight=p)

        for edge in obs_edges:
            v1 = edge[0]
            v2 = edge[1]
            s1 = self.states[v1]
            s2 = self.observations[v2-len(self.states)]
            p = str(np.round(self.ep[v1, v2-len(self.states)], 2))
            self.subgraph_obs.edge(s1, s2, label=p, weight=p)

        self.graph.subgraph(self.subgraph_states)
        self.graph.subgraph(self.subgraph_obs)

        return self.graph
