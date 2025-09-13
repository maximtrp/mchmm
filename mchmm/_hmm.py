__all__ = ["HiddenMarkovModel"]

from itertools import product
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as ss
from graphviz import Digraph
from numpy.typing import ArrayLike


class HiddenMarkovModel:
    def __init__(
        self,
        observations: Optional[ArrayLike] = None,
        states: Optional[ArrayLike] = None,
        tp: Optional[ArrayLike] = None,
        ep: Optional[ArrayLike] = None,
        pi: Optional[ArrayLike] = None,
    ):
        """Hidden Markov model.

        Parameters
        ----------
        observations : Optional[Union[list, np.ndarray]]
            Observations space (of size N).

        states : Optional[Union[list, np.ndarray]]
            List of states (of size K).

        tp : Optional[Union[list, np.ndarray]]
            Transition matrix of size K × K which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : Optional[Union[list, np.ndarray]]
            Emission matrix of size K × N which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space O = [o_1, o_2, ..., o_N].

        pi : Optional[Union[list, np.ndarray]]
            Initial state probabilities array (of size K).
        """

        self.observations = np.array(observations)
        self.states = np.array(states)
        self.tp = np.array(tp)
        self.ep = np.array(ep)
        self.pi = np.array(pi)
        self.log = None

    def _transition_matrix(
        self,
        seq: Optional[Union[str, np.ndarray, list]] = None,
        states: Optional[Union[str, np.ndarray, list]] = None,
    ):
        """Calculate a transition probability matrix which stores transition
        probability of transiting from state i to state j.

        Parameters
        ----------
        seq : Optional[Union[str, numpy.ndarray, list]]
            Sequence of states.

        states : Optional[Union[str, numpy.ndarray, list]]
            List of unique states.

        Returns
        -------
        matrix : numpy.ndarray
            Transition frequency matrix.
        """
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

        # Use safe division to handle zero row sums
        row_sums = matrix.sum(axis=1)[:, None]
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        return matrix

    def _emission_matrix(
        self,
        obs_seq: Optional[Union[str, list, np.ndarray]] = None,
        states_seq: Optional[Union[str, list, np.ndarray]] = None,
        obs: Optional[Union[str, list, np.ndarray]] = None,
        states: Optional[Union[str, list, np.ndarray]] = None,
    ) -> np.ndarray:
        """Calculate an emission probability matrix.

        Parameters
        ----------
        obs_seq : str or array_like
            Sequence of observations (of size N).
            Observation space = [o_1, o_2, ..., o_N].

        states_seq : str or array_like
            Sequence of states (of size K).
            State space = [s_1, s_2, ..., s_K].

        Returns
        -------
        ep : numpy.ndarray
            Emission probability matrix of size K × N.
        """
        _os = self.obs_seq if obs_seq is None else np.array(list(obs_seq))
        _ss = self.states_seq if states_seq is None else np.array(list(states_seq))

        obs_space = np.unique(_os) if obs is None else np.sort(np.array(list(obs)))
        states_space = (
            np.unique(_ss) if states is None else np.sort(np.array(list(states)))
        )
        k = states_space.size
        n = obs_space.size

        ef = np.zeros((k, n))

        for i in range(k):
            for j in range(n):
                o = _os[_ss == states_space[i]]
                ef[i, j] = np.count_nonzero(o == obs_space[j])

        # Use safe division to handle zero row sums
        row_sums = ef.sum(axis=1)[:, None]
        ep = np.divide(ef, row_sums, out=np.zeros_like(ef), where=row_sums != 0)
        return ep

    def from_seq(
        self,
        obs_seq: Union[str, list, np.ndarray],
        states_seq: Union[str, list, np.ndarray],
        pi: Optional[Union[str, list, np.ndarray]] = None,
        end: Optional[Union[str, list, np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> object:
        """Analyze sequences of observations and states.

        Parameters
        ----------
        obs_seq : Union[str, list, numpy.ndarray]
            Sequence of observations (of size N).
            Observation space, O = [o_1, o_2, ..., o_N].

        states_seq : Union[str, list, numpy.ndarray]
            Sequence of states (of size K).
            State space = [s_1, s_2, ..., s_K].

        pi : Optional[Union[str, list, numpy.ndarray]]
            Initial state probabilities array (of size K).
            If None, array is sampled from a uniform distribution.

        end : Optional[Union[str, list, numpy.ndarray]]
            Initial state probabilities array (of size K).
            If None, array is sampled from a uniform distribution.

        seed : Optional[int]
            Random state used to draw random variates. Passed to
            `scipy.stats.uniform` method.

        Returns
        -------
        model : HiddenMarkovModel
            Hidden Markov model learned from the given data.
        """
        # Input validation
        if len(obs_seq) != len(states_seq):
            raise ValueError(f"Observation sequence length ({len(obs_seq)}) must match states sequence length ({len(states_seq)})")

        if len(obs_seq) == 0:
            raise ValueError("Input sequences cannot be empty")

        self.obs_seq = np.array(list(obs_seq))
        self.observations = np.unique(self.obs_seq)
        self.states_seq = np.array(list(states_seq))
        self.states = np.unique(self.states_seq)
        self.tp = self._transition_matrix(self.states_seq, self.states)
        self.ep = self._emission_matrix(self.obs_seq, self.states_seq)

        if pi is None:
            self.pi = ss.uniform().rvs(size=self.states.size, random_state=seed)
            self.pi /= self.pi.sum()

        if end is None:
            self.end = ss.uniform().rvs(size=self.states.size, random_state=seed)
            self.end /= self.end.sum()

        return self

    def viterbi(
        self,
        obs_seq: Union[str, list, np.ndarray],
        obs: Optional[Union[list, np.ndarray]] = None,
        states: Optional[Union[list, np.ndarray]] = None,
        tp: Optional[np.ndarray] = None,
        ep: Optional[np.ndarray] = None,
        pi: Optional[Union[list, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Viterbi algorithm.

        Parameters
        ----------
        obs_seq : Union[str, list, np.ndarray]
            Sequence of observations.

        obs : Optional[Union[list, np.ndarray]]
            Observations space (of size N).

        states : Optional[Union[list, np.ndarray]]
            List of states (of size K).

        tp : Optional[numpy.ndarray]
            Transition matrix (of size K × K) which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : Optional[numpy.ndarray]
            Emission matrix (of size K × N) which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space, O = [o_1, o_2, ..., o_N].

        pi : Optional[Union[list, np.ndarray]]
            Initial probabilities array (of size K).

        Returns
        -------
        x : numpy.ndarray
            Sequence of states.

        z : numpy.ndarray
            Sequence of state indices.
        """
        # Input validation
        if len(obs_seq) == 0:
            raise ValueError("Observation sequence cannot be empty")

        if states is None:
            states = self.states

        if tp is None:
            tp = self.tp

        if ep is None:
            ep = self.ep

        pi = self.pi if pi is None else np.array(pi)

        obs_seq = np.array(list(obs_seq))

        if obs is None:
            obs = np.unique(obs_seq)

        T = len(obs_seq)
        K = len(states)

        def s(i):
            matches = np.argwhere(obs == obs_seq[i]).flatten()
            if len(matches) == 0:
                # Handle unknown observations by using uniform distribution
                return 0  # Default to first observation index
            return matches.item()

        t1 = np.zeros((K, T))
        t2 = np.zeros((K, T))

        t1[:, 0] = pi * ep[:, s(0)]
        t1[:, 0] /= t1[:, 0].sum()
        t2[:, 0] = 0

        for i in range(1, T):
            t1[:, i] = np.max(t1[:, i - 1] * tp * ep[:, s(i)], axis=1)
            t2[:, i] = np.argmax(t1[:, i - 1] * tp * ep[:, s(i)], axis=1)
            # Safe normalization to prevent division by zero
            t1_sum = t1[:, i].sum()
            if t1_sum > 0:
                t1[:, i] /= t1_sum

        z = np.argmax(t1, axis=0)
        x = states[z]

        for i in reversed(range(1, T)):
            z[i - 1] = t2[z[i], i]
            x[i - 1] = states[z[i - 1]]

        return x, z

    def from_baum_welch(
        self,
        obs_seq: Union[str, list, np.ndarray],
        states: Union[list, np.ndarray],
        thres: float = 0.001,
        obs: Optional[Union[str, list, np.ndarray]] = None,
        tp: Optional[np.ndarray] = None,
        ep: Optional[np.ndarray] = None,
        pi: Optional[Union[list, np.ndarray]] = None,
    ) -> object:
        """Baum-Welch algorithm.

        Parameters
        ----------
        obs_seq : Union[str, list, numpy.ndarray]
            Sequence of observations.

        states : Optional[Union[list, numpy.ndarray]]
            List of states (of size K).

        thres : Optional[float]
            Convergence threshold. Kullback-Leibler divergence value below
            which model training is stopped.

        obs : Optional[Union[list, numpy.ndarray]]
            Observations space (of size N).

        tp : Optional[numpy.ndarray]
            Transition matrix (of size K × K) which stores transition
            probability of transiting from state i (row) to state j (col).

        ep : Optional[numpy.ndarray]
            Emission matrix (of size K × N) which stores probability of
            seeing observation j (col) from state i (row). N is the length of
            observation space, O = {o_1, o_2, ..., o_N}.

        pi : Optional[Union[list, numpy.ndarray]]
            Initial probabilities array (of size K).

        Returns
        -------
        HiddenMarkovModel
            Hidden Markov model trained using Baum-Welch algorithm.
        """
        obs_seq = np.array(list(obs_seq))

        if obs is None:
            obs = np.unique(obs_seq)

        K = len(states)
        N = len(obs)
        T = len(obs_seq)

        if tp is None:
            tp = np.random.random((K, K))
            tp /= tp.sum(axis=1)[:, None]

        if ep is None:
            ep = np.random.random((K, N))
            ep /= ep.sum(axis=1)[:, None]

        pi = np.array(pi) if pi else np.random.random(K)
        pi /= pi.sum()

        def s(i):
            matches = np.argwhere(obs == obs_seq[i]).flatten()
            if len(matches) == 0:
                # Handle unknown observations by using uniform distribution
                return 0  # Default to first observation index
            return matches.item()

        alpha = np.zeros((T, K))
        beta = np.zeros((T, K))
        running = True

        log = {"tp": [], "ep": [], "pi": []}

        while running:
            alpha[0] = pi * ep[:, s(0)]
            alpha_sum = alpha[0].sum()
            if alpha_sum > 0:
                alpha[0] /= alpha_sum

            for i in range(1, T):
                alpha[i] = np.sum(alpha[i - 1] * tp, axis=1) * ep[:, s(i)]
                alpha_sum = alpha[i].sum()
                if alpha_sum > 0:
                    alpha[i] /= alpha_sum

            beta[T - 1] = 1
            beta_sum = beta[T - 1].sum()
            if beta_sum > 0:
                beta[T - 1] /= beta_sum

            for i in reversed(range(T - 1)):
                beta[i] = np.sum(beta[i + 1] * tp * ep[:, s(i + 1)], axis=1)  # i + 1
                beta_sum = beta[i].sum()
                if beta_sum > 0:
                    beta[i] /= beta_sum

            ksi = np.zeros((T, K, K))
            gamma = np.zeros((T, K))

            for i in range(T - 1):
                ksi[i] = alpha[i] * tp * beta[i + 1] * ep[:, s(i + 1)]
                ksi_sum = ksi[i].sum()
                if ksi_sum > 0:
                    ksi[i] /= ksi_sum

                gamma[i] = alpha[i] * beta[i]
                gamma_sum = gamma[i].sum()
                if gamma_sum > 0:
                    gamma[i] /= gamma_sum

            _pi = gamma[1]
            # Safe division for transition probability update
            gamma_sums = gamma[:-1].sum(axis=0)
            _tp = np.divide(np.sum(ksi[:-1], axis=0), gamma_sums,
                           out=np.zeros((K, K)), where=gamma_sums != 0)
            # Safe normalization
            tp_row_sums = _tp.sum(axis=1)[:, None]
            _tp = np.divide(_tp, tp_row_sums, out=np.zeros_like(_tp), where=tp_row_sums != 0)
            _ep = np.zeros((K, N))

            for n, ob in enumerate(obs):
                gamma_sums_total = gamma.sum(axis=0)
                _ep[:, n] = np.divide(
                    gamma[np.argwhere(obs_seq == ob).ravel(), :].sum(axis=0),
                    gamma_sums_total,
                    out=np.zeros(K),
                    where=gamma_sums_total != 0
                )

            tp_entropy = ss.entropy(tp.ravel(), _tp.ravel())
            ep_entropy = ss.entropy(ep.ravel(), _ep.ravel())
            pi_entropy = ss.entropy(pi, _pi)

            log["tp"].append(tp_entropy)
            log["ep"].append(ep_entropy)
            log["pi"].append(pi_entropy)

            if tp_entropy < thres and ep_entropy < thres and pi_entropy < thres:
                running = False

            ep = _ep.copy()
            tp = _tp.copy()
            pi = _pi.copy()

            if not running:
                break

        model = self.__class__(
            observations=list(obs), states=states, tp=tp, ep=ep, pi=pi
        )
        model.obs_seq = obs_seq
        model.log = log
        return model

    def graph_make(self, *args, **kwargs) -> Digraph:
        """Make a directed graph of a Hidden Markov model using `graphviz`.

        Parameters
        ----------
        args : optional
            Arguments passed to the underlying `graphviz.Digraph` method.

        kwargs : optional
            Keyword arguments passed to the underlying `graphviz.Digraph`
            method.

        Returns
        -------
        graph : graphviz.dot.Digraph
            Digraph object with its own methods.

        Note
        ----
        `graphviz.dot.Digraph.render` method should be used to output a file.
        """
        self.graph = Digraph(*args, **kwargs)

        self.subgraph_states = Digraph(
            name="states",
            node_attr=[("shape", "rect"), ("style", "filled"), ("fillcolor", "gray")],
        )

        self.subgraph_obs = Digraph(name="obs", node_attr=[("shape", "circle")])

        states_ids = range(len(self.states))
        obs_ids = range(len(self.states), len(self.observations) + len(self.states))

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
            s2 = self.observations[v2 - len(self.states)]
            p = str(np.round(self.ep[v1, v2 - len(self.states)], 2))
            self.subgraph_obs.edge(s1, s2, label=p, weight=p)

        self.graph.subgraph(self.subgraph_states)
        self.graph.subgraph(self.subgraph_obs)

        return self.graph
