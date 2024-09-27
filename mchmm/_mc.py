__all__ = ["MarkovChain"]
import numpy as np
import numpy.linalg as nl
import scipy.stats as ss
from itertools import product
from graphviz import Digraph
from typing import Union, Tuple, Optional


class MarkovChain:
    def __init__(
        self,
        states: Optional[Union[np.ndarray, list]] = None,
        obs: Optional[Union[np.ndarray, list]] = None,
        obs_p: Optional[Union[np.ndarray, list]] = None,
    ):
        """Discrete Markov Chain.

        Parameters
        ----------
        states : Optional[Union[numpy.ndarray, list]
            State names list.

        obs : Optional[Union[numpy.ndarray, list]
            Observed transition frequency matrix.

        obs_p : Optional[Union[numpy.ndarray, list]
            Observed transition probability matrix.
        """

        self.states = np.array(states)
        self.observed_matrix = np.array(obs)
        self.observed_p_matrix = np.array(obs_p)

    def _transition_matrix(
        self,
        seq: Optional[Union[str, list, np.ndarray]] = None,
        states: Optional[Union[str, list, np.ndarray]] = None,
    ) -> np.ndarray:
        """Calculate a transition frequency matrix.

        Parameters
        ----------
        seq : Optional[Union[str, list, numpy.ndarray]]
            Observations sequence.

        states : Optional[Union[str, list, numpy.ndarray]]
            List of states.

        Returns
        -------
        matrix : numpy.ndarray
            Transition frequency matrix.
        """

        seql = self.seq if seq is None else np.array(list(seq))
        states = self.states if states is None else np.array(list(states))
        matrix = np.zeros((len(states), len(states)))

        for x, y in product(range(len(states)), repeat=2):
            xid = np.argwhere(seql == states[x]).flatten()
            yid = xid + 1
            yid = yid[yid < len(seql)]
            s = np.count_nonzero(seql[yid] == states[y])
            matrix[x, y] = s

        return matrix

    def n_order_matrix(
        self, mat: Optional[np.ndarray] = None, order: int = 2
    ) -> np.ndarray:
        """Create Nth order transition probability matrix.

        Parameters
        ----------
        mat : numpy.ndarray, optional
            Observed transition probability matrix.

        order : int, optional
            Order of transition probability matrix to return.
            Default is 2.

        Returns
        -------
        x : numpy.ndarray
            Nth order transition probability matrix.
        """

        return nl.matrix_power(self.observed_p_matrix if mat is None else mat, order)

    def prob_to_freq_matrix(
        self, mat: Optional[np.ndarray] = None, row_totals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate a transition frequency matrix given a transition
        probability matrix and row totals. This method is meant to be
        used to calculate a frequency matrix for a Nth order
        transition probability matrix.

        Parameters
        ----------
        mat : numpy.ndarray, optional
            Transition probability matrix.

        row_totals : numpy.ndarray, optional
            Row totals of transition frequency matrix.

        Returns
        -------
        x : numpy.ndarray
            Transition frequency matrix.
        """

        _mat = self.observed_p_matrix if mat is None else mat
        _rt = self._obs_row_totals if row_totals is None else row_totals
        _rt = _rt[:, None] if _rt.ndim == 1 else _rt
        return _mat * _rt

    def from_data(self, seq: Union[str, np.ndarray, list]) -> object:
        """Infer a Markov chain from data. States, frequency and probability
        matrices are automatically calculated and assigned to as class
        attributes.

        Parameters
        ----------
        seq : Union[str, np.ndarray, list]
            Sequence of events. A string or an array-like object exposing the
            array interface and containing strings or ints.

        Returns
        -------
        MarkovChain : object
            Trained MarkovChain class instance.
        """
        # states list
        self.seq = np.array(list(seq))
        self.states = np.unique(list(seq))

        # observed transition frequency matrix
        self.observed_matrix = self._transition_matrix(seq)
        self._obs_row_totals = np.sum(self.observed_matrix, axis=1)

        # observed transition probability matrix
        self.observed_p_matrix = np.nan_to_num(
            self.observed_matrix / self._obs_row_totals[:, None]
        )

        # filling in a row containing zeros with uniform p values
        uniform_p = 1 / len(self.states)
        zero_row = np.argwhere(self.observed_p_matrix.sum(1) == 0).ravel()
        self.observed_p_matrix[zero_row, :] = uniform_p

        # expected transition frequency matrix
        self.expected_matrix = ss.contingency.expected_freq(self.observed_matrix)

        return self

    def chisquare(
        self,
        obs: Optional[np.ndarray] = None,
        exp: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Wrapper function for carrying out a chi-squared test using
        `scipy.stats.chisquare` method.

        Parameters
        ----------
        obs : numpy.ndarray
            Observed transition frequency matrix.

        exp : numpy.ndarray
            Expected transition frequency matrix.

        kwargs : optional
            Keyword arguments passed to `scipy.stats.chisquare` method.

        Returns
        -------
        chisq : float or numpy.ndarray
            Chi-squared test statistic.

        p : float or numpy.ndarray
            P value of the test.
        """

        _obs = self.observed_matrix if obs is None else obs
        _exp = self.expected_matrix if exp is None else exp
        return ss.chisquare(f_obs=_obs, f_exp=_exp, **kwargs)

    def graph_make(self, *args, **kwargs) -> Digraph:
        """Make a directed graph of a Markov chain using `graphviz`.

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

        ids = range(len(self.states))
        edges = product(ids, ids)

        for edge in edges:
            v1 = edge[0]
            v2 = edge[1]
            s1 = self.states[v1]
            s2 = self.states[v2]
            p = str(np.round(self.observed_p_matrix[v1, v2], 2))
            self.graph.edge(s1, s2, label=p, weight=p)

        return self.graph

    def simulate(
        self,
        n: int,
        tf: Optional[np.ndarray] = None,
        states: Optional[Union[np.ndarray, list]] = None,
        start: Optional[Union[str, int]] = None,
        ret: str = "both",
        seed: Optional[Union[list, np.ndarray]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Markov chain simulation based on `scipy.stats.multinomial`.

        Parameters
        ----------
        n : int
            Number of states to simulate.

        tf : numpy.ndarray, optional
            Transition frequency matrix.
            If None, `observed_matrix` instance attribute is used.

        states : Optional[Union[np.ndarray, list]]
            State names. If None, `states` instance attribute is used.

        start : Optional[str, int]
            Event to begin with.
            If integer is passed, the state is chosen by index.
            If string is passed, the state is chosen by name.
            If `random` string is passed, a random state is taken.
            If left unspecified (None), an event with maximum
            probability is chosen.

        ret : str, optional
            Return state indices if `indices` is passed.
            If `states` is passed, return state names.
            Return both if `both` is passed.

        seed : Optional[Union[list, numpy.ndarray]]
            Random states used to draw random variates (of size `n`).
            Passed to `scipy.stats.multinomial` method.

        Returns
        -------
        x : numpy.ndarray
            Sequence of state indices.

        y : numpy.ndarray, optional
            Sequence of state names.
            Returned if `return` arg is set to 'states' or 'both'.
        """
        # matrices init
        if tf is None:
            tf = self.observed_matrix
            fp = self.observed_p_matrix
        else:
            fp = tf / tf.sum(axis=1)[:, None]

        # states init
        if states is None:
            states = self.states
        if not isinstance(states, np.ndarray):
            states = np.array(states)

        # choose a state to begin with
        # `_start` is always an index of state
        if start is None:
            row_totals = tf.sum(axis=1)
            _start = np.argmax(row_totals / tf.sum())
        elif isinstance(start, int):
            _start = start if start < len(states) else len(states) - 1
        elif start == "random":
            _start = np.random.randint(0, len(states))
        elif isinstance(start, str):
            _start = np.argwhere(states == start).item()

        # simulated sequence init
        seq = np.zeros(n, dtype=int)
        seq[0] = _start

        # random seeds
        r_states = np.random.randint(0, n, n) if seed is None else seed

        # simulation procedure
        for i in range(1, n):
            _ps = fp[seq[i - 1]]
            _sample = np.argmax(ss.multinomial.rvs(1, _ps, 1, random_state=r_states[i]))
            seq[i] = _sample

        if ret == "indices":
            return seq
        elif ret == "states":
            return states[seq]
        else:
            return seq, states[seq]
