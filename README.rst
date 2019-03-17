=====
mchmm
=====

*mchmm* is a Python package implementing Markov chains and Hidden Markov models.


Installation
------------

.. code:: bash

  $ git clone https://github.com/maximtrp/mchmm.git
  $ cd mchmm
  $ pip install . --user

Features
--------

Discrete Markov chains
~~~~~~~~~~~~~~~~~~~~~~

Initialize a Markov chain using your data.

.. code:: python

  >>> import mchmm as mc
  >>> a = mc.MarkovChain().from_data('AABCABCBAAAACBCBACBABCABCBACBACBABABCBACBBCBBCBCBCBACBABABCBCBAAACABABCBBCBCBCBCBCBAABCBBCBCBCCCBABCBCBBABCBABCABCCABABCBABC')


Get an observed transition frequency matrix.

.. code:: python

  >>> a.observed_matrix
  array([[ 7., 18.,  7.],
         [19.,  5., 29.],
         [ 5., 30.,  3.]])


Get an expected transition frequency matrix.

.. code:: python

  >>> a.expected_matrix
  array([[ 8.06504065, 13.78861789, 10.14634146],
         [13.35772358, 22.83739837, 16.80487805],
         [ 9.57723577, 16.37398374, 12.04878049]])


Carry out a chi-squared test.

.. code:: python

  >>> a.chisquare(a.observed_matrix, a.expected_matrix, axis=None)
  Power_divergenceResult(statistic=47.89038802624337, pvalue=1.0367838347591701e-07)


Finally, simulate a Markov chain given your data.

.. code:: python

  >>> ids, states = a.simulate(10, start='A', seed=100)
  >>> ids
  array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
  >>> states
  array(['A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A'], dtype='<U1')
  >>> "".join(states)
  'ACBACBACBA'
