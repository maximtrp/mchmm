import unittest
import tests.test_mchmm


def mchmm_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests.test_mchmm)
    return suite
