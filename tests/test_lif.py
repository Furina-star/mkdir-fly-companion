"""
Test cases for LIFParams, LIFNeuron and LIFNetwork.

Covers the three behaviors from the voltage-trace explanation (leak toward
rest, accumulate without firing, fire and reset at threshold), network-level
spike propagation, and -- importantly for the refactor -- that the vectorized
LIFNetwork produces exactly the same spikes as the scalar LIFNeuron.
"""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from fly_companion.core.lif import LIFParams, LIFNeuron, LIFNetwork

# Resolved once at import time so the sparse tests can be skipped cleanly
# rather than guarded inside every test body.
try:
    from scipy import sparse
except ImportError:
    sparse = None


class TestLIFParams(unittest.TestCase):
    def test_decay_is_dt_over_tau(self):
        self.assertAlmostEqual(LIFParams(tau=10.0, dt=2.0).v_decay, 0.2)

    def test_rejects_nonpositive_tau(self):
        with self.assertRaises(ValueError):
            LIFParams(tau=0.0)

    def test_rejects_threshold_at_or_below_rest(self):
        with self.assertRaises(ValueError):
            LIFParams(v_rest=1.0, v_threshold=1.0)

    def test_is_immutable(self):
        params = LIFParams()
        # setattr rather than `params.tau = ...` so the frozen-dataclass
        # violation is raised at runtime instead of flagged statically.
        with self.assertRaises(FrozenInstanceError):
            setattr(params, "tau", 5.0)


class TestLIFNeuron(unittest.TestCase):
    def test_leaks_toward_rest_without_input(self):
        neuron = LIFNeuron(v_rest=0.0, tau=10.0, dt=1.0)
        neuron.v = 0.8
        neuron.step()
        self.assertLess(neuron.v, 0.8)      # decayed
        self.assertGreater(neuron.v, 0.0)   # but didn't overshoot rest

    def test_accumulates_input_without_firing(self):
        neuron = LIFNeuron(v_threshold=5.0)  # high threshold, shouldn't fire
        self.assertFalse(neuron.step(0.5))
        self.assertGreater(neuron.v, 0.0)

    def test_fires_and_resets_at_threshold(self):
        neuron = LIFNeuron(v_threshold=1.0, v_reset=-0.2)
        self.assertTrue(neuron.step(2.0))
        self.assertEqual(neuron.v, -0.2)

    def test_reset_clears_state(self):
        neuron = LIFNeuron()
        neuron.step(5.0)
        neuron.reset()
        self.assertEqual(neuron.v, neuron.params.v_rest)
        self.assertEqual(neuron.spike_history, [])

    def test_rejects_params_and_overrides_together(self):
        # Constructed outside the with-block so only ONE call inside it can raise.
        params = LIFParams()
        with self.assertRaises(TypeError):
            LIFNeuron(params, tau=5.0)


class TestLIFNetwork(unittest.TestCase):
    def test_rejects_non_square_weights(self):
        weights = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            LIFNetwork(weights)

    def test_spike_propagates_to_connected_neuron(self):
        # 2 neurons: neuron 0 -> neuron 1, strong connection
        weights = np.array([
            [0.0, 2.0],
            [0.0, 0.0],
        ])
        net = LIFNetwork(weights, v_threshold=1.0, v_reset=-0.2, tau=10.0)

        spikes = net.simulate(np.array([
            [2.0, 0.0],  # step 0: kick neuron 0 only
            [0.0, 0.0],  # step 1: nothing external
        ]))

        self.assertTrue(spikes[0, 0])    # neuron 0 fires at step 0
        self.assertFalse(spikes[0, 1])   # neuron 1 hasn't received it yet
        self.assertTrue(spikes[1, 1])    # neuron 1 fires at step 1, driven by neuron 0

    def test_no_input_no_firing(self):
        net = LIFNetwork(np.zeros((3, 3)))
        self.assertFalse(net.simulate(n_steps=5).any())

    def test_simulate_requires_steps_or_input(self):
        # Built outside the with-block so only simulate() can raise inside it.
        net = LIFNetwork(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            net.simulate()

    def test_reset_returns_network_to_rest(self):
        net = LIFNetwork(np.zeros((2, 2)))
        net.simulate(np.full((3, 2), 5.0))
        net.reset()
        np.testing.assert_allclose(net.v, net.params.v_rest)
        self.assertFalse(net.fired.any())

    def test_matches_scalar_neuron_exactly(self):
        """The vectorized path must give identical results to the scalar one."""
        params = LIFParams(tau=15.0, v_threshold=1.0, v_reset=-0.2)
        drive = np.array([0.3, 0.4, 0.0, 0.9, 0.2, 0.7, 0.1, 0.5])

        neuron = LIFNeuron(params)
        expected = [neuron.step(x) for x in drive]

        # A 1-neuron network with no connections: pure single-neuron dynamics.
        net = LIFNetwork(np.zeros((1, 1)), params)
        actual = net.simulate(drive.reshape(-1, 1))[:, 0]

        np.testing.assert_array_equal(actual, expected)


@unittest.skipIf(sparse is None, "scipy not installed -- sparse support is optional")
class TestSparseWeights(unittest.TestCase):
    """Sparse support matters because real connectome matrices are ~99.9% zeros."""

    def test_sparse_matches_dense(self):
        rng = np.random.default_rng(0)
        dense = (rng.random((12, 12)) < 0.2) * rng.random((12, 12)) * 3.0
        external = rng.random((25, 12)) * 0.4

        dense_spikes = LIFNetwork(dense, tau=12.0).simulate(external)
        sparse_spikes = LIFNetwork(sparse.csr_matrix(dense), tau=12.0).simulate(external)

        np.testing.assert_array_equal(dense_spikes, sparse_spikes)


if __name__ == "__main__":
    unittest.main()