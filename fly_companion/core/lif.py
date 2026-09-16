"""
Leaky Integrate-and-Fire (LIF) neuron simulation.

The core update loop (integrate -> leak -> threshold -> fire -> reset),
built so the same LIFNetwork class can later be handed a real MaleCNS
connectome matrix, without changing the class itself.

Units are intentionally dimensionless (not millivolts or milliseconds) to
keep the toy model easy to reason about. Swapping in real biophysical units
later only means changing the values in LIFParams.

Design notes:
    - LIFNeuron models ONE neuron in plain Python. It is for learning,
      debugging and unit tests -- readable over fast.
    - LIFNetwork does NOT hold a list of LIFNeuron objects. It keeps all
      neuron state in NumPy arrays and updates every neuron at once. A
      per-neuron Python loop would be the bottleneck the moment you scale
      past a few hundred neurons, and the connectome has far more than that.
    - LIFNetwork accepts SciPy sparse weight matrices as well as dense ones.
      Real connectome connectivity is extremely sparse (a neuron connects to
      a few hundred partners, not to all 176,000), so a dense matrix would
      waste almost all of its memory storing zeros.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class LIFParams:
    """Parameters shared by LIF neurons.

    Frozen so a single params object can be safely shared between a neuron
    and a network without one of them mutating it underneath the other.

    Attributes:
        v_rest: Resting (baseline) potential.
        v_threshold: Potential at which a neuron fires.
        v_reset: Potential a neuron drops to immediately after firing.
        tau: Leak time constant -- larger tau means slower decay.
        dt: Simulation timestep size.
    """
    v_rest: float = 0.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    tau: float = 20.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must be above v_rest, or neurons fire forever")

    @property
    def v_decay(self) -> float:
        """Precomputed dt/tau -- the fraction of the gap to rest lost per step.

        Computing this once here avoids repeating the division inside the
        simulation loop, which runs thousands of times.
        """
        return self.dt / self.tau


def _resolve_params(params: Optional[LIFParams], overrides: dict) -> LIFParams:
    """Build an LIFParams from either an instance or keyword overrides.

    Shared by LIFNeuron and LIFNetwork so the two constructors behave
    identically and the rule lives in exactly one place.
    """
    if params is None:
        return LIFParams(**overrides)
    if overrides:
        raise TypeError("pass either 'params' or keyword overrides, not both")
    return params


class LIFNeuron:
    """A single Leaky Integrate-and-Fire neuron.

    Scalar and readable -- use this to reason about or test the dynamics of
    one neuron. For anything with more than a handful of neurons, use
    LIFNetwork, which does the same math vectorized.
    """

    # Declared for type checkers; __slots__ still controls actual storage.
    params: LIFParams
    v: float
    fired: bool

    __slots__ = ("params", "v", "fired", "_spike_history")

    def __init__(self, params: Optional[LIFParams] = None, **overrides) -> None:
        """
        Args:
            params: An LIFParams instance. Defaults are used if omitted.
            **overrides: Individual parameter overrides, e.g. v_threshold=2.0.
                Convenient for tests. Cannot be combined with `params`.
        """
        self.params = _resolve_params(params, overrides)
        self.v = self.params.v_rest
        self.fired = False
        self._spike_history: List[bool] = []

    @property
    def spike_history(self) -> List[bool]:
        """One boolean per elapsed timestep -- whether this neuron fired then."""
        return self._spike_history

    def step(self, input_current: float = 0.0) -> bool:
        """Advance the neuron state by one timestep.

        Args:
            input_current: The input current to this neuron for this timestep.

        Returns:
            True if the neuron fired, False otherwise.
        """
        p = self.params
        # Rearranged from dt * (-(v - v_rest)/tau) into a single multiply by
        # the precomputed decay factor. Mathematically identical, one op cheaper.
        self.v += p.dt * input_current - p.v_decay * (self.v - p.v_rest)

        self.fired = self.v >= p.v_threshold
        if self.fired:
            self.v = p.v_reset

        self._spike_history.append(self.fired)
        return self.fired

    def reset(self) -> None:
        """Reset the neuron to its initial state."""
        self.v = self.params.v_rest
        self.fired = False
        self._spike_history.clear()

    def __repr__(self) -> str:
        return "LIFNeuron(v={:.3f}, fired={})".format(self.v, self.fired)


class LIFNetwork:
    """A network of LIF neurons connected by a synaptic weight matrix.

    weights[i, j] is the synaptic weight from neuron i -> neuron j. A spike
    from neuron i contributes weights[i, j] to neuron j's input on the NEXT
    timestep -- that one-step delay is what makes activity propagate through
    the network rather than resolving instantly.

    All neuron state lives in NumPy arrays and every neuron is updated in a
    single vectorized pass, so cost scales with the connectivity matrix
    rather than with Python loop overhead.
    """

    # Declared for type checkers; __slots__ still controls actual storage.
    params: LIFParams
    n_neurons: int
    v: np.ndarray
    fired: np.ndarray

    __slots__ = ("params", "weights", "n_neurons", "v", "fired", "_pending_input")

    def __init__(self, weights, params: Optional[LIFParams] = None, **overrides) -> None:
        """
        Args:
            weights: Square (n x n) connectivity matrix -- a NumPy array, or
                any SciPy sparse matrix. Sparse is strongly preferred for real
                connectome data.
            params: An LIFParams instance. Defaults are used if omitted.
            **overrides: Individual parameter overrides, e.g. tau=10.0.
        """
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError(
                "weights must be a square (n x n) matrix, got shape {}".format(weights.shape)
            )

        self.params = _resolve_params(params, overrides)
        self.n_neurons = int(weights.shape[0])

        # Dense matrices are cast to float64 once up front; sparse matrices are
        # left alone so their storage format (CSR etc.) is preserved.
        if isinstance(weights, np.ndarray):
            self.weights = np.ascontiguousarray(weights, dtype=np.float64)
        else:
            self.weights = weights

        self.v = np.full(self.n_neurons, self.params.v_rest, dtype=np.float64)
        self.fired = np.zeros(self.n_neurons, dtype=bool)
        self._pending_input = np.zeros(self.n_neurons, dtype=np.float64)

    def reset(self) -> None:
        """Return every neuron to rest and clear any in-flight synaptic input."""
        self.v.fill(self.params.v_rest)
        self.fired.fill(False)
        self._pending_input.fill(0.0)

    def step(self, external_input=None) -> np.ndarray:
        """Advance every neuron in the network by one timestep.

        Args:
            external_input: Optional array of length n_neurons -- input injected
                from outside the network this timestep (e.g. a sensory signal).
                None means no external input at all.

        Returns:
            A boolean array of which neurons fired this timestep. This is a
            VIEW of the network's internal state, not a copy -- if you intend
            to keep it across steps, copy it (or use `simulate`, which does).
        """
        p = self.params
        v = self.v

        # In-place arithmetic throughout: these operate on the existing array
        # instead of allocating a fresh one every timestep.
        v -= p.v_decay * (v - p.v_rest)   # leak
        v += self._pending_input * p.dt   # integrate input from last step
        if external_input is not None:
            v += np.asarray(external_input, dtype=np.float64) * p.dt

        np.greater_equal(v, p.v_threshold, out=self.fired)  # threshold
        # Only reset the neurons that fired, leaving the rest untouched.
        v[self.fired] = p.v_reset

        # This step's spikes become next step's synaptic input. Summing only
        # the rows of the neurons that fired beats a full vector-matrix
        # multiply, because in a realistic network only a tiny fraction of
        # neurons fire on any given timestep.
        spiking = np.flatnonzero(self.fired)
        if spiking.size:
            contribution = self.weights[spiking].sum(axis=0)
            # SciPy sparse row sums come back 2D; ravel gives a flat 1D array.
            self._pending_input[:] = np.asarray(contribution).ravel()
        else:
            self._pending_input.fill(0.0)

        return self.fired

    def simulate(self, external_input=None, n_steps: Optional[int] = None) -> np.ndarray:
        """Run the network across multiple timesteps in one call.

        Args:
            external_input: Either an array of shape (n_steps, n_neurons), or
                None to run with no external input (requires n_steps).
            n_steps: Number of steps. Inferred from external_input if omitted.

        Returns:
            Boolean array of shape (n_steps, n_neurons) -- the full spike record.
        """
        if external_input is None:
            if n_steps is None:
                raise ValueError("provide either external_input or n_steps")
            per_step = [None] * n_steps
        else:
            drive = np.asarray(external_input, dtype=np.float64)
            if drive.ndim != 2 or drive.shape[1] != self.n_neurons:
                raise ValueError(
                    "external_input must have shape (n_steps, {})".format(self.n_neurons)
                )
            n_steps = drive.shape[0]
            per_step = drive

        # Preallocated once, then filled in place -- appending to a list and
        # stacking at the end would allocate n_steps intermediate arrays.
        spikes = np.zeros((n_steps, self.n_neurons), dtype=bool)
        for t in range(n_steps):
            spikes[t] = self.step(per_step[t])
        return spikes

    def __repr__(self) -> str:
        kind = "dense" if isinstance(self.weights, np.ndarray) else "sparse"
        return "LIFNetwork(n_neurons={}, kind={})".format(self.n_neurons, kind)