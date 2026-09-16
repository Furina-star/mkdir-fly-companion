"""
Phase 1 demo: a toy 4-neuron chain standing in for the Giant Fiber escape
circuit, driven by a brief "startle" stimulus.

The chain is:
    0 sensory  ->  1 giant fiber  ->  2 interneuron  ->  3 motor

Nothing here is real connectome data yet -- the weights are hand-typed. The
point is to confirm the simulation machinery works end to end: a stimulus
goes in one end, and a motor spike comes out the other a few steps later.
That motor spike is what will eventually make the companion sprite jump.

Run from the project root with:  python main.py
"""

import numpy as np

from fly_companion.core.lif import LIFParams, LIFNetwork

NEURON_NAMES = ["sensory", "giant fiber", "interneuron", "motor"]
MOTOR_INDEX = 3
N_STEPS = 40
STIMULUS_STEPS = (5, 6, 7)      # when the "startle" arrives
STIMULUS_STRENGTH = 0.6


def build_chain() -> np.ndarray:
    """Build the toy feed-forward chain: each neuron drives the next one."""
    n = len(NEURON_NAMES)
    weights = np.zeros((n, n))
    weights[0, 1] = 1.5   # sensory     -> giant fiber
    weights[1, 2] = 1.5   # giant fiber -> interneuron
    weights[2, 3] = 1.5   # interneuron -> motor
    return weights


def build_stimulus(n_neurons: int) -> np.ndarray:
    """A brief burst of input into the sensory neuron only, then silence."""
    stimulus = np.zeros((N_STEPS, n_neurons))
    for t in STIMULUS_STEPS:
        stimulus[t, 0] = STIMULUS_STRENGTH
    return stimulus


def print_raster(spikes: np.ndarray) -> None:
    """Print an ASCII spike raster -- one row per neuron, one column per step."""
    print("\nSpike raster ('|' = fired)")
    print("-" * 60)
    for i, name in enumerate(NEURON_NAMES):
        row = "".join("|" if fired else "." for fired in spikes[:, i])
        print("{:>12} | {}".format(name, row))
    ruler = "".join(str(t // 10) if t % 10 == 0 else " " for t in range(N_STEPS))
    print("{:>12} | {}".format("", ruler))
    print("{:>12}   {}".format("", "(each column = 1 timestep)"))


def report(spikes: np.ndarray) -> bool:
    """Summarize what happened. Returns True if the signal reached the motor."""
    print("\nFirst spike per neuron:")
    for i, name in enumerate(NEURON_NAMES):
        when = np.flatnonzero(spikes[:, i])
        timing = "step {}".format(when[0]) if when.size else "never fired"
        print("  {:>12}: {}".format(name, timing))

    motor_spikes = np.flatnonzero(spikes[:, MOTOR_INDEX])
    print()
    if motor_spikes.size:
        print("SUCCESS: the stimulus propagated all the way to the motor neuron.")
        print("         Motor fired at step(s): {}".format([int(t) for t in motor_spikes]))
        print("         This is the signal that will drive the sprite's startle.")
        return True

    print("The motor neuron never fired. Try raising STIMULUS_STRENGTH,")
    print("the connection weights, or tau -- the signal is decaying out.")
    return False


def main() -> int:
    """Run the demo. Returns a process exit code: 0 on success, 1 on failure."""
    params = LIFParams(v_threshold=1.0, v_reset=-0.2, tau=8.0, dt=1.0)
    net = LIFNetwork(build_chain(), params)

    print("Network:", net)
    print("Stimulus: {} into '{}' at steps {}".format(
        STIMULUS_STRENGTH, NEURON_NAMES[0], list(STIMULUS_STEPS)))

    spikes = net.simulate(build_stimulus(net.n_neurons))

    print_raster(spikes)
    return 0 if report(spikes) else 1