"""A variety of callback functions to be used in classes derived from
`qisresearch.i_vqe.iterative.IterativeVQE` as the `step_callbacks` argument.

The structure of this module is inspired by `skopt.callbacks`.
https://scikit-optimize.github.io/stable/modules/classes.html#module-skopt.callbacks
"""
import logging
import pickle
from abc import abstractmethod
from time import time
from typing import List, Dict, Callable

import numpy as np

logger = logging.getLogger(__name__)


# Inspired by
# https://scikit-optimize.github.io/stable/_modules/skopt/callbacks.html#VerboseCallback

class Callback:
    """Base callback class.
    """

    @abstractmethod
    def __init__(self):
        pass

    def _halt(self, step_history: List[Dict]) -> bool:
        halt = self.halt(step_history)
        if halt:
            logger.info('Halting iVQE because: {}'.format(
                self.halt_reason(step_history)
            ))
        return halt

    @abstractmethod
    def halt(self, step_history: List[Dict]) -> bool:
        """`abstractmethod` to be implemented. This method must return a
        `bool` whether or not the algorithm should halt, based on the current
        `step_history`.

        Parameters
        ----------
        step_history : List[Dict]
            Current step history for the `IterativeVQE`.

        Returns
        -------
        bool
            `True` if the algorithm should halt, `False` otherwise.

        """
        pass

    @abstractmethod
    def halt_reason(self, step_history):
        """Return a reason for halting. This will be sent to the logger.

        Parameters
        ----------
        step_history : type
            Current step history for the `IterativeVQE`.

        Returns
        -------
        str
            A justification for halting the `IterativeVQE`.

        """
        pass


class EarlyStopper(Callback):

    def __init__(self, max_time: int):
        """Stopper to stop the algorithm if it exceeds a set time.
        If the `IterativeVQE` exceeds a given time, it will stop after it finishes
        the iteration that it is on.

        Parameters
        ----------
        max_time : int
            Maximum number of seconds to run the algorithm for.
        """
        self.max_time = max_time
        self._start_time = None

    def halt(self, step_history) -> bool:
        if self._start_time is None:
            self._start_time = time()
        elapsed_time = time() - self._start_time
        logger.info('Time elapsed since first evaluation: {}'.format(
            elapsed_time
        ))
        logger.info('Maximum time left: {}'.format(
            self.max_time - elapsed_time
        ))
        if elapsed_time >= self.max_time:
            return True
        else:
            return False

    def halt_reason(self, step_history):
        return 'Maximum time exceeded'


class TimerCallback(Callback):

    def __init__(self):
        """Keep track of the time between steps in the `IterativeVQE`.

        Attributes
        ----------
        times: List[float]
            List of times between each step of the `IterativeVQE`.

        """
        self.times = []
        self._time = time()

    def halt(self, step_history) -> bool:
        current_time = time()
        time_difference = current_time - self._time
        self.times.append(time_difference)
        self._time = current_time
        logger.info('Time since last evaluation: {}'.format(
            time_difference
        ))
        return False

    def halt_reason(self, step_history) -> bool:
        return ''


class DeltaYStopper(Callback):

    def __init__(self, delta, n_best=5):
        """Stop the algorithm if `n_best` of the best results are all within a
        certain range, `delta`. This callback is used to help the algorithm
        terminate if subsequent steps do not produce much better results.

        Parameters
        ----------
        delta : float
            Range for the best results to be within for convergence.
        n_best : int
            The number of results to consider for convergence.
        """
        self.delta = delta
        self.n_best = n_best
        self._close_list = []

    def halt(self, step_history):
        energies = np.sort(np.array([step['energy'] for step in step_history]))
        if len(energies) <= self.n_best:
            return False
        best = energies[0]
        worst = energies[self.n_best - 1]
        if worst - best < self.delta:
            self._close_list = energies[0:self.n_best - 1]
            return True
        else:
            return False

    def halt_reason(self, step_history):
        return 'Best results are all within {} of each other: {}'.format(
            self.delta,
            self._close_list
        )


class DeltaXStopper(Callback):

    def __init__(self, delta):
        """Stop the algorithm if subsequent optimal parameters are close together.
        For two subsequent sets of optimal parameters in the `IterativeVQE`,
        `x_1` and `x_2`, if `|x_1 - x_2| < delta`, then the algorithm halts.
        If `x_1` and `x_2` are of different lengths (say `m` and `n`), then
        the first `min(m,n)` elements are considered.

        Parameters
        ----------
        delta : float
            Tolerance for two sets of parameters to be considered close enough.
        """
        self.delta = delta

    def halt(self, step_history) -> bool:
        if len(step_history) < 2:
            return False
        x_1 = np.array(step_history[-1]['opt_params'])
        x_2 = np.array(step_history[-2]['opt_params'])

        min_pars = min(len(x_1), len(x_2))
        x_1 = x_1[0:min_pars]
        x_2 = x_2[0:min_pars]

        if np.linalg.norm(x_1 - x_2) < self.delta:
            return True
        else:
            return False

    def halt_reason(self, step_history):
        return 'Last two steps are within {} of each other in their parameters'.format(self.delta)


class VerbosePrinter(Callback):

    def __init__(self):
        """Prints the current step, energy, best energy, and parameters at each
        step of the algorithm.
        """
        pass

    def halt(self, step_history) -> bool:
        print('Currently on Step {}'.format(len(step_history)))
        print('Current energy: {}'.format(step_history[-1]['energy']))
        best_en = np.min(np.array([step['energy'] for step in step_history]))
        print('Best energy: {}'.format(best_en))
        print('Current params: {}'.format(step_history[-1]['opt_params']))
        print('\n')
        return False

    def halt_reason(self, step_history):
        return ''


class CheckpointSaver(Callback):

    def __init__(self, checkpoint_path, **dump_options):
        """Saves the results to a file after each step of the algorithm. This
        is intended for long-running algorithms that could potentially crash
        at some point. The object saved at each step is `IterativeVQE.step_history`.

        Parameters
        ----------
        checkpoint_path : str
            Location to dump the `step_history` to.
        **dump_options : options
            Additional `kwargs` to pass to `pickle.dump` when saving the file.
        """
        self.checkpoint_path = checkpoint_path
        self.dump_options = dump_options

    def halt(self, step_history) -> bool:
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(step_history, f, **self.dump_options)
        return False

    def halt_reason(self, step_history):
        return ''


class ParameterStopper(Callback):

    def __init__(self, max_pars: int):
        """Stop the algorithm if the variational form exceeds `max_pars` number
        of parameters.

        Parameters
        ----------
        max_pars : int
            Maximum number of parameters.
        """
        self.max_pars = max_pars

    def halt(self, step_history) -> bool:
        pars = step_history[-1]['opt_params']
        n_pars = len(pars)
        if n_pars > self.max_pars:
            return True
        else:
            return False

    def halt_reason(self, step_history):
        return 'Maximum number of parameters exceeded'


class NoImprovementStopper(Callback):

    def __init__(self, n_steps: int):
        """Stop the algorithm if the best energy estimate has not improved for
        `n_steps` number of steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to go without improving the best energy estimate.
        """
        self.n_steps = n_steps

    def halt(self, step_history) -> bool:
        if len(step_history) < self.n_steps:
            return False
        energies = np.array([step['energy'] for step in step_history])
        recent_energies = energies[-self.n_steps:]
        if recent_energies[0] == np.min(recent_energies):
            return True
        else:
            return False

    def halt_reason(self, step_history):
        return 'Improvement not found in {} steps'.format(self.n_steps)


class FloorStopper(Callback):

    def __init__(self, energy_floor: float, delta: float):
        """Stop the algorithm if the energy estimate comes within `delta` of
        `energy_floor`. This is used to artificially stop the algorithm if you
        are just looking to test an problem that you can solve without the use of
        a quantum device.

        Note that if the algorithm undershoots `energy_floor` by more than `delta`,
        this callback will do nothing.

        Parameters
        ----------
        energy_floor : float
            Exact lowest energy of problem to solve.
        delta : float
            Tolerance for comparing current energy with exact lowest energy.
        """
        self.energy_floor = energy_floor
        self.delta = delta

    def halt(self, step_history) -> bool:
        current_energy = step_history[-1]['energy']
        if np.abs(self.energy_floor - current_energy) < self.delta:
            return True
        else:
            return False

    def halt_reason(self, step_history):
        return 'Current energy within delta of floor'


class OverlapStopper(Callback):

    def __init__(self, exact_vec_list: List[np.array], overlap_ceiling: float):
        self.overlap_ceiling = overlap_ceiling
        self.exact_vec_list = exact_vec_list
    
    def halt(self, step_history) -> bool:
        try:
            vec = step_history[-1]['min_vector']
        except:
            logger.info('Could not get vector at this step')
            return False
        eta = 0.0
        for ex_vec in self.exact_vec_list:
            eta += np.abs(np.vdot(vec, ex_vec))**2
        return eta >= self.overlap_ceiling
    
    def halt_reason(self, step_history):
        return 'Overlap with exact state is greater than ceiling'


class EnergyErrorPrinter(Callback):

    def __init__(self, ground_state_energy: float):
        """Given the exact ground state energy determined by some other method,
        print information about the energy relative to the exact ground state
        energy at each step, as well as improvements from previous steps.

        Useful for diagnosing and keeping track of long jobs.

        Parameters
        ----------
        ground_state_energy: float
            Exact ground state energy determined from some other method.
        """
        self._ground_state_energy = ground_state_energy
        self._last_energy = None

    def halt(self, step_history) -> bool:
        step = len(step_history)
        delta_e = step_history[-1]['energy'] - self._ground_state_energy
        out_strs = []
        out_strs.append('\nEnergy error at step {} reported: {}'.format(
            step,
            delta_e
        ))
        out_strs.append('Log_10(|Energy error|) at step {} reported: {}'.format(
            step,
            np.log10(np.abs(delta_e))
        ))
        if self._last_energy is not None:
            improvement = self._last_energy - step_history[-1]['energy']
            out_strs.append('Improvement from last (positive good): {}'.format(
                improvement
            ))
            out_strs.append('Log_10(|improvement|): {}'.format(
                np.log10(np.abs(improvement))
            ))
            self._last_energy = step_history[-1]['energy']
        else:
            self._last_energy = step_history[-1]['energy']

        for s in out_strs:
            logger.info(s)
            print(s)
        return False

    def halt_reason(self, step_history):
        return ''


class MaxGradientStopper(Callback):

    def __init__(self, max_gradient_tolerance: float):
        self.max_gradient_tolerance = max_gradient_tolerance

    def halt(self, step_history) -> bool:
        max_grad = abs(step_history[-1]['max_grad'][0])
        return max_grad < self.max_gradient_tolerance

    def halt_reason(self, step_history):
        return 'Gradient threshhold satisfied'
