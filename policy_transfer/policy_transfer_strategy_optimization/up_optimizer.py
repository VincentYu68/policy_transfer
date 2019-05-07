import cma, sys, gym, joblib, tensorflow as tf
from baselines import logger
import numpy as np
from policy_transfer.policy_transfer_strategy_optimization.bayesian_optimization import *

class UPOptimizer:
    def __init__(self, env, policy, dim, eval_num = 2, verbose=True, terminate_threshold = -np.inf,
                 max_rollout_length=None):
        self.env = env
        self.policy = policy
        self.dim = dim
        self.eval_num = eval_num
        self.sample_num = 0
        self.rollout_num = 0
        self.verbose = verbose
        self.terminate_threshold = terminate_threshold

        self.solution_history = []
        self.sample_num_history = []
        self.best_f = 100000
        self.best_x = None
        self.best_meanrollout_length = 0
        self.max_rollout_length = max_rollout_length

    def reset(self):
        self.sample_num = 0
        self.rollout_num = 0

        self.best_f = 100000
        self.best_x = None
        self.best_meanrollout_length = 0
        self.solution_history = []
        self.sample_num_history = []
        self.max_steps = 200000

    def fitness(self, x):
        app = np.copy(x)
        avg_perf = []
        rollout_len = []
        for _ in range(self.eval_num):
            o = self.env.reset()
            d = False
            rollout_rew = 0
            rollout_len.append(0)
            while not d:
                a, _ = self.policy.act(False, np.concatenate([o, app]))
                o, r, d, _ = self.env.step(a)
                rollout_rew += r
                self.sample_num += 1
                rollout_len[-1] += 1
                if self.max_rollout_length is not None:
                    if rollout_len[-1] >= self.max_rollout_length:
                        break
            self.rollout_num += 1
            avg_perf.append(rollout_rew)

        if -np.mean(avg_perf) < self.best_f:
            self.best_x = np.copy(x)
            self.best_f = -np.mean(avg_perf)
            self.best_meanrollout_length = np.mean(rollout_len)
        #print('Sampled perf: ', np.mean(avg_perf))
        return -np.mean(avg_perf)

    def cames_callback(self, es):
        self.solution_history.append(self.best_x)
        self.sample_num_history.append(self.sample_num)
        if self.verbose:
            logger.record_tabular('CurrentBest', repr(self.best_x))
            logger.record_tabular('EpRewMean', self.best_f)
            logger.record_tabular("TimestepsSoFar", self.sample_num)
            logger.record_tabular("EpisodesSoFar", self.rollout_num)
            logger.record_tabular("EpLenMean", self.best_meanrollout_length)
            logger.dump_tabular()
        return self.sample_num

    def termination_callback(self, es):
        if self.sample_num > self.max_steps: # stop if average length is over 900
            return True
        return False


    def optimize(self, maxiter = 20, max_steps = 200000):
        if self.dim > 1:
            self.max_steps = max_steps
            init_guess = [0.5] * self.dim
            #init_guess = np.random.random(self.dim)
            init_std = 0.25

            bound = [0.0, 1.0]

            print(bound)

            '''xopt, es = cma.fmin2(self.fitness, init_guess, init_std, options={'bounds': bound, 'maxiter': maxiter,
                                                                          'ftarget': self.terminate_threshold,
                                                                          'termination_callback': self.termination_callback
                                                                          },
                                 callback=self.cames_callback)'''

            xs, ys, _, _, _ = bayesian_optimisation(maxiter, self.fitness, np.array([bound]* self.dim), x0=None, n_pre_samples=10,
                                  gp_params=None, random_search=1000, alpha=1e-5, epsilon=1e-7, max_steps=max_steps,
                                  callback=self.cames_callback)
            xopt = xs[np.argmin(ys)]

            print('optimized: ', repr(xopt))
        else:
            # 1d case, not used
            candidates = np.arange(-1, 1, 0.05)
            fitnesses = [self.fitness([candidate]) for candidate in candidates]
            xopt = [candidates[np.argmin(fitnesses)]]



        return xopt