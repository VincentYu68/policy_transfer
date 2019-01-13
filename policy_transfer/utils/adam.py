from mpi4py import MPI
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np

class Adam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None,
                 multithread=False, wd = 1e-4):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.multithread = multithread
        self.wd = wd

    def update(self, localg, stepsize):
        if self.t % 100 == 0 and self.multithread:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        if self.multithread:
            self.comm.Allreduce(localg, globalg, op=MPI.SUM)
            if self.scale_grad_by_procs:
                globalg /= self.comm.Get_size()
        else:
            globalg = localg

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon) - stepsize * self.wd * self.getflat()
        self.setfromflat(self.getflat() + step)

    def sync(self, force_sync = False):
        if self.multithread or force_sync:
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
            self.setfromflat(theta)

    def check_synced(self):
        if self.multithread:
            if self.comm.Get_rank() == 0: # this is root
                theta = self.getflat()
                self.comm.Bcast(theta, root=0)
            else:
                thetalocal = self.getflat()
                thetaroot = np.empty_like(thetalocal)
                self.comm.Bcast(thetaroot, root=0)
                assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)
