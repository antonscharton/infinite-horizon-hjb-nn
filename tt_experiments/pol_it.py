
import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle
import scipy
import time
import sys
sys.path.insert(0, '../..')

class Pol_it:
    def __init__(self, initial_valuefun, ode, polit_params):
        self.v = initial_valuefun
        self.ode = ode
        [self.nos, self.nos_test_set, self.n_sweep, self.rel_val_tol, self.rel_tol, self.max_pol_iter, self.max_iter_Phi] = polit_params
        load_me = np.load('save_me.npy')
        self.interval_half = load_me[2]
        self.samples, self.samples_test = self.build_samples(-self.interval_half, self.interval_half, quasi_MC = False)
        self.data_x = self.v.prepare_data_before_opt(self.samples)
        self.constraints_list = self.construct_constraints_list()


    def build_samples(self, samples_min, samples_max, quasi_MC = False):
        samples_dim = self.ode.A.shape[0]
        samples_mat = np.zeros(shape=(samples_dim, self.nos))
        samples_mat_test_set = np.zeros(shape=(samples_dim, self.nos_test_set))
        np.random.seed(1)
        if not quasi_MC:
            for i0 in range(self.nos):
                samples_mat[:, i0] = np.random.uniform(samples_min, samples_max, samples_dim)
                if i0 == 0:
                    print(samples_mat[:, i0])
            print('la.norm(samples_mat)', la.norm(samples_mat))
        else:
            intlognos = int(np.round(np.log2(self.nos)))
            print('samples_mat.shape', samples_mat.shape)

        for i0 in range(self.nos_test_set):
            samples_mat_test_set[:, i0] = np.random.uniform(samples_min, samples_max, samples_dim)
        return samples_mat, samples_mat_test_set

    def construct_constraints_list(self):
        # return None
        n = self.ode.A.shape[0]
        xvec = np.zeros(shape=(n, n+1))
        P_list = self.v.P_batch(xvec)
        dP_list = self.v.dP_batch(xvec)
        for i0 in range(n):
            P_list[i0][:,i0] = dP_list[i0][:,i0]
        return P_list

    
    def solve_HJB(self):
        pol_iter = 0 
        rel_diff = 1
        pol_it_counter = 0
        while(rel_diff > self.rel_tol and pol_iter < self.max_pol_iter):
            pol_iter += 1
            V_old = 1*self.v.V
            y_mat , rew_MC = self.build_rhs_batch(self.samples, self.max_iter_Phi)
            y_mat_test, rew_MC_test = self.build_rhs_batch(self.samples_test, self.max_iter_Phi)
            # y_mat , rew_MC = self.build_rhs_batch_experimental(self.samples, self.max_iter_Phi)
            # y_mat_test, rew_MC_test = self.build_rhs_batch_experimental(self.samples_test, self.max_iter_Phi)
            # t00 = time.time()
            # t01 = time.perf_counter()
            # y_mat , rew_MC = self.build_rhs_batch(self.samples, self.max_iter_Phi)
            # t10 = time.time()
            # t11 = time.perf_counter()
            # print('The calculations took:, time(), perf_counter()', t10 - t00, t11 - t01 )
            # t00 = time.time()
            # t01 = time.perf_counter()
            # y_matexp , rew_MCexp = self.build_rhs_batch_experimental(self.samples, self.max_iter_Phi)
            # t10 = time.time()
            # t11 = time.perf_counter()
            # print('The calculations took:, time(), perf_counter()', t10 - t00, t11 - t01 )
            data_y = self.v.prepare_data_while_opt(y_mat)
            data = [self.data_x, data_y, rew_MC, self.constraints_list]
            params = [self.n_sweep, self.rel_val_tol]
            print('rhs built')
            
            self.v.solve_linear_HJB(data, params)
            pickle.dump(self.v.V, open('V_{}'.format(pol_it_counter), 'wb'))
            try:
                rel_diff = xe.frob_norm(self.v.V - V_old) / xe.frob_norm(V_old)
            except:
                rel_diff = 1
            mean_error_test_set = self.calc_mean_error(self.samples_test, y_mat_test, rew_MC_test)
            print('num', pol_it_counter, "rel_diff", rel_diff, 'frob_norm(V)', xe.frob_norm(self.v.V), 'frob_norm(V_old)', xe.frob_norm(V_old), 'avg. gen error', mean_error_test_set)
            # print('mean', mean_error, 'eval_V_batch(V, xmat)[0]', eval_V_batch(V, samples_mat)[0], eval_V_batch(V, samples_mat).shape, samples_mat[:,0], y_mat[:,0])
            pol_it_counter += 1
    
    def calc_mean_error(self, xmat, ymat, rew_MC):
        error = (self.v.eval_V(0, xmat) - self.v.eval_V(0, ymat) - rew_MC)



#         x_vec = np.zeros(shape=(100, n))
#         u_vec = np.zeros(shape=(100, 1))
#         x_vec[0, :] = xmat[:,0]
#         cost = 1/2*self.ode.calc_reward(0, x_vec[0, :], u_vec[0, :])
#         for i0 in range(len(steps)-1):
#             u_vec[i0, :] = _calc_u(x_vec[i0, :], steps[i0])
#             x_vec[i0+1, :] = _step(0, x_vec[i0, :], u_vec[i0, :])
#             add_cost = _calc_cost(0, x_vec[i0, :], u_vec[i0, :])
#             cost += add_cost
#         cost -= add_cost/2
        return np.linalg.norm(error)**2 / rew_MC.size


    def calc_u(self, t, x):
        grad = self.v.calc_grad(t, x)
        return self.ode.calc_u(t, x, grad)

    
    def build_rhs_batch(self, samples, steps):
        x_mat = samples
        u_mat = self.calc_u(0, x_mat)
        rew_MC = 1/2*self.ode.calc_reward(0, x_mat, u_mat)
        #_rew_MC[i0] += self.ode.calc_reward(x, u)
        for i1 in range(self.max_iter_Phi):
            x_mat = self.ode.step(0, x_mat, u_mat)
            u_mat = self.calc_u(0, x_mat)
            reward = self.ode.calc_reward(0, x_mat, u_mat)
            rew_MC += reward
        y_mat = x_mat
        rew_MC -= reward/2
        # rew_MC += eval_V_batch(V_eval, x_mat)
        return y_mat, rew_MC

    def build_rhs_batch_experimental(self, samples, steps):
        t_points = np.linspace(0, steps*self.ode.tau, steps+1)
        sol =  self.ode.solver(t_points, samples, self.calc_u)
        solshapebefore = sol.shape
        solreshaped = sol.reshape((sol.shape[0], -1))
        umat = self.calc_u(0, solreshaped)
        rewards = self.ode.calc_reward(0, solreshaped, umat)
        reward_mat = rewards.reshape((solshapebefore[1], solshapebefore[2]))
        rew_MC = np.trapz(reward_mat, axis=1)
        larger_bool = np.logical_and(sol>=-self.interval_half, sol<= self.interval_half)
        numentries = sol.shape[0]*sol.shape[2]
        for i0 in range(samples.shape[1]):
            if np.count_nonzero(larger_bool[:, i0, :]) != numentries:
                    sol[:,i0, -1] = sol[:, i0, 0]
                    rew_MC[i0] = 0

        larger = np.count_nonzero(np.logical_or(sol<-self.interval_half, sol > self.interval_half))
        if larger > 0:
            print('num entries larger than', self.interval_half,':', larger)
        y_mat = sol[:, :, -1]
        return y_mat, rew_MC
        




