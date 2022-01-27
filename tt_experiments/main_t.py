#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:58:33 2020

@author: sallandt

"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '../')

run_set_dynamics = True
if run_set_dynamics == True:
    import set_dynamics
    
# load_num = 82
# load_num = 'new'
# load_num = 'notoptimized'
load_num = 'optimized'

print('loaded V_'+ load_num)
vfun = valuefunction_TT.Valuefunction_TT('V_{}'.format(load_num))
testOde = ode.Ode()
# testOde.test()
T = 5
n = vfun.V.order(); a = -1; b = 1
x = np.linspace(a, b, n)
max_num_initial_values = 4
min_num_initial_values = 0

max_iter_gradient_descent = 000
grad_tol_gradient_descent = 1e-12
# stepsize_gradient_descent = 1
# stepsize_before_gradient_descent = 0.1
# stepsize_gradient_descent = .01; stepsize_before_gradient_descent = 0.001 # initial value delta*( x - 1)**2*(x+1)**2
stepsize_gradient_descent = .001; stepsize_before_gradient_descent = 0.0001 # initial value 1.2 iter_grad: 100000 rest: [2.44297776e-11] / 2.829593175497833



initial_values = np.zeros(shape=(n, max_num_initial_values-min_num_initial_values))
initial_values = np.load('polysamples_x.npy')
# initial_values /= 2

plt.figure()
plt.plot(initial_values)

load_me = np.load('save_me.npy')
lambd = load_me[0]
interval_half = load_me[2]
tau = load_me[3]

steps = np.linspace(0, T, int(T/tau)+1)
m = len(steps)
control_dim = 1

def calc_u_V(x, t):
    return testOde.calc_u(0, x, vfun.calc_grad(0, x))



Pi_cont = np.load('Pi_cont.npy')
def calc_u_riccati(x, t):
    return testOde.calc_u(0, x, 2*Pi_cont@x)


def test_value_function(_step, _calc_u, _x0, _calc_cost):
    x_vec = np.zeros(shape=(len(steps), n))
    u_vec = np.zeros(shape=(len(steps), control_dim))
    x_vec[0, :] = _x0
    cost = 1/2*_calc_cost(0, x_vec[0, :], u_vec[0, :])
    u_vec[0, :] = _calc_u(x_vec[0, :], steps[0])
    for i0 in range(len(steps)-1):
        x_vec[i0+1, :] = _step(0, x_vec[i0, :], u_vec[i0, :])
        u_vec[i0, :] = _calc_u(x_vec[i0, :], steps[i0])
        add_cost = _calc_cost(0, x_vec[i0, :], u_vec[i0, :])
        cost += add_cost
    cost -= add_cost/2
    return x_vec, u_vec, cost

def test_value_function_batch(_step, _calc_u, samples_mat, _calc_cost):
    x_vec = samples_mat.T
    u_vec = _calc_u(x_vec, steps[0])
    cost = 1/2*_calc_cost(0, x_vec, u_vec)
    for i0 in range(len(steps)-1):
        x_vec = _step(0, x_vec, u_vec)
        u_vec = _calc_u(x_vec, steps[i0])
        add_cost = _calc_cost(0, x_vec, u_vec)
        cost += add_cost
    cost -= add_cost/2
    return cost


cost_hjb = test_value_function_batch(testOde.step, calc_u_V, initial_values, testOde.calc_reward)
np.save('cost_TT.npy', cost_hjb)
print('cost_hjb', cost_hjb)
# cost_lqr = np.load('polysamples_cost_ricc.npy')
# cost_opt = np.load('polysamples_cost_opt.npy')
