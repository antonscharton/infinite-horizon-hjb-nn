import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle
import orth_pol

class Valuefunction_TT:
    def __init__(self, valuefunction_name=None):
        if valuefunction_name == None:
            self.V = pickle.load(open('V_new', 'rb'))
        else:
            self.V = pickle.load(open(valuefunction_name, 'rb'))
        load_me = np.load('save_me.npy')
        self.integrate_min = -load_me[2]
        self.integrate_max = load_me[2]
        self.pol_deg = np.max(self.V.dimensions)
        self.pol, self.dpol = orth_pol.calc_pol(self.integrate_max, self.integrate_min, self.pol_deg-1)


    def load_valuefun(self, valuefunction_name):
        self.V = pickle.load(open(valuefunction_name, 'rb'))
        self.pol_deg = np.max(self.V.dimensions)
        self.pol, self.dpol = orth_pol.calc_pol(self.integrate_max, self.integrate_min, self.pol_deg-1)


    def eval_V(self, t, x):
        if len(x.shape) == 1:
            ii, jj, kk = xe.indices(3)
            feat = self.P(x)
            temp = xe.Tensor([1])
            comp = xe.Tensor()
            temp[0] = 1
            r = self.V.order()
            for iter_1 in range(r):
                comp = self.V.get_component(iter_1)
                temp(kk) << temp(ii)*comp(ii, jj, kk)*xe.Tensor.from_buffer(feat[iter_1])(jj)
            return temp[0]
        else:
            feat = self.P_batch(x)
            temp = np.ones(shape=(1, x.shape[1]))
            for iter_1 in range(x.shape[0]):
                comp = self.V.get_component(iter_1).to_ndarray()
                temp = np.einsum('il,ijk,jl->kl', temp, comp, feat[iter_1])
            return temp[0]

    def calc_grad(self, t, x):
        if len(x.shape) == 1:
            r = self.V.order()
            c1, c2, c3 = xe.indices(3)
            feat = self.P(x)
            dfeat = self.dP(x)
            dV = np.zeros(shape=r)
            temp = xe.Tensor([1])
            comp = xe.Tensor()
            temp_right = xe.Tensor.ones([1])
            temp_left = xe.Tensor.ones([1])
            list_right = [None]*(r)
            list_right[r-1] = xe.Tensor(temp_right)
            for iter_0 in range(r-1, 0, -1):
                comp = self.V.get_component(iter_0)
                temp_right(c1) << temp_right(c3) * comp(c1, c2, c3) * xe.Tensor.from_buffer(feat[iter_0])(c2)
    #            temp_right = xe.contract(comp, False, temp_right, False, 1)
    #            temp_right = xe.contract(temp_right, False, xe.Tensor.from_buffer(feat[iter_0]), False, 1)
                list_right[iter_0-1] = xe.Tensor(temp_right)
            for iter_0 in range(r):
                comp = self.V.get_component(iter_0)
                temp() << temp_left(c1) * comp(c1, c2, c3) * xe.Tensor.from_buffer(dfeat[iter_0])(c2) * list_right[iter_0](c3)
    #            temp = xe.contract(comp, False, list_right[iter_0], False, 1)
    #            temp = xe.contract(temp, False, xe.Tensor.from_buffer(dfeat[iter_0]), False, 1)
    #            temp = xe.contract(temp, False, temp_left, False, 1)
                temp_left(c3) << temp_left(c1) * comp(c1, c2, c3) * xe.Tensor.from_buffer(feat[iter_0])(c2)
    #            temp_left = xe.contract(temp_left, False, comp, False, 1)
    #            temp_left = xe.contract(xe.Tensor.from_buffer(feat[iter_0]), False, temp_left, False, 1)
    
                dV[iter_0] = temp[0]
            return dV
        else:
            r = x.shape[0]
            nos = x.shape[1]
            feat = self.P_batch(x)
            dfeat = self.dP_batch(x)
            dV_mat = np.zeros(shape=x.shape)
            temp = np.zeros(1)
            temp_right = np.ones(shape = (1,nos))
            temp_left = np.ones(shape=(1,nos))
            list_right = [None]*(r)
            list_right[r-1] = temp_right
            for iter_0 in range(r-1, 0, -1):
                comp = self.V.get_component(iter_0).to_ndarray()
    #            temp_right(c1) << temp_right(c3) * comp(c1, c2, c3) * feat[iter_0](c2)
                list_right[iter_0-1] = np.einsum('kl,ijk,jl->il', list_right[iter_0], comp, feat[iter_0])
            for iter_0 in range(r):
                comp = self.V.get_component(iter_0).to_ndarray()
    #            temp() << temp_left(c1) * comp(c1, c2, c3) * dfeat[iter_0](c2) \
    #                * list_right[iter_0](c3)
                temp = np.einsum('il,ijk,jl,kl->l', temp_left, comp, dfeat[iter_0], list_right[iter_0])
    #            temp(c3) << temp_left(c1) * comp(c1, c2, c3) * feat[iter_0](c2)
                temp_left = np.einsum('il,ijk,jl->kl', temp_left, comp, feat[iter_0])
                dV_mat[iter_0,:] = temp
    #        _u = -gamma/lambd*np.dot(dV, B) - shift_TT
            return dV_mat


    def P(self, x):
        ret = []
        r = len(x)
        ret = [np.zeros(shape=(self.pol_deg)) for _ in range(r)]
        for i0 in range(r):
            for i1 in range(self.pol_deg):
                ret[i0][i1] = self.pol[i1](x[i0])
        return ret
    
    
    def dP(self, x):
        ret = []
        r = len(x)
        ret = [np.zeros(shape=(self.pol_deg)) for _ in range(r)]
        for i0 in range(r):
            for i1 in range(1, self.pol_deg):
                ret[i0][i1] = self.dpol[i1](x[i0])
        return ret


    'needs a r times nos matrix of samples and returns a list of size r with '
    'elements of size pol_deg times nos with polynomials evaluated' 
    def P_batch(self, x):
        r = x.shape[0]
        ret = []
        ret = [np.zeros(shape=(self.pol_deg, x.shape[1])) for _ in range(r)]
        for i0 in range(r):
            for i1 in range(self.pol_deg):
                ret[i0][i1,:] = self.pol[i1](x[i0,:])
        return ret
    
    'needs a r times nos matrix of samples and returns a list of size r with '
    'elements of size pol_deg times nos with polynomials evaluated' 
    def dP_batch(self, x):
        r = x.shape[0]
        ret = []
        ret = [np.zeros(shape=(self.pol_deg, x.shape[1])) for _ in range(r)]
        for i0 in range(r):
            for i1 in range(1, self.pol_deg):
                ret[i0][i1,:] = self.dpol[i1](x[i0,:])
        return ret


    def prepare_data_before_opt(self, x):
        return self.P_batch(x)


    def prepare_data_while_opt(self, x):
        return self.P_batch(x)


    def solve_linear_HJB(self, data, params):
        mat_list, mat_list_y, rew_MC, P_vec = data
        n_sweep, rel_val_tol = params

        _n_sweep = 0; rel_val = 1; val = 1e9
        omega = 1e-6
        while _n_sweep < n_sweep and rel_val > rel_val_tol:
            # V = 1e-4*xe.TTTensor.random(V.dimensions, V.ranks())
            self.V.move_core(0)
            _n_sweep += 1
            old_val, val = self.update_components_np(self.V, omega, mat_list, mat_list_y, rew_MC, _n_sweep, P_vec)
            omega = val
            rel_val = (old_val - val) / old_val
        print('val', val, 'rel_val', rel_val, '_n_sweep', _n_sweep, 'frob_norm(v)', xe.frob_norm(self.V))
        
    def update_components_np(self, G, w, mat_list, mat_list_y, rew_MC, n_sweep, P_constraints_vec):
        constraints_constant = 100
        num_constraints = P_constraints_vec[0].shape[1]
        d = G.order()
          # building Stacks for operators   
        lStack_x = [np.ones(shape=[1,rew_MC.size])]
        rStack_x = [np.ones(shape=[1,rew_MC.size])]
        lStack_y = [np.ones(shape=[1,rew_MC.size])]
        rStack_y = [np.ones(shape=[1,rew_MC.size])]
        G0_lStack = [np.ones(shape=(1, num_constraints))]
        G0_rStack = [np.ones(shape=(1, num_constraints))]
    
        for i0 in range(d-1,0,-1): 
            G_tmp = G.get_component(i0).to_ndarray()
            A_tmp_x = mat_list[i0]
            A_tmp_y = mat_list_y[i0]
            rStack_xnp = rStack_x[-1]
            rStack_ynp = rStack_y[-1]
            G_tmp_np_x = np.tensordot(G_tmp, A_tmp_x, axes=((1),(0)))
            G_tmp_np_y = np.tensordot(G_tmp, A_tmp_y, axes=((1),(0)))
            rStack_xnpres = np.einsum('jkm,km->jm',G_tmp_np_x, rStack_xnp)
            rStack_x.append(rStack_xnpres)
            rStack_ynpres = np.einsum('jkm,km->jm',G_tmp_np_y, rStack_ynp)
            rStack_y.append(rStack_ynpres)
            
            rStack_G0_tmp = G0_rStack[-1]            
            G0_tmp_np = np.tensordot(G_tmp, P_constraints_vec[i0], axes=((1),(0)))
            G0_tmp = np.einsum('jkm,km->jm',G0_tmp_np, rStack_G0_tmp)
            # G0_tmp = np.einsum('ijk,jl,kl->il',G_tmp, P_constraints_vec[i0], rStack_G0_tmp)
            G0_rStack.append(G0_tmp)
        #loop over each component from left to right
        for i0 in range(0, d):
            # first move core, then update Stacks
            if i0 > 0:
                G.move_core(i0, True)
                G_tmp = G.get_component(i0-1).to_ndarray()
                A_tmp_x = mat_list[i0-1]
                A_tmp_y = mat_list_y[i0-1]
    #            G_tmp_np = np.einsum('ijk,jl->ikl', G_tmp, A_tmp_x)
                G_tmp_np_x = np.tensordot(G_tmp, A_tmp_x, axes=((1),(0)))
                G_tmp_np_y = np.tensordot(G_tmp, A_tmp_y, axes=((1),(0)))
                lStack_xnp = lStack_x[-1]
                lStack_xnpres = np.einsum('jm,jkm->km', lStack_xnp, G_tmp_np_x)
                lStack_x.append(lStack_xnpres)
                del rStack_x[-1]
                lStack_ynp = lStack_y[-1]
                lStack_ynpres = np.einsum('jm,jkm->km', lStack_ynp, G_tmp_np_y)
                lStack_y.append(lStack_ynpres)
                del rStack_y[-1]
                G0_lStack_tmp = G0_lStack[-1]
                G0_tmp_np = np.tensordot(G_tmp, P_constraints_vec[i0-1], axes=((1),(0)))
                G0_tmp = np.einsum('jm,jkm->km', G0_lStack_tmp, G0_tmp_np)
                # G0_tmp = np.einsum('il,ijk,jl->kl',G0_lStack_tmp, G_tmp, P_constraints_vec[i0-1])
                G0_lStack.append(G0_tmp)
                del G0_rStack[-1]
    
            Ai_x = mat_list[i0]; Ai_y = mat_list_y[i0]
            lStack_xnp = lStack_x[-1]; rStack_xnp = rStack_x[-1]
            lStack_ynp = lStack_y[-1]; rStack_ynp = rStack_y[-1]
            op_pre = np.einsum('il,jl,kl->ijkl',lStack_xnp,Ai_x,rStack_xnp) - np.einsum('il,jl,kl->ijkl',lStack_ynp,Ai_y,rStack_ynp)   
    #        op = np.einsum('ijkl,mnol->ijkmno', op_pre, op_pre)
            op_G0 = np.einsum('il,jl,kl->ijkl', G0_lStack[-1], P_constraints_vec[i0], G0_rStack[-1])
            op = np.tensordot(op_pre, op_pre, axes=((3),(3)))
            op += 2*rew_MC.size*constraints_constant*np.tensordot(op_G0, op_G0, axes=((3),(3)))
    #        rhs = np.einsum('ijkl,l->ijk', op_pre, rew_MC)
            rhs = np.tensordot(op_pre, rew_MC, axes=((3),(0)))
            op_dim = op.shape
            op = op.reshape((op_dim[0]*op_dim[1]*op_dim[2], op_dim[3]*op_dim[4]*op_dim[5]))
            rhs_dim = rhs.shape
            if(n_sweep == 1 and i0 == 0):
                comp = G.get_component(i0).to_ndarray()
                Ax = np.tensordot(op_pre, comp, axes=([0,1,2],[0,1,2]))
                curr_const = np.einsum('il,jl,kl,ijk ->l', G0_lStack[-1], P_constraints_vec[i0], G0_rStack[-1], comp)
                # print(curr_const)
                # print('before', constraints_constant*np.linalg.norm(curr_const)**2)
                w = np.linalg.norm(Ax - rew_MC)**2/rew_MC.size + constraints_constant*np.linalg.norm(curr_const)**2
                # w =  constraints_constant*np.linalg.norm(curr_const)**2
                # print('first_res', w, np.linalg.norm(Ax - rew_MC)**2/rew_MC.size, constraints_constant*np.linalg.norm(curr_const)**2)
            op += 1e-3*w * np.eye(op.shape[0])
            rhs_reshape = rhs.reshape((rhs_dim[0] * rhs_dim[1] * rhs_dim[2]))
            sol_arr = np.linalg.solve(op, rhs_reshape)
            sol_arr_reshape = sol_arr.reshape((rhs_dim[0], rhs_dim[1], rhs_dim[2]))
            sol = xe.Tensor.from_buffer(sol_arr_reshape)
            G.set_component(i0, sol)
    
        # calculate residuum
    #    Ax = np.einsum('jkli,jkl->i', op_pre, sol_arr_reshape)
        # print(i0)
        # comp = G.get_component(d-1).to_ndarray()
        Ax = np.tensordot(op_pre, sol_arr_reshape, axes=([0,1,2],[0,1,2]))
        curr_const = np.einsum('il,jl,kl,ijk ->l', G0_lStack[-1], P_constraints_vec[d-1], G0_rStack[-1], sol_arr_reshape)
        # print(curr_const)
        error1 = np.linalg.norm(Ax - rew_MC)**2/rew_MC.size
        error2 = constraints_constant*np.linalg.norm(curr_const)**2
        # print('after', error1, error2)
        return w, error1 + error2

    def calc_dof(self):
        dof = 0  # calculate orders of freedom
        for i0 in range(self.V.order()):
            dof += self.V.get_component(i0).size
        return dof


    def test(self):
        self.load_valuefun('V_new')
        n = 4
        x = np.zeros(n)
        # x = np.zeros((n, 2))
        print('x', x)
        print('eval', self.eval_V(0, x))
        print('grad', self.calc_grad(0, x))

# vfun = Valuefunction_TT()
# vfun.test()
