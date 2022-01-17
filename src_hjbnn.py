import numpy as np
import tensorflow as tf
import scipy.optimize as optimize
import scipy.linalg as linalg
import scipy.integrate as integrate
import time

def setup_from_files(A, B, Q, R, rhs, rhs_np, dtype):

    # load matrices
    A = np.load(A)
    B = np.load(B)
    Q = np.load(Q)
    R = np.load(R)

    # create dynamics
    env = Empty(dtype=dtype)
    xi = np.linspace(-1, 1, 32)
    env.x0 = np.array([2*(xi-1)**2 *(xi)**2], dtype=dtype)
    env.setup(tf.convert_to_tensor(A, dtype=dtype),
              tf.convert_to_tensor(B, dtype=dtype),
              rhs)
    env.RHS_np = rhs_np
    case = InfiniteHorizonQR(env, 0., Q=tf.convert_to_tensor(Q, dtype=dtype),
                                     R=tf.convert_to_tensor(R, dtype=dtype))

    # calculate Riccati feedback
    _ , _, = case.solve_riccati()
    return env, case


# basic dynamics class
class Dynamics:
    def __init__(self, dtype):
        self.dtype = dtype

    def f(self, x, u):
        return tf.matmul(x, self.A, transpose_b=True) + tf.matmul(u, self.B, transpose_b=True) + self.RHS(x)

    def f_np(self, x, u):
        return x.dot(self.A.numpy().T) + u.dot(self.B.numpy().T) + self.RHS_np(x)

    def b(self, x):
        return tf.convert_to_tensor(np.tile(self.B.numpy(), [len(x),1,1]), dtype=self.dtype)

    def set_feedback(self, feedback):
        self.feedback = feedback
        print("feedback set: {}".format(feedback))

    def set_feedback_linear(self, K):
        if not tf.is_tensor(K):
            K = tf.convert_to_tensor(K, dtype=self.dtype)
        self.feedback = lambda x: -tf.matmul(x, K, transpose_b=True)
        print("feedback set: {}".format(K.numpy()))

    def set_feedback_zero(self):
        K = tf.zeros([self.dim_u, self.dim_x], dtype=self.dtype)
        self.set_feedback_linear(K)

    def integrate_step_feedback(self, x, dt):
        k1 = dt * self.f(x, self.feedback(x))
        k2 = dt * self.f(x + k1/2, self.feedback(x+k1/2))
        k3 = dt * self.f(x + k2/2, self.feedback(x + k2/2))
        k4 = dt * self.f(x + k3, self.feedback(x + k3))
        return x + 1/6*(k1 + 2*k2 + 2*k3 + k4)

    def integrate_step_openloop(self, u1, u2, x, dt):
        k1 = self.f_np(x, u1)
        x2 = x + 0.5*dt*k1
        k2 = self.f_np(x2, (u1+u2)/2.)
        x3 = x + 0.5*dt*k2
        k3 = self.f_np(x3, (u1+u2)/2.)
        x4 = x + dt*k3
        k4 = self.f_np(x4, u2)
        return x + dt*(k1 + 2.*k2 + 2.*k3 + k4)/6.

    def integrate_closedloop(self, x0, dt, k):
        with tf.device('/device:CPU:0'):
            print("integrating system dynamics with assigned feedback control, done on ...", end="\r")
            ts = time.time()
            t = np.linspace(0., dt * k, k + 1, dtype=self.dtype)
            X = x0 * np.ones([k + 1, self.dim_x], dtype=self.dtype)
            x0 = tf.convert_to_tensor(x0, dtype=self.dtype)
            U = self.feedback(x0).numpy() * np.ones([k + 1, self.dim_u], dtype=self.dtype)
            for i in range(k):
                    x0 = self.integrate_step_feedback(x0, dt)
                    X[i + 1] = x0.numpy()
                    U[i + 1] = self.feedback(x0).numpy()
            print("integrating system dynamics with assigned feedback control, done in {} s".format(time.time() - ts))
            return (t,X,U)

    def integrate_closedloop_np(self, x0, dt, k):
        with tf.device('/device:CPU:0'):
            print("integrating system dynamics with assigned feedback control, done on ...", end="\r")
            ts = time.time()
            t = np.linspace(0., dt * k, k + 1, dtype=self.dtype)
            X = x0 * np.ones([k + 1, self.dim_x], dtype=self.dtype)
            U = self.feedback(x0) * np.ones([k + 1, self.dim_u], dtype=self.dtype)
            for i in range(k):
                    x0 = self.integrate_step_feedback(x0, dt)
                    X[i + 1] = x0
                    U[i + 1] = self.feedback(x0)
            print("integrating system dynamics with assigned feedback control, done in {} s".format(time.time() - ts))
            return (t,X,U)

    def integrate_openloop(self, x0, u, dt):
        k = len(u) - 1
        t = np.linspace(0., dt * k, k + 1)
        X = x0 * np.ones([k + 1, x0.size])
        for i in range(k):
            X[i + 1] = self.integrate_step_openloop(u[i], u[i+1], X[i], dt)
        return (t,X,u)


# Empty dynamics from matrices
class Empty(Dynamics):
    def __init__(self, dtype):
        super().__init__(dtype)

    def setup(self, A, B, RHS):
        self.A = A
        self.B = B
        self.RHS = RHS
        self.dim_x = A.shape[1]
        self.dim_u = B.shape[1]

    def view(self, trj, title='trajectory', k=1, legend=True):
        t = trj[0]
        X = trj[1]
        U = trj[2]
        figure = plt.figure(k)
        for i in range(self.dim_x):
            plt.plot(t, X[:, i], label='x_{}(t)'.format(i))
        for i in range(self.dim_u):
            plt.plot(t, U[:, i], linestyle=':', label='u_{}(t)'.format(i))
        if legend == True:
            plt.legend()
        plt.title(title)
        plt.xlabel("t")
        return figure


# control problem class
class InfiniteHorizon:
    def __init__(self, env, rho):
        self.env = env
        self.dtype = env.dtype
        self.rho = tf.constant(rho, dtype=self.dtype)

    def pde(self, x, u, v, dv):
        return -self.rho*v + tf.reshape(tf.reduce_sum(dv*self.env.f(x, u), axis=1), (-1,1)) + self.L(x, u)

    def transport_states(self, x0, dt, k, compile=True, summary=False):
        print("transporting {} samples with assigned feedback control, done in ...".format(len(x0)), end="\r")
        tic = time.time()

        x0 = tf.convert_to_tensor(x0, dtype=self.dtype)

        @tf.function
        def step_compiled(x0, cost0):
            x0 = self.env.integrate_step_feedback(x0, dt)
            cost0 = cost0 + 2. * self.L(x0, self.env.feedback(x0))
            return x0, cost0

        def step_uncompiled(x0, cost0):
            x0 = self.env.integrate_step_feedback(x0, dt)
            cost0 = cost0 + 2. * self.L(x0, self.env.feedback(x0))
            return x0, cost0

        if compile:
            step = step_compiled
        else:
            step = step_uncompiled

        s = self.L(x0, self.env.feedback(x0))
        for i in range(k):
            x0, s = step(x0, s)
            if summary and i%summary == 0:
                print('iteration: {}/{}'.format(i, k))


        s = s - self.L(x0, self.env.feedback(x0))
        print("transporting {} samples with assigned feedback control, done in {} s".format(len(x0), time.time() - tic))
        return x0.numpy(), (s * dt / 2.).numpy()

    def transport_states_np(self, x0, dt, k, summary=False):
        print("transporting {} samples with assigned feedback control, done in ...".format(len(x0)), end="\r")
        tic = time.time()

        def step(x0, cost0):
            x0 = self.env.integrate_step_feedback(x0, dt)
            cost0 = cost0 + 2. * self.L_np(x0, self.env.feedback(x0))
            return x0, cost0

        s = self.L_np(x0, self.env.feedback(x0))
        for i in range(k):
            x0, s = step(x0, s)
            if summary and i%summary == 0:
                print('iteration: {}/{}'.format(i, k))


        s = s - self.L_np(x0, self.env.feedback(x0))
        print("transporting {} samples with assigned feedback control, done in {} s".format(len(x0), time.time() - tic))
        return x0, s * dt / 2.

    def value_trajectory(self, trj):
        X = tf.convert_to_tensor(trj[1], dtype=self.dtype)
        U = tf.convert_to_tensor(trj[2], dtype=self.dtype)
        running_cost = self.L(X, U).numpy().reshape(-1)
        cost = integrate.simps(running_cost, trj[0])
        print('Kosten: ', cost)
        return running_cost, cost

    def feedback_from_value(self, value_model):
        Rinv = tf.linalg.inv(self.R)

        def output_and_derivative(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = value_model(x, training=False)
            dy = tape.gradient(y, x)
            return y, dy

        def feedback(x):
            b = self.env.b(x)
            _, dv = output_and_derivative(x)
            return -0.5*tf.matmul(tf.einsum('ijk,ij->ik', b, dv), Rinv, transpose_b=True)

        return feedback


class InfiniteHorizonQR(InfiniteHorizon):
    def __init__(self, env, rho, **kwargs):
        super().__init__(env, rho)

        if 'Q' in kwargs:
            self.Q = kwargs['Q']
        else:
            self.Q = tf.eye(env.dim_x, dtype=env.dtype)
        if 'R' in kwargs:
            self.R = kwargs['R']
        else:
            self.R = tf.eye(env.dim_u, dtype=env.dtype)

    def L(self, x, u):
        return tf.reshape(tf.reduce_sum(x*tf.matmul(x, self.Q, transpose_b=True), axis=1) + tf.reduce_sum(u*tf.matmul(u, self.R, transpose_b=True), axis=1), (-1,1))

    def L_np(self, x, u):
        return (np.sum(x*x.dot(self.Q.numpy().T), axis=1) + np.sum(u*u.dot(self.R.numpy().T), axis=1)).reshape((-1,1))

    def dLdx(self, x, u):
        return 2.*np.matmul(x, self.Q.numpy())

    def solve_riccati(self):
        self.V = linalg.solve_continuous_are(self.env.A.numpy(), self.env.B.numpy(), self.Q.numpy(), self.R.numpy())
        self.K = np.linalg.inv(self.R.numpy()).dot(self.env.B.numpy().T).dot(self.V)
        return self.V, self.K

    def val_and_valgrad_from_riccati(self, x):
        dv = 2.*x.dot(self.V)
        v = np.sum(dv*x/2., axis=1).reshape(-1,1)
        return v, dv

    def dv(self, x):
        return 2.*x.dot(self.V)

    def v(self, x):
        dv = self.dv(x)
        return np.sum(dv*x/2., axis=1).reshape(-1,1)


# create monte carlo samples
def random_states(limits, n):
    dim = len(limits)
    x = np.random.rand(n,dim)
    return limits[:, 0] + x * (limits[:, 1] - limits[:, 0])


# bell curve activation function
def Bellcurve(alpha):
    def func(x):
        return tf.keras.backend.exp(-(alpha*x)**2)
    return func


# multilayer perceptron with linear output layer
def MLP(width, depth, normalize, initializer, limits, activation, dtype):

    input_dim = len(limits)
    input = tf.keras.Input(shape=(input_dim,))

    # normalizer
    if normalize:
        hidden = Normalizer(limits, [-normalize, normalize], dtype)(input)
    else:
        hidden = input

    # hidden layer
    for _ in range(depth):
        hidden = tf.keras.layers.Dense(width, kernel_initializer=initializer)(hidden)
        hidden = tf.keras.layers.Lambda(activation)(hidden)
    output = tf.keras.layers.Dense(1, kernel_initializer=initializer, use_bias=False)(hidden)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


# normalization layer
class Normalizer(tf.keras.layers.Layer):
    def __init__(self, limits, bounds, dtype, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.limits = limits
        self.y1 = tf.constant(bounds[0], dtype=dtype)
        self.y2 = tf.constant(bounds[1], dtype=dtype)
        self.a = tf.constant(np.min(np.array(limits), axis=1), dtype=dtype)
        self.b = tf.constant(np.max(np.array(limits), axis=1), dtype=dtype)

    def call(self, inputs, trainable=None, mask=None):
        return (inputs - self.a) * (self.y2 - self.y1) / (self.b - self.a) + self.y1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'limits': self.limits,
            'bounds': np.array([self.y1, self.y2]),
            'dtype': self.dtype
        })
        return config


# one cycle learning rate
class OneCycle:
    def __init__(self, lr_range, momentum_range=(0.95,0.85), frac_end=0.1, frac_lr_min = 0.1):
        self.lr_range = lr_range
        self.momentum_range = momentum_range
        self.frac_end = frac_end
        self.frac_lr_min = frac_lr_min
        self.lr = []
        self.mom = []

    def __call__(self, x):
        return self.get_lr_and_momentum(x)

    def setup(self, max_iter):
        self.max_iter = max_iter
        self.abc = (0, (1-self.frac_end)*max_iter/2, (1-self.frac_end)*max_iter)
        self.m_lr = (self.lr_range[1] - self.lr_range[0])/self.abc[1]
        self.m_lr2 = self.lr_range[0]*(self.frac_lr_min-1)/(self.frac_end*max_iter)
        self.m_mom = (self.momentum_range[1]-self.momentum_range[0])/self.abc[1]

    def get_lr_and_momentum(self, iter):
        if iter < self.abc[1]:  #[a,b)
            lr = self.m_lr*iter + self.lr_range[0]
            mom = self.m_mom*iter + self.momentum_range[0]
        elif iter < self.abc[2]:    #[b,c)
            lr = -(iter-self.abc[1])*self.m_lr + self.lr_range[1]
            mom = -(iter-self.abc[1])*self.m_mom + self.momentum_range[1]
        else:   #[c,...)
            lr = (iter - self.abc[2])*self.m_lr2 + self.lr_range[0]
            mom = self.momentum_range[0]
        self.lr.append(lr)
        self.mom.append(mom)
        return (lr, mom)


# trainer class
class Trainer:
    def __init__(self, case):
        self.dtype = case.env.dtype
        self.case = case
        self.model = None
        self.reset()

    def reset(self):
        self.losshistory_train = np.array([]).reshape(-1,3)
        self.losshistory_val = np.array([]).reshape(-1,3)
        self.step = 0
        self.last_traintime = 0

    def compile(self, model, weight_losses, optimizer=None, normalize=True, track_weights=False, track_learningrates=False, track_grads=False, track_vals=False):

        self.model = model

        if track_weights:
            self.tracker_weights = []
            self.tracker_counter = 0
        else:
            self.tracker_weights = None

        if track_learningrates:
            self.tracker_learningrates = []
        else:
            self.tracker_learningrates = None

        if track_grads:
            self.tracker_grads = []
            self.tracker_grads_counter = 0
        else:
            self.tracker_grads = None

        if track_vals:
            self.tracker_vals = []
            self.tracker_vals_counter = 0
            self.activation_for_forwardpass_test = track_vals
        else:
            self.tracker_vals = None
        # determine errors at the start for normalization
        # for this data must have been loaded already
        if normalize:
            y, dy = self.output_and_derivative(self.model, self.x_pde, training=False)
            u = self.case.env.feedback(self.x_pde)
            y_bc_pred = self.model(self.x_bc, training=False)
            y_pde_pred = self.case.pde(self.x_pde, u, y, dy)
            y_trans_pred = self.model(self.x_trans_0, training=False) - self.model(self.x_trans_1, training=False)
            start_losses = self._losses(self.y_pde, y_pde_pred, self.y_bc, y_bc_pred, self.y_trans, y_trans_pred).numpy()
            self.weight_losses_normalize = start_losses
        else:
            self.weight_losses_normalize = np.array([1.,1.,1.])

        for i in range(len(self.weight_losses_normalize)):
            if self.weight_losses_normalize[i] == 0.:
                self.weight_losses_normalize[i] = 1.

        self.weight_losses_user = weight_losses
        self.weight_losses = self.weight_losses_user/self.weight_losses_normalize

        def get_mse(f):
            def loss(target, prediction):
                return tf.constant(f, dtype=self.dtype)*tf.reduce_mean(tf.square(target - prediction))
            return loss

        def get_rse(f):
            def loss(target, prediction):
                return tf.constant(f, dtype=self.dtype)*tf.sqrt(tf.reduce_mean(tf.square(target - prediction)))
            return loss

        def loss_fn_mse(target, prediction):
            return tf.reduce_mean(tf.square(target - prediction))

        def loss_fn_zero(target, prediction):
            return tf.constant(0., dtype=self.dtype)

        if weight_losses[0] != 0.:
            self.loss_fn_pinn = get_mse(self.weight_losses[0])
        else:
            self.loss_fn_pinn = loss_fn_zero

        if weight_losses[1] != 0.:
            self.loss_fn_transported = get_mse(self.weight_losses[1])
        else:
            self.loss_fn_transported = loss_fn_zero

        if weight_losses[2] != 0.:
            self.loss_fn_boundary = get_mse(self.weight_losses[2])
        else:
            self.loss_fn_boundary = loss_fn_zero

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam()
        self._train_step = tf.function(self._train_step_precompiled)

    def datatransporter(self, X, transport_dt, transport_k, type='training'):
        # create one single boundary sample
        X_bc = np.zeros_like(X[0:1])
        Y_bc = np.array([[0.]])
        Y_pde = np.zeros([len(X), 1])
        X_transported, Y_cost= self.case.transport_states(X, dt=transport_dt, k=transport_k)
        self.dataloader(X, Y_pde, X_bc, Y_bc, X, X_transported, Y_cost, type=type)

    def dataloader(self, x_pde, y_pde, x_bc, y_bc, x_trans_0, x_trans_1, y_trans, type='training'):
        #loading training or validation data
        if type == 'training':
            self.n_samples = [len(x_pde), len(x_trans_0), len(x_bc)]
            self.x_pde = tf.convert_to_tensor(x_pde, dtype=self.dtype)
            self.y_pde = tf.convert_to_tensor(y_pde, dtype=self.dtype)
            self.x_bc = tf.convert_to_tensor(x_bc, dtype=self.dtype)
            self.y_bc = tf.convert_to_tensor(y_bc, dtype=self.dtype)
            self.x_trans_0 = tf.convert_to_tensor(x_trans_0, dtype=self.dtype)
            self.x_trans_1 = tf.convert_to_tensor(x_trans_1, dtype=self.dtype)
            self.y_trans = tf.convert_to_tensor(y_trans, dtype=self.dtype)
        elif type == 'test':
            self.x_pde_val = tf.convert_to_tensor(x_pde, dtype=self.dtype)
            self.y_pde_val = tf.convert_to_tensor(y_pde, dtype=self.dtype)
            self.x_bc_val = tf.convert_to_tensor(x_bc, dtype=self.dtype)
            self.y_bc_val = tf.convert_to_tensor(y_bc, dtype=self.dtype)
            self.x_trans_0_val = tf.convert_to_tensor(x_trans_0, dtype=self.dtype)
            self.x_trans_1_val = tf.convert_to_tensor(x_trans_1, dtype=self.dtype)
            self.y_trans_val = tf.convert_to_tensor(y_trans, dtype=self.dtype)
        else:
            print('ERROR: data type not recognised. possible choises for type: \'training\', \'test\' ')

    def train(self, mode, batchsize, trainsteps, tracking, shuffle=True, max_time=False, early_stopping=False, **kwargs):
        #print(kwargs)
        print('STARTING TRAINING \nmethod = {} \nbatch size = {}  \ntrain steps = {}\nshuffle \t= {}\n'.format(mode,
                                                                                                    batchsize,
                                                                                                    trainsteps,
                                                                                                    shuffle))
        print('SAMPLES (pde, trans, bc) = \t{}'.format(self.n_samples))
        print('LOSS (pde, trans, bc) = \t{}/\t{} (user)/(normalize) = {} (actual)\n\n'.format(self.weight_losses_user, self.weight_losses_normalize,self.weight_losses))

        self.writer = kwargs.get('writer')
        if self.writer is not None:
            self._tensorboard_setup(self.model)

        tic = time.time()
        self.timer = tic

        if mode == "sc":
            loss_train_history, loss_val_history = self._train_sc(trainsteps, tracking, early_stopping)

        elif mode == "tf":
            self._dataset(batchsize, trainsteps, shuffle)
            loss_train_history, loss_val_history = self._train_tf(batchsize, trainsteps, tracking, kwargs.get('clr'), max_time, early_stopping)
        elif mode =="tfp":
            loss_train_history, loss_val_history = self._train_tfp(trainsteps, tracking)
        else:
            print("training mode not specified or not understood. possibilities: \'sc\', \'tf\', \'tfp\'")

        self.step += len(loss_train_history)
        self.last_traintime = time.time() - tic
        print("training finished: {} train steps in {} s".format(len(loss_train_history), "{:.2f}".format(self.last_traintime)))
        self.losshistory_train = np.append(self.losshistory_train, loss_train_history, axis=0)
        self.losshistory_val = np.append(self.losshistory_val, loss_val_history, axis=0)

    def _train_tf(self, batchsize, trainsteps, tracking, clr, max_time, early_stopping):

        class EarlyStopping:
            def __init__(self, max_steps):
                self.max_steps = max_steps
                self.counter = 0
                self.loss_min = 1000000.
            def check(self, losses_early_train, losses_early, weights, model):
                loss_early = np.sum(losses_early*weights)
                loss_early_train = np.sum(losses_early_train*weights)
                if loss_early < self.loss_min:
                    self.losses_train = losses_early_train
                    self.losses_val = losses_early

                    self.loss_train_min = loss_early_train
                    self.loss_min = loss_early
                    self.best_weights = model.get_weights()
                    self.counter = 0
                else:
                    self.counter += 1
                    print('\t stopping counter: \t [{}]'.format(self.counter))

                if self.counter >= self.max_steps:
                    return True
                else:
                    return False

        self.early_stopper = EarlyStopping(early_stopping)
        self.train_status = "tf"
        loss_history = np.zeros([trainsteps, 3])
        loss_history_validation = np.zeros([trainsteps, 3])
        trackinterval = trainsteps//tracking

        starttime = time.time()

        if clr is not None:
            setup = getattr(clr, 'setup', None)
            if callable(setup):
                clr.setup(trainsteps)

        if self.tracker_learningrates is not None:
            self.tracker_learningrates_w0 = self._flatten(self.model.get_weights())

        batches = max(self.n_samples)//batchsize

        dataset = self.dataset.repeat()
        epoch = 0
        for step, ((x_pde, y_pde), (x_trans_0, x_trans_1, y_trans)) in zip(range(trainsteps), dataset):

            # update optimizer hyperparameters
            if clr is not None:
                lr, mom = clr(step)
                tf.keras.backend.set_value(self.optimizer.learning_rate, lr)
                if hasattr(self.optimizer, 'momentum'):
                    tf.keras.backend.set_value(self.optimizer.momentum, mom)
            losses, grad = self._train_step(x_pde, y_pde, self.x_bc, self.y_bc, x_trans_0, x_trans_1, y_trans)

            if self.tracker_learningrates is not None:
                new_weights = self._flatten(self.model.get_weights())
                w_difference = new_weights - self.tracker_learningrates_w0
                self.tracker_learningrates.append(w_difference/self._flatten_tf(grad))
                self.tracker_learningrates_w0 = new_weights

            loss_history[step] = losses.numpy()/self.weight_losses
            if step % batches == 0:
                epoch += 1

            if step % trackinterval == 0:
                ll = self._loss_validation_and_print(self.step + step , epoch, loss_history[step])
                loss_history_validation[step] = ll

                if early_stopping:
                    if step >= trackinterval:
                        check = self.early_stopper.check(helper_nan_replacer(loss_history[step]), loss_history_validation[step], self.weight_losses, self.model)
                        if check:
                            try:
                                self.model.set_weights(self.early_stopper.best_weights)
                            except:
                                print('setting weights failed')
                            break

            step += 1

            if max_time:
                if time.time() - starttime >= 0.99*max_time:
                    break



        loss_history_validation = loss_history_validation[:step]
        loss_history = loss_history[:step]
        try:
            loss_history_validation[-1] = self._loss_validation_and_print(self.step + step, step + 1, loss_history[-1])
        except:
            pass

        return loss_history, loss_history_validation

    def _train_sc(self, trainsteps, tracking, early_stopping, tol = 1.0e1):
        self.train_status = "sc"

        loss_history = np.zeros([trainsteps, 3])
        loss_history_validation = np.zeros([trainsteps, 3])
        trackinterval = trainsteps // tracking
        self.w0 = self.model.get_weights()

        class Callback:
            def __init__(self, trackinterval, val_loss_func, loss_weights, external_step, model, early_stopping):
                self.step = 0
                self.external_step = external_step
                self.trackinterval = trackinterval
                self.val_loss_func = val_loss_func
                self.model = model
                self.weights = loss_weights
                self.loss_val_min = 100000000.
                self.counter = 0
                self.early_stopping = early_stopping

                self.best_losses_train = 0.
                self.best_losses_val = 0.

            def __call__(self, x):
                if self.step % self.trackinterval == 0:
                    loss_history_validation[self.step] = self.val_loss_func(self.external_step + self.step, self.step + 1, loss_history[self.step])

                    if self.early_stopping:
                        loss_early_train = np.sum(helper_nan_replacer(loss_history[self.step])*self.weights)
                        loss_early_val = np.sum(helper_nan_replacer(loss_history_validation[self.step])*self.weights)
                        if loss_early_val < self.loss_val_min:
                            self.best_losses_train = loss_history[self.step]
                            self.best_losses_val = loss_history_validation[self.step]
                            self.loss_train_min = loss_early_train
                            self.loss_val_min = loss_early_val
                            self.best_weights = self.model.get_weights()
                            self.counter = 0
                        else:
                            self.counter += 1
                            print('previous < actuall: {} < {}'.format(self.loss_val_min, loss_early_val))
                            print('\t stopping counter: \t [{}]'.format(self.counter))

                        if self.counter >= self.early_stopping:
                            # set best weights if stopped early
                            self.model.set_weights(self.best_weights)
                            raise RuntimeWarning('stopped early!')

                self.step += 1



        callback = Callback(trackinterval, self._loss_validation_and_print, self.weight_losses, self.step, self.model, early_stopping)




        def target(weights):
            w = self._unflatten(self.w0, weights)
            self.model.set_weights(w)
            losses, grad = self._train_step(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.x_trans_0, self.x_trans_1, self.y_trans)
            loss_history[callback.step] = losses.numpy()/self.weight_losses
            return [np.sum(losses.numpy()), self._flatten_tf(grad)]

        # call optimizer
        try:
            self.optimizer_scipy = optimize.minimize(fun=target,
                                                x0=self._flatten(self.w0),
                                                callback=callback,
                                                jac=True,
                                                method='L-BFGS-B',
                                                options={'maxiter': trainsteps,
                                                         'maxfun': 10*trainsteps,
                                                         'maxcor': 50,
                                                         'maxls': 50,
                                                         'ftol' : tol * np.finfo(float).eps,
                                                         'gtol': tol * np.finfo(float).eps
                                                         })
            final_steps = self.optimizer_scipy.nit
            mess = self.optimizer_scipy.message
        except:
            final_steps = callback.step
            mess = 'stopped early!'

        loss_history_validation = loss_history_validation[:final_steps]
        loss_history = loss_history[:final_steps]
        try:
            loss_history_validation[-1] = self._loss_validation_and_print(self.step + callback.step, callback.step + 1, loss_history[-1])
        except:
            pass

        if early_stopping:
            try:
                loss_history_validation[-1] = callback.best_losses_val
                loss_history[-1] = callback.best_losses_train
            except:
                loss_history = np.ones([1, 3])*1000.
                loss_history_validation = np.ones([1, 3])*1000.
        print("exit message: ", mess, " iterations: ", final_steps)

        return loss_history, loss_history_validation

    def _train_step_precompiled(self, x_pde, y_pde, x_bc, y_bc, x_trans_0, x_trans_1, y_trans):
        with tf.GradientTape() as tape:
            y, dy = self.output_and_derivative(self.model, x_pde, training=True)
            u = self.case.env.feedback(x_pde)
            losses = tf.stack([self.loss_fn_pinn(y_pde, self.case.pde(x_pde, u, y, dy)),
                               self.loss_fn_transported(y_trans, self.model(x_trans_0, training=True) - self.model(x_trans_1, training=True)),
                               self.loss_fn_boundary(y_bc, self.model(x_bc, training=True))], axis=0)
            loss = tf.reduce_sum(losses, axis=0) + sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.train_status == "tf":
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return losses, gradients

    def _losses(self, y_pde_target, y_pde_pred, y_bc_target, y_bc_pred, y_transported_target, y_transported_pred):
        loss_pinn = tf.reduce_mean(tf.square(y_pde_target - y_pde_pred))
        loss_transported = tf.reduce_mean(tf.square(y_transported_target - y_transported_pred))
        loss_boundary = tf.reduce_mean(tf.square(y_bc_target - y_bc_pred))
        return tf.stack([loss_pinn, loss_transported, loss_boundary], axis=0)

    def _loss_validation_and_print(self, step, epoch, losses):
        #validation error
        def loss_validation():
            y, dy = self.output_and_derivative(self.model, self.x_pde_val, training=False)
            u = self.case.env.feedback(self.x_pde_val)
            y_bc_pred = self.model(self.x_bc, training=False)
            y_pde_pred = self.case.pde(self.x_pde_val, u, y, dy)
            y_transported_pred = self.model(self.x_trans_0_val, training=False) - self.model(self.x_trans_1_val, training=False)
            losses = self._losses(self.y_pde_val, y_pde_pred, self.y_bc, y_bc_pred, self.y_trans_val, y_transported_pred)
            return losses.numpy()

        if self.writer is not None:
            self._tensorboard_write(step)

        if self.tracker_weights is not None:
            self.tracker_counter += 1
            self.tracker_weights.append("tracker\\weights_{}_{}.tracker".format(self.timestamp, self.tracker_counter))
            with open(self.tracker_weights[-1], "wb") as file:
                pickle.dump(self.model.get_weights(), file)

        if self.tracker_grads is not None:
            self.tracker_grads_counter += 1
            self.tracker_grads.append("tracker\\grads_{}_{}.tracker".format(self.timestamp, self.tracker_grads_counter))
            with open(self.tracker_grads[-1], "wb") as file:
                grads = self._train_step_precompiled(self.x_pde, self.y_pde, self.x_bc, self.y_bc, self.x_trans_0, self.x_trans_1, self.y_trans)[1]
                grads = [g.numpy() for g in grads]
                pickle.dump(grads, file)

        if self.tracker_vals is not None:
            self.tracker_vals_counter += 1
            self.tracker_vals.append("tracker\\vals_{}_{}.tracker".format(self.timestamp, self.tracker_vals_counter))
            with open(self.tracker_vals[-1], "wb") as file:
                vals, _ = forward_pass(self.model, self.activation_for_forwardpass_test, self.x_pde.numpy())
                pickle.dump(vals, file)

        val_losses = loss_validation()

        print("step {}, epoch {}   |   loss [pde, trans, bc]:    train = [{}  {}  {}]    val = [{}  {}  {}]  {} s".format(step, epoch, "{0:.2E}".format(losses[0]), "{0:.2E}".format(losses[1]), "{0:.2E}".format(losses[2]), "{0:.2E}".format(val_losses[0]), "{0:.2E}".format(val_losses[1]), "{0:.2E}".format(val_losses[2]), "{:.2f}".format(time.time()-self.timer)))
        self.timer = time.time()
        return val_losses

    def _dataset(self, batchsize, steps, shuffle=True):
        ratio = self.n_samples[0]/self.n_samples[1]
        if self.n_samples[0] >= self.n_samples[1]:
            batchsize_pde = batchsize
            batchsize_trans = round(batchsize/ratio)
        else:
            batchsize_trans = batchsize
            batchsize_pde = round(batchsize*ratio)
        self.batchsizes = [batchsize_pde, batchsize_trans]

        dataset_pde = tf.data.Dataset.from_tensor_slices((self.x_pde, self.y_pde))
        dataset_trans = tf.data.Dataset.from_tensor_slices((self.x_trans_0, self.x_trans_1, self.y_trans))

        if shuffle == True:
            dataset_pde = dataset_pde.shuffle(batchsize_pde)
            dataset_trans = dataset_trans.shuffle(batchsize_trans)

        dataset_pde = dataset_pde.batch(batchsize_pde, True)
        dataset_trans = dataset_trans.batch(batchsize_trans, True)
        self.dataset = tf.data.Dataset.zip((dataset_pde, dataset_trans)).prefetch(2)

    def _flatten(self, w_list):
        n = len(w_list)
        weights = np.array([])
        for i in range(n):
            weights = np.append(weights, w_list[i].flatten())
        return weights

    def _flatten_tf(self, w_list):
        n = len(w_list)
        weights = np.array([])
        for i in range(n):
            weights = np.append(weights, w_list[i].numpy().flatten())
        return weights

    def _unflatten(self, w_list, w_new):
        #w_new = w_new.astype(self.float)
        for i in range(len(w_list)):
            w = w_new[:w_list[i].size]
            w_new = w_new[w_list[i].size:]
            w_list[i] = w.reshape(w_list[i].shape)
        return w_list

    def output_and_derivative(self, model, x, training):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=training)
        dy = tape.gradient(y, x)
        return y, dy


# helper function nan replacer
def helper_nan_replacer(array, value=0.):
    i = np.isnan(array)
    array[i] = value
    return array


# policy iteration step
def pi_step(case,
            controller,
            model_class,
            initializer,
            loss_weights,
            model_pretrain,
            X_train,
            X_test,
            trans_dt=1e-3,
            trans_T=0.5,
            train_tf = 2000,
            train_sc = 100000):

    # set feedback
    case.env.set_feedback(controller)

    # create container model
    v = model_class(*initializer)

    # create trainable model
    model = model_class(*initializer)

    # trainer class
    trainer = Trainer(case)
    trainer.datatransporter(X_train, trans_dt, int(trans_T/trans_dt), type='training')
    trainer.datatransporter(X_test, trans_dt, int(trans_T/trans_dt), type='test')

    # training
    trainer.compile(model, loss_weights, normalize=True)
    if model_pretrain != 0:
        trainer.train('tf', batchsize=max(len(X_train)//8,1), trainsteps=train_tf, tracking=20, clr=OneCycle(lr_range=(1e-6, 1e-2)), early_stopping=2)
    trainer.train('sc', batchsize=None, trainsteps=train_sc, tracking=int(train_sc/1000))

    # save weights into container model
    v.set_weights(model.get_weights())

    # create controller
    controller = case.feedback_from_value(v)
    return controller, v
