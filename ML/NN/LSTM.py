from activations import *


class LSTM:

    h_state = []
    c_state = []
    initial_h_state = None
    initial_c_state = None

    """---------------
    Gate activations
    ---------------"""
    a = None
    i = None
    f = None
    o = None

    """--------------------
    Trainable parameters
    --------------------"""
    W_f = None
    W_i = None
    W_o = None
    W_a = None

    """-------------
    Local Gradients
    -------------"""
    ft_grad = []
    it_grad = []
    ct_grad = []
    ut_grad = []

    def __init__(self, input_dim, output_dim, batch_size, T):
        input_dim += output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.W_f = np.random.standard_normal(size=[input_dim, output_dim])
        self.W_i = np.random.standard_normal(size=[input_dim, output_dim])
        self.W_o = np.random.standard_normal(size=[input_dim, output_dim])
        self.W_a = np.random.standard_normal(size=[input_dim, output_dim])
        self.initial_h_state = np.zeros([batch_size, output_dim])
        self.initial_c_state = np.zeros([batch_size, output_dim])

        # Logical gate initialisation
        self.a = np.zeros([T, batch_size, output_dim])
        self.i = np.zeros([T, batch_size, output_dim])
        self.f = np.zeros([T, batch_size, output_dim])
        self.o = np.zeros([T, batch_size, output_dim])

    def __clear_grad(self):
        self.f_grad = []
        self.i_grad = []
        self.c_grad = []
        self.u_grad = []

    def __forward(self, h_state, c_state, x, t):
        """
        Forward pass of LSTM unit for a single time step
        :param h_state: [N x hidden size]
        :param c_state: [N x hidden size]
        :param x: input at time step t
        :return: updated cell state, and hidden state
        """
        input_data = np.hstack([h_state, x])
        self.a[t] = tanh(input_data @ self.W_a)
        self.i[t] = sigmoid(input_data @ self.W_i)
        self.f[t] = sigmoid(input_data @ self.W_f)
        self.o[t] = sigmoid(input_data @ self.W_o)
        # cell state
        state = self.a[t] * self.i[t] + self.f[t] * c_state
        # hidden state
        output = tanh(state) * self.o[t]

        return state, output

    def forward(self, X, training=True):
        """
        Forward pass of LSTM unit for all time steps
        :param x: sequence of one hot encoded arrays, each array representing a time step
        :param training: switch that signals the program to compute the gradient
        :return: updated cell state, and hidden state
        """

        if not training:
            self.clear_grad()
            self.h_state = []
        T, N, vocab_size = X.shape
        h_state = self.initial_h_state
        c_state = self.initial_c_state
        h_state_list = []
        c_state_list = []
        for t in range(T):
            input_data = X[t]
            c_state, h_state = self.__forward(h_state, c_state, input_data, t=t)
            c_state_list.append(c_state.reshape([1, N, self.output_dim]))
            h_state_list.append(h_state.reshape([1, N, self.output_dim]))
        c_state = np.concatenate(c_state_list, axis=0)
        h_state = np.concatenate(h_state_list, axis=0)
        if training:
            self.h_state = h_state
            self.c_state = c_state  # Just the last time step alone
        return h_state

    def __weight_update(self, inc_grad1, inc_grad2, X, t):
        input_data = None

        if t == 0:
            input_data = np.hstack([self.initial_h_state, X[0]])
        else:
            input_data = np.hstack([self.h_state[t - 1], X[t]])

        f = self.f[t]
        a = self.a[t]
        i = self.i[t]
        o = self.o[t]
        f_acc = input_data.T @ (inc_grad1 * self.c_state[t - 1] * (f * (1 - f)))
        a_acc = input_data.T @ (inc_grad1 * i * (1 - a ** 2))
        i_acc = input_data.T @ (inc_grad1 * a * (i * (1 - i)))
        o_acc = input_data.T @ (inc_grad2 * (o * (1 - o)))

        back = inc_grad1 * f

        return {'f_acc': f_acc, 'a_acc': a_acc, 'i_acc': i_acc, 'o_acc': o_acc,
                'back': back}

    def weight_update(self, inc_grad, X):
        """

        :param inc_grad: gradient from layer above. Dims: list of matrix N x Hidden size whose length should be T
        :param X: T x N x Feature size
        :return:
        """

        T = X.shape[0]
        # f_grad = np.zeros(self.W_f.shape)
        # i_grad = np.zeros(self.W_i.shape)
        # o_grad = np.zeros(self.W_o.shape)
        # a_grad = np.zeros(self.W_a.shape)

        top_grad = None
        top_grad2 = None
        for t in range(T-1, -1, -1):
            if t == T-1:
                # input_data = np.hstack([self.h_state[t - 1], X[t]])
                # top_grad = inc_grad * (input_data @ self.W_o) * \
                #            (1 - tanh(self.c_state[t])**2)
                tanh_c = tanh(self.c_state[t])
                top_grad = inc_grad * self.o[t] * (1 - tanh_c**2)
                top_grad2 = inc_grad * tanh_c

            grad = self.__weight_update(top_grad, top_grad2, X, t)
            # f_grad += grad['f_acc']
            # i_grad += grad['i_acc']
            # o_grad += grad['o_acc']
            # a_grad += grad['a_acc']

            # Delete this if the network doesn't train
            tmp_grad = grad['f_acc'] + grad['i_acc'] + grad['o_acc'] + grad['a_acc']
            # f_grad += tmp_grad
            # i_grad += tmp_grad
            # o_grad += tmp_grad
            # a_grad += tmp_grad
            #------------------------------------------


            top_grad = grad['back']
            #top_grad2 = grad['back']
            # Delete this if the network doesn't train
            top_grad2 = 1

        self.W_f -= 1e-3 * tmp_grad
        self.W_i -= 1e-3 * tmp_grad
        self.W_o -= 1e-3 * tmp_grad
        self.W_a -= 1e-3 * tmp_grad


#         return {'ft_acc': ft_acc, 'it_acc': it_acc, 'ct_acc': ct_acc, 'ut_acc': ut_acc,
#                 'ft_back': ft_back, 'it_back': it_back, 'ct_back': ct_back, 'ut_back': ut_back}