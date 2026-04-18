import numpy as np

np.random.seed(42)

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
    
    def backprop(self, dL_dout, learning_rate):
        for i, gradient in enumerate(dL_dout):
            if gradient == 0.:
                continue
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            dout_dt = -t_exp[i] * t_exp / (S**2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = self.weights
            dL_dt = gradient * dout_dt
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_inputs = dt_dinputs @ dL_dt

            self.weights = self.weights - learning_rate * dL_dw
            self.biases = self.biases - learning_rate * dL_db
            return dL_inputs.reshape(self.last_input_shape)

