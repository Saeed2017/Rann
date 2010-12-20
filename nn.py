import numpy as N
import scipy as S
import scipy.optimize as O

import itertools as I

@N.vectorize
def logistic(x):
    return 1/(1 + N.exp(-x))

class Net(object):
    def __init__(self, n_in, n_out, n_hidden, n_recurrent = None):
        if n_recurrent == None: n_recurrent = n_hidden
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent
        self.topo = (n_in, n_out, n_hidden, n_recurrent)

        self.w_init()

    def w_init(self):
        # The first layer of weights, input -> hidden
        w1 = N.random.normal(0, 0.1, (self.n_hidden, self.n_in))

        # The weights from the memorized nodes, memory -> hidden
        wm = N.random.normal(0, 0.1, (self.n_hidden, self.n_recurrent))

        # The second layer of weights, hidden -> output
        w2 = N.random.normal(0, 0.1, (self.n_out, self.n_hidden))

        # The biases to the hidden and output layer
        wbh = N.random.normal(0, 0.1, (self.n_hidden, 1))
        wbo = N.random.normal(0, 0.1, (self.n_out, 1))

        self.w_0 = self.flatten_weights(w1, wm, wbh, w2, wbo)

        # The weight vector indicies
        self.i_w1  = N.array((0, self.n_hidden*self.n_in))
        self.i_wm  = self.i_w1[1] + N.array((0, self.n_hidden*self.n_recurrent))
        self.i_wbh = self.i_wm[1] + N.array((0, self.n_hidden))
        self.i_w2  = self.i_wbh[1] + N.array((0, self.n_out*self.n_hidden))
        self.i_wbo = self.i_w2[1] + N.array((0,  self.n_out))

    def flatten_weights(self, w1, wm, wbh, w2, wbo):
        w = N.r_[N.reshape(w1, -1),
                N.reshape(wm, -1),
                N.reshape(wbh, -1),
                N.reshape(w2, -1),
                N.reshape(wbo, -1)]
        return N.atleast_2d(w).transpose()        

    def get_w1(self, w):
        i0, i1 = self.i_w1
        return N.reshape(w[i0:i1], (self.n_hidden, self.n_in))

    def get_wm(self, w):
        i0, i1 = self.i_wm
        return N.reshape(w[i0:i1], (self.n_hidden, self.n_recurrent))

    def get_wbh(self, w):
        i0, i1 = self.i_wbh
        return N.reshape(w[i0:i1], (self.n_hidden, 1))

    def get_w2(self, w):
        i0, i1 = self.i_w2
        return N.reshape(w[i0:i1], (self.n_out, self.n_hidden))

    def get_wbo(self, w):
        i0, i1 = self.i_wbo
        return N.reshape(w[i0:i1], (self.n_out, 1))

    def set_w1(self, w, w1):
        i0, i1 = self.i_w1
        w[i0:i1] = N.reshape(w1, -1)
        return w

    def set_wm(self, w, wm):
        i0, i1 = self.i_wm
        w[i0:i1] = N.reshape(wm, -1)
        return w

    def set_wbh(self, w, wbh):
        i0, i1 = self.i_wbh
        w[i0:i1] = N.reshape(wbh, -1)
        return w

    def set_w2(self, w, w2):
        i0, i1 = self.i_w2
        w[i0:i1] = N.reshape(w2, -1)
        return w

    def set_wbo(self, w, wbo):
        i0, i1 = self.i_wbo
        w[i0:i1] = N.reshape(wbo, -1)
        return w

    def forward(self, w, epoch):
        I, T = epoch
        l = I.shape[1]

        H = N.dot(self.get_w1(w), I) + self.get_wbh(w)
        for n in xrange(1, l):
            H[:, n] = H[:, n] + N.dot(self.get_wm(w), H[0:self.n_recurrent, n-1])
        H = logistic(H)

        O = N.dot(self.get_w2(w), H) + self.get_wbo(w)
        O = logistic(O)

        return (H, O)  

    def cross_entropy(self, w, epoch):
        I, T = epoch
        H, O = self.forward(w, epoch)
        return N.sum(- T*N.log(O) - (1-T)*N.log(1-O))

    def gradient(self, w, epoch):
        I, T = epoch
        H, O = self.forward(w, epoch)

        # Compute deltas
        Do = O - T
        Dh = N.dot(self.get_w2(w).transpose(), Do)*H*(1-H)

        # Compute gradients
        Hdelay = N.zeros(H.shape)
        Hdelay[:, 1:] = H[:, :-1]

        gradw2 = N.dot(Do, H.transpose())
        gradw1 = N.dot(Dh, I.transpose())
        gradwm = N.dot(Dh, Hdelay[0:self.n_recurrent, :].transpose())
        gradbo = N.sum(Do, axis = 1)
        gradbh = N.sum(Dh, axis = 1)

        grad = self.flatten_weights(gradw1, gradwm, gradbh, gradw2, gradbo)
        return N.squeeze(grad)

    def optimize_weights(self, w, epoch, *args, **kwargs):
        w = S.optimize.fmin_bfgs(self.cross_entropy, \
                w, self.gradient, (epoch,), \
                maxiter = 10, disp = True, full_output = False,
                *args, **kwargs)
        return w



#     def map(self, stream, independent = True):
#         """
#         Passed a stream of compatible inputs, `stream`, returns an
#         iterator which computes the output values from the inputs in
#         the stream. If `independent` is set then the memory will be
#         reset before running the stream.
#         """

#         if independent: self.reset()

#         for i in stream:
#             i = N.asarray(i)

#             # Compute the hidden nodes
#             h = N.dot(self.w1, i) \
#               + N.dot(self.wm, self.__memory) \
#               + self.wbh
#             h = logistic(h)

#             # Store the new memory
#             self.__memory = h[0:self.n_recurrent]

#             # Compute the output nodes
#             o = N.dot(self.w2, h) \
#               + self.wbo
#             o = logistic(o)

#             # Yield the output for this turn of the iterator
#             yield o

def train(model, epoch, n_models = 40):
    out = []
    for n in xrange(n_models):
        model.w_init()
        w = model.optimize_weights(model.w_0, epoch)
        out.append((w, model.cross_entropy(w, epoch)))
    m = min(out, key = lambda x: x[1]) 
    return m
        
