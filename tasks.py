import numpy as N
import scipy as S
import scipy.optimize as O

import itertools as I

def singleton(klass, *args, **kwargs):
    """
    Class decorator which turns the class name into a singleton instance instead of
    a constructor. Input parameters are specified statically.
    """
    k = klass(*args, **kwargs)
    return k

@singleton
class Timer(object):
    """ 
    The *Timer* task is a purely temporal task where the net learns a
    particular delay in sequence trained by the length of an activation period
    which is immediately followed by an equal-length inactivation period. The
    target output of the network is silence with an impulse when the net
    expects the next training run to appear.

    This task tests long-range temporal learning and is ideal for testing the
    multiple-delay memory register hypotheses.
    """

    def __init__(self):
        pass

    def epoch(self, mu, n = 100):
        delays = N.random.poisson(mu, n)
        L = N.sum(delays)*2
        I = N.zeros((1, L))
        O = N.zeros((1, L))

        i0 = 0
        for d in delays:
            I[:, i0:(i0+d)] = N.ones((1, d))
            I[:, (i0+d):(i0+2*d)] = N.zeros((1, d))
            O[:, i0+2*d-1] = 1
            i0 = i0+2*d
        return (I, O)


class Utterance(object):

    def __init__(self):
        self.d = 6
        self.c = 3
        self.consonants = 'bdg'
        self.dictionary = {
                'b': [1,0,1,0,0,1],
                'd': [1,0,1,1,0,1],
                'g': [1,0,1,0,1,1],
                'a': [0,1,0,0,1,1],
                'i': [0,1,0,1,0,1],
                'u': [0,1,0,1,1,1]
               }
        self.extension = {
                'b': 'ba',
                'd': 'dii',
                'g': 'guuu',
                }

    def to_binary(self, string):
        l = len(string)
        out = N.zeros((self.d, l))
        for i, s in enumerate(string):
            out[:, i] = self.dictionary[s]
        return out

    def make_seq(self, n_base = 100):
        out = []
        for i in xrange(n_base):
            idx = N.random.randint(0, self.c)
            out.append(self.consonants[idx])
        return ''.join(out)

    def extend(self, string):
        out = []
        for s in string:
            out.append(self.extension[s])
        return ''.join(out)

    def to_string(self, binary):
        [D, L] = N.asarray(binary).shape
        out = []
        for i in xrange(L):
            for k in self.dictionary.keys():
                if N.sum(N.abs(binary[:, i] - self.dictionary[k])) == 0:
                    out.append(k)
        return ''.join(out)


    def epoch(self, n_base = 100):
        I = self.to_binary(self.extend(self.make_seq(n_base)))
        T = N.c_[I[:, 1:], N.zeros((self.d, 1))]
        return (I, T)

def make_epoch(stream, length):
    itest, ttest = stream.next()
    I = N.zeros([itest.shape and itest.shape[0] or 1, length])
    T = N.zeros([ttest.shape and ttest.shape[0] or 1, length])
    
    for n in xrange(length):
        i, t = stream.next()
        I[:, n] = i
        T[:, n] = t
    return (I, T)        

def itake(n, it):
    try:
        ans = N.array([it.next()])
    except StopIteration:
        raise ValueError('iterator contains 0 items')
    shape0 = ans.shape[1:]
    for i in xrange(n):
        x = it.next()
        ans.resize((i+2,)+shape0)
        ans[i+1] = x
    return ans

def xor_stream():
    opts = N.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    n = 0
    while True:
        i = N.squeeze(opts[:, N.mod(n, 4)])
        o = N.int_(N.logical_xor(i[0], i[1]))
        n = n + 1
        yield (i, o)
xor_stream = xor_stream()

def xor_stream_random():
    while True:
        i = N.random.randint(0, 2, 2)
        o = N.int_(N.logical_xor(i[0], i[1]))
        yield (i, o)
xor_stream_random = xor_stream_random()

def xor_sequential():
    i0 = N.random.randint(0, 2)
    o = N.random.randint(0, 2)
    yield (N.array([i0]), N.array([o]))
    while True:
        i1 = N.random.randint(0, 2)
        o = N.int_(N.logical_xor(i0, i1))
        yield (N.array([i1]), N.array([o]))
        i0 = i1
xor_sequential = xor_sequential()
