# App: solve some difference equation
# Problem: relation between callback and yield/iterator.
import sys

# logistic growth, no storage!
def logistic_growth(u0, r, M, N, user_action=lambda u, n: None):
    index_set = range(N)
    for n in index_set:
        u = u0 + r*u0*(1 - u0/float(M(n, u0)))
        user_action(u, n+1)
        u0 = u

# one action is to store for later plotting
# one action is to plot the evolution as animation
simulator = logistic_growth

def mycompute():
    print 'in mycompute'
    u0 = 10
    r = 0.1
    N = 100

    def M(n, u):
        return 100

    u_list = []
    def myaction(u, step):
        u_list.append(u)

    simulator(u0, r, M, N, myaction)
    print u_list

mycompute()

sys.exit(0)

def visualizer2(compute):
    # compute is an iterator
    import matplotlib.pyplot as plt
    import time
    plt.ion()
    for result in compute():
        r, s, text = result
        plt.clf()
        plt.axis([-pi/2, 2*pi, -0.5, 1.8])
        plt.text(r, s, text, fontsize=14, rotation=r/pi*180,
                 horizontalalignment='center',
                 verticalalignment='center')
        plt.draw()
        time.sleep(1)
    plt.show()

def visualizer3(compute):
    # compute is not an iterator, will call its things with callbacks,
    # leave the technical steps of animation to compute
    import matplotlib.pyplot as plt
    plt.ion()
    compute(plt)
    plt.show()

def mycompute3(plt):
    print 'in mycompute3'
    import time
    def mycallback(r, s):
        text = 'Hello, World! sin(%g)=%g' % (r, s)
        plt.clf()
        plt.axis([-pi/2, 2*pi, -0.5, 1.8])
        plt.text(r, s, text, fontsize=14, rotation=r/pi*180,
                 horizontalalignment='center',
                 verticalalignment='center')
        plt.draw()
        time.sleep(1)

    simulator(5, mycallback)

visualizer3(mycompute3)

# Simpler visualizers
def visualizer(compute):
    print 'in visualizer'
    for result in compute():
        print 'in visualizer loop:', result

def visualizer(compute):
    # Iterate over compute or call it once
    print 'in visualizer'
    def process(result):
        print result

    import collections
    if isinstance(compute, collections.Iterable):
        for result in compute():
            process(result)
    else:
        result = compute()
        print 'compute in visualizer was not iterator'
        process(result)


# How to glue visualizer with results from mycallback?

# Does not work because functions without yield are not iterators
# and functions with yield must be called in a for loop:
#visualizer(mycompute)

# Class for mycompute callback that is an interator
#import scitools.debug as debug

# Plain loop
n = 5
for i in range(n):
    r = i*pi/(n-1)
    s = sin(r)
    print 'Hello, World! sin(%g)=%g' % (r, s)

# Pythonic variant
r_values = [i*pi/(n-1) for i in range(n)]
s_values = [sin(r) for r in r_values]
for r, s in zip(r_values, s_values):
    print 'Hello, World! sin(%g)=%g' % (r, s)

class MyCompute1:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return MyComputeIterator(self.n)

class MyComputeIterator:
    def __init__(self, n):
        self.n = n
        self.i = 0  # iteration counter

    def next(self):
        if self.i < self.n:
            result = self.simulate()
            self.i += 1
            return result
        else:
            raise StopIteration

    def simulate(self):
        r = self.i*pi/(self.n-1)
        s = sin(r)
        return r, s

class MyCompute2:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i < self.n:
            result = self.simulate()
            self.i += 1
            return result
        else:
            raise StopIteration

    def simulate(self):
        r = self.i*pi/(self.n-1)
        s = sin(r)
        return r, s

class MyCompute3:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            self.i = i
            yield self.simulate()

    def simulate(self):
        r = self.i*pi/(self.n-1)
        s = sin(r)
        return r, s

for r, s in MyCompute1(5):
    print 'Hello, World! sin(%g)=%g' % (r, s)

for r, s in MyCompute2(5):
    print 'Hello, World! sin(%g)=%g' % (r, s)

for r, s in MyCompute3(5):
    print 'Hello, World! sin(%g)=%g' % (r, s)

class MyCompute4:
    # Using simulator and visualizer, storing results
    def __init__(self, n):
        self.n = n
        self.database = []
        simulator(self.n, self.mycallback)

    def __iter__(self):
        for result in self.database:
            yield result

    def mycallback(self, r, s):
        self.database.append(
            (r, s, 'Hello, World! sin(%g)=%g' % (r, s)))

    def __call__(self):
        return self.__iter__()

visualizer2(MyCompute4(5))
visualizer(MyCompute4(5))

def simulator2(n, callback):
    print 'in simulator2'
    for i in range(n):
        r = i*pi/(n-1)
        s = sin(r)
        print 'in simulator2, s', s
        yield r, s

class MyCompute5:
    # Using simulator and visualizer
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for r, s in simulator2(self.n, self.mycallback):
            yield self.mycallback(r, s)

    def mycallback(self, r, s):
        return r, s, 'Hello, World! sin(%g)=%g' % (r, s)

    def __call__(self):
        return self.__iter__()

visualizer(MyCompute5(5))

