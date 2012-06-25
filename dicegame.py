import random

"""
Andrew Miller  June 2012

In this dice game, a Player and the House roll dice in rounds. They
are both trying to collect 1's (successes). During each round, a total 
of N dice are rolled. In a normal round, the Player rolls (N-f) dice
while the House rolls (f) dice. If at any time the Player rolls a 1, then 
D penalty rounds begin. During a penalty round, the House gets to roll
all of the dice.

My goal is to obtain:

  a) an expression for r, a number of rounds to play for, so that 
     after r rounds the Player wins except for a negligible probability

  b) a choice of p, the probability of success on single dice roll,
     so that r is minimized

"""

# Fixed environmental parameters
N = 3   # Total number of dice rolled in each round
f = 1   # Number of dice rolled by the House in a normal round
D = 2   # Number of penalty rounds that begin when Player rolls a 1
assert N >= 2*f + 1

# Protocol parameters
p = 1/24.   # Probability of success on a dice roll


class DiceGame():
    def __init__(self):
        self.penalty = 0  # Number of penalty rounds
        self.house = 0    # House's score
        self.player = 0   # Player's score

    def roll(self, hand):
        assert hand in ('player','house')
        if random.random() <= p:  # Pr[success] = p
            if hand == 'player': 
                self.player += 1
                self.penalty = D
            else:
                self.house += 1

    def round(self):
        # Player gets to roll (N-f) dice, 
        for _ in range(N-f):
            if not self.penalty: self.roll('player')
            else: self.roll('house')

        for _ in range(f):
            self.roll('house')

        if self.penalty: self.penalty -= 1



"""
Some attempts at analysis begin here
"""

def make_transition_kernel():
    S = D

    global P
    P = np.zeros((S, S))

    P[0,0] = (1-p)**(N-f)
    P[0,1] = 1 - P[0,0]

    for s in range(1,S):
        P[s, (s+1)%S] = 1


    return P

class MarkovChain():
    def __init__(self, P):
        self.P = P
        s = P.shape[0]
        assert P.shape == (s,s)

        # Stationary distribution
        w, v = np.linalg.eig(P)

        state = 0

import scipy.linalg

def simulate(rounds):
    game = DiceGame()
    
    global X_C, X_A, R_C, R_A
    X_C = [0]
    X_A = [0]
    R_C = [0]
    R_A = [0]

    for r in range(rounds):
        game.round()
        X_C.append(game.player)
        X_A.append(game.house)
        if len(R_C) == game.player: R_C.append(r)
        if len(R_A) == game.house: R_A.append(r)


def player_bound(k, r):
    r = float(r)
    A = 1 + 1./((N-f)*D*p)
     #B = -sqrt(k/(2*p))
    B = -sqrt(2*k)
    C = -r/D
    sqrt_mu = (-B + sqrt(B**2 - 4*A*C)) / (2*A)
    #sqrt_mu = (-B - sqrt(B**2 - 4*A*C)) / (2*A)
    x = (r - sqrt_mu**2 / ((N-f)*p) ) / D
    return x

def house_bound(k, r):
    # Return a threshold x such that Pr(X_A[r] >= x) < exp(-k)

    #Chernoff's bound, multiplicative form
    #mu = N*r*p
    #b = sqrt(2*k/mu)
    #return (1+b) * mu / 2
    #return (mu + sqrt(2*mu*k))/2

    # Chernoff's bound, additive form
    # (Additive form is a much weaker bound! Use multiplicative!)
    #within_eps = lambda a, b: np.abs(a-b) < 1e-5
    #assert within_eps((N*r*p + np.sqrt(N*r*k/2))/2, (p + e) * N * r)
    e = sqrt(float(k)/(2*N*r))/2
    #return (p + e) * N * r
    return (N*r*p + np.sqrt(N*r*k/2))/2


def simulate_player(rounds):
    x = []
    

    
def once(rounds):
    simulate(rounds)
    r = np.array(range(2,rounds))[::1000]
    xc = np.array(X_C)[r].astype('f')
    xa = np.array(X_A)[r].astype('f')

    figure(1)
    plot(r, xc/r, 'b')
    plot(r, xa/r, 'r')
    plot(r, (xa+xc)/2/r, 'g')
    plot(r, (xc-xa)/2/r, 'y')
    #plot(x, (xc+xa)/2, 'g')
    k = 6
    global pb
    #pb = map(lambda r: player_bound(k, r)/x, x)
    plot(r, map(lambda r: player_bound(k, r)/r, r), 'k')
    plot(r, map(lambda r: house_bound(k, r)/r, r), 'c')


    x = range(max(xc))
    r = map(lambda x: x*(D +1./((N-f)*p)), x)
    #plot(r, x, 'k')


    #figure(2)
    rc = np.array(R_C)
    ra = np.array(R_A)
    #plot(rc/x, 'b')
    #plot(ra/x, 'r')
