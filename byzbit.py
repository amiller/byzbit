"""
Andrew Miller   June 2012

A simulation of Bitcoin [฿] as a distributed Byzantine consensus algorithm.


In a distributed consensus algorithm, a network of N processes tries to come to
a unanimous agreement about some value. Each process ultimately has to 'decide'
on a particular value, which becomes its final answer / its output value. For
a protocol to solve the consensus problem, the following conditions must met:

    (Agreement) If any process decides on a value v, then no process decides 
         decide on any distinct value v'.

    (Termination) All processes eventually decide.

    (Weak Validity) If there are no faulty processes, then each process decides
         on a value that was given to them as an input.


What makes this problem difficult is that we have to cope with two kinds of
uncertainty in our environment: a) latency in the communications subsystem, 
and b) faulty (e.g., malicious) behavior from some minority (N >= 2f+1) of 
the processes. We model this uncertainty in our simulation by handing over 
control to an explicit Adversary. Here are the key features of our computation
model:

    a) byzantine faults
        The Adversary is given full control over the f faulty processes, i.e.,
        'pawns'.

    b) unknown network size (optional)
        We allow the protocol is allowed to depend on f, the maximum number of 
        faulty processes. However, in one variation of our model, the total 
        network size is an unknown parameter [?].

    c) anonymous processes
        Processes are not associated with any unique identifiers [?].

    d) randomization and non-determinism
        Each process is able to perform randomized coin flips [?]. The 
        Adversary can not know the outcome of future coin flips ahead of time.

    e) synchronous (and partially synchronous) communications
        Processes communicate by sending broadcast messages. All messages are
        eventually delivered after D rounds, and no messages are lost. When a
        process receives a message, it has know way of knowing where the message
        came from. In a) the synchronous model, the maximum delay D is a known 
        parameter that the protocol can depend on. In b) the partially 
        synchronous model, D exists but the protocol may not depend on it [1].

    f) collision-resistant hash functions, and randomization
        We can use collision-resistant hash functions to construct the coinflip
        primitive modeled here (see make_mint()). This is a typical kind of 
        assumption in popular cryptographic systems, although it is not known
        to be achievable in the standard model [?]. This model is at least as
        strong as one that supports digital signatures, since those can be
        implemented using hash functions [?]. However, the processes have no
        identifiers and thus no public keys.

    g) processor synchrony
        We assume that the processes are given fair access to the coinflip
        primitive. This implies that all processes run at the same speeds,
        i.e., no process gets ahead or falls behind. Another way to look at it
        is that the difference between total process speeds is included in the
        relation N >= 2f+1. In any given time interval, the Adversary can 
        perform no more coin flips than a fraction (1 - 1/f) of the number
        flipped by the correct processes, in total.


The protocol we model is a simplified form of Bitcoin. It based on observing a 
majority vote. Since the processes are anonymous, you'd think that any voting
mechanism would be susceptible to some form of Sybil attack [?]. However, 
our mechanism for voting involves computing a pricing function [?], and by
assumption the faulty processes cannot afford to vote more often than (1-1/f)
relative to the number of correct processes.



[1] Consensus in the Presence of Partial Synchrony
    Cynthia Dwork, Nancy Lynch, Larry Stockmeyer
    http://groups.csail.mit.edu/tds/papers/Lynch/jacm88.pdf

[2] Relationships Between Broadcast and Shared Memory in Reliable Anonymous 
    Distributed Systems
    James Aspnes, Faith Ellen, Eric Ruppert
    http://www.cs.yale.edu/homes/aspnes/papers/reliable-anonymous-full.pdf

[3] Random Oracles in Constantinople:
    Christian Cachin, Klaus Kursawe, Victor Shoup
    ftp://cse.osu.edu/pub/anish/dcpo/allpdf/51.pdf

[4] Randomized Protocols for Asynchronous Consensus
    James Aspnes
    http://arxiv.org/pdf/cs.DS/0209014.pdf

[฿] Bitcoin: A Peer-to-Peer Electronic Cash System
    Satoshi Nakamoto
    http://bitcoin.org/bitcoin.pdf

"""
from collections import defaultdict
import operator
import random

# Parameters
p = 0.01      # Probability of successful coinflip
N = 9         # Total number of processes
f = 4         # Maximum number of faulty processes
D = 400       # Maximum message delay
r_hat = 2000
assert N >= 2*f + 1


def Adversary(coins,                  # a set() of coins (gets replenished)
              state,                  # a read-only proxy of the simulator
              send_message,           
              let_process_run,
              let_process_receive):

    global adversary_successes
    adversary_successes = 0
    
    while not all(state.decisions[p] is not None for p in state.procs):

        # As the scheduling adversary, our goal is to disrupt the network, 
        # causing the correct processes either a) to reach a disagreement, 
        # or b) to run for as many rounds as possible

        # We have several tools at our disposal. First of all, we are able to
        # observe the entire simulation state, including the results of coin
        # flips. The only things we don't know are the results of future flips.

        # We also have the stash of coins that would have been given to the
        # faulty processes. We can flip them immediately, or save for later. 
        # More coins are added to this set at the beginning of each round 
        # (after all the correct processes have gotten to run the previous 
        # round).

        # One strategy might be to try to create winning votes for a value
        # that the correct processes aren't 
        #least_popular = min(state.procs)
        #T = coinflip(least_popular)

        # Or we could try to send confusing messages to correct procs
        # send_message(T, least_popular)

        # Also, at any time, we can trigger the delivery of any message in a 
        # process's mailbox. However, even if we never trigger a message
        # delivery, the simulator will automatically deliver each message
        # after a maximum of D rounds.
        # proc = iter(state.ready_procs).next()
        # msg = iter(proc.mailbox).next()
        # let_process_receive(proc, msg)

        # Finally, we get to decide the order in which the processes run. Each
        # process gets to run once in one round, so we can only choose from the
        # set of ready processes. Note that it's not satisfactory to simply
        # refuse to run any processes. The adversary only 'wins' if the round 
        # counter keeps incrementing.

        # The best strategy is to wait until the correct processes win 
        # a coinflip. We then flip as many coins as it takes until we 
        # find a coin flip for the opposite value. If we're successful,
        # then we can deliver our opposite value first to half of the 
        # processes, and the original value to the other half.

        # Sample strategy (just flip all our coins for a constant value)
        while coins:
            T = coins.pop()(0)
            # Store statistics
            if T: adversary_successes += 1

        # Just let the next process run
        proc = iter(state.ready_procs).next()
        let_process_run(proc)


class Process(object):
    def __init__(self, proposed):
        self.r = 0                            # Keep a round counter
        self.V = defaultdict(lambda: set())   # Store all the votes we receive
        self.V[proposed]                      # Initially prefer our input
        self.r_hat = r_hat                    # Decide after round r_hat

    def receive(self, T, v):
        # Tally up all the valid votes we learn about
        #print 'Process %s received %s' % (self, (T, v))
        if valid(T, v): self.V[v].add(T)

    def compute(self, coinflip, broadcast, decide):

        # Always prefer the value with the most votes (we know of)
        Tr = max(map(len, self.V.values()))
        prefer = filter(lambda v: len(self.V[v]) >= Tr, self.V)[0]

        # Coinflip with Pr[success]: p
        #print '- I prefer %s' % (prefer,)
        T = coinflip(prefer)
        if valid(T, prefer):
            #print 'Wow, I won!'
            broadcast(T, prefer)

        # Decision condition
        if self.r >= self.r_hat:
            decide(prefer)
        self.r += 1


def Simulate(adversary=Adversary):

    # Simulation state
    _r = [0]
    procs = [Process(random.randint(0,1)) for _ in range(N-f)]
    mailboxes = defaultdict(lambda: set())
    decisions = defaultdict(lambda: None)
    ready_procs = set(procs)
    adversary_coins = set(make_coin() for _ in range(f))

    # Some stats collection
    global R_C, R_A
    R_C = defaultdict(None)
    R_A = defaultdict(None)
    X_C = defaultdict(None)
    X_A = defaultdict(None)
        

    def broadcast(T, v): 
        # Send a message to all processes at once
        for p in procs: send_message(p, T, v)

    def send_message(p, T, v):
        # Send a message to a single process
        # Each item in the mailbox is keyed with its delay timeout
        timeout = _r[0] + D
        mailboxes[p].add((timeout, T, v))
        #print 'Sending a message to process %s: %s (by round %d)' % \
        #    (procs.index(p), (hash(T),v), timeout)


    """ These are capabilities that we hand to the scheduling adversary
    """

    def let_process_run(p):
        # Make sure the processes get run once each round
        assert p in ready_procs
        ready_procs.remove(p)

        # Immediately deliver any late messages
        late = filter(lambda (timeout, T, v): timeout <= _r[0], mailboxes[p])
        for timeout, T, v in late: 
            mailboxes[p].remove((timeout, T, v))
            p.receive(T, v)

        # Let the process run one turn
        def decide(v): 
            if decisions[p] is None: 
                print 'Process %s decided on %s' % (procs.index(p),v)
                decisions[p] = v

        #print "Process %s computing" % (procs.index(p),)
        p.compute(make_coin(), broadcast, decide)

        # Proceed to the next round after all the processes have run
        if not ready_procs: next_round()


    def let_process_receive(p, msg):
        assert p in procs
        assert msg in mailboxes[p]
        mailboxes[p].remove(msg)
        p.receive(msg)


    """Internal simulation functions
    """

    def next_round():
        # Collect some statistics
        r, = _r
        X_C[r] = min(max(len(v) for v in p.V.values()) for p in procs)
        total = set()
        for p in procs: 
            for v in p.V.values(): total.update(v)
        X_A[r] = adversary_successes + len(total)
        if X_C[r] not in R_C: R_C[X_C[r]] = r
        if X_A[r] not in R_A: R_A[X_A[r]] = r

        # Quit after all processes have decided
        if all(decisions[p] is not None for p in procs): return finished()

        _r[0] += 1
        #print "Next round: ", _r[0]

        # All the processes will get to run again
        ready_procs.update(procs)

        # Give the adversary some freshly minted coins to flip at his leisure
        adversary_coins.update(make_coin() for _ in range(f))

    def finished():
        if len(unique(decisions.values())) == 1:
            print 'Consensus!'
        else:
            print 'Disagreement!!!'


    # This is a very leaky sham of a 'proxy'
    sim_locals = locals()
    class ReadOnlyState(object):
        def __getattr__(self, name):
            return sim_locals[name]
    state = ReadOnlyState()

    # Let the adversary take over from here
    adversary(adversary_coins, state, send_message,
              let_process_run, let_process_receive)


"""This implementation of coinflips is a bit indulgent, but I'm trying out
   a style of 'objcap' programming. Each coin is a 'one time use' abstraction.
   There's no other way to create a 'success' result except by flipping one of
   the coins.
"""

def make_mint(p):
    winners = defaultdict(lambda: set())
    class Token(): pass
    valid = lambda T, v: T in winners[v]

    def make_coin():
        v = yield
        if random.random() <= p: 
            T = Token()
            winners[v].add(T)
            yield T
        else: yield None

    return lambda: (lambda x: x.next() or x.send)(make_coin()), valid
make_coin, valid = make_mint(p)


def once():
    # Run the simulator, make some graphs
    Simulate()
    x = np.arange(max(R_C.keys()))
    plot(x, map(R_C.get, x), 'b')
    plot(x, map(R_A.get, x*2), 'r')
    title('R^C_x (blue) vs R^A_2x (red)')
    xlabel('x (number of votes)')
    ylabel('r (rounds taken to find x votes)')
