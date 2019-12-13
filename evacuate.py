#!/usr/bin/env python3

'''
This is the main file in the evacuation simulation project.
people: Nick B., Matthew J., Aalok S.

In this file we define a useful class to model the building floor, 'Floor'

Also in this file, we proceed to provide a main method so that this file is
meaningfully callable to run a simulation experiment
'''

# stdlib imports
import simulus 
import sys
import pickle
import random
import pprint
# from floorplan.floorplan import FloorGUI
from argparse import ArgumentParser
try:
    from randomgen import PCG64, RandomGenerator as Generator
except ImportError:
    from randomgen import PCG64, Generator

# local project imports
from person import Person
from bottleneck import Bottleneck
from floorparse import FloorParser

pp = pprint.PrettyPrinter(indent=4).pprint

class Floor:
    sim = None
    graph = None
    gui = None
    r = None
    c = None

    numpeople = 0
    numdead = 0
    numsafe = 0
    nummoving = 0


    bottlenecks = dict()#[]
    fires = set()
    people = []

    exit_times = []
    avg_exit = 0 # tracks sum first, then we /

    def __init__(self, input, n, location_sampler=random.sample,
                 strategy_generator=lambda: random.uniform(.5, 1.),
                 rate_generator=lambda: abs(random.normalvariate(1, .5)),
                 person_mover=random.uniform, fire_mover=random.sample):
        '''
        constructor method
        ---
        graph (dict): a representation of the floor plan as per our
                      specification
        n (int): number of people in the simulation
        '''
        self.sim = simulus.simulator()
        self.parser = FloorParser()
        with open(input, 'r') as f:
            self.graph = self.parser.parse(f.read())
        self.numpeople = n

        self.location_sampler = location_sampler
        self.strategy_generator = strategy_generator
        self.rate_generator = rate_generator
        self.person_mover = person_mover
        self.fire_mover = fire_mover

        self.setup()


    def precompute(self):
        '''
        precompute stats on the graph, e.g. nearest safe zone, nearest fire
        '''
        graph = self.graph

        def bfs(target, pos): # iterative dfs
            if graph[pos]['W']: return float('inf')
            q = [(pos, 0)]
            visited = set()
            while q:
                node, dist = q.pop()
                if node in visited: continue
                visited.add(node)

                node = graph[node]
                if node['W'] or node['F']: continue
                if node[target]: return dist

                for n in node['nbrs']:
                    if n in visited: continue
                    q = [(n, dist+1)] + q

            return float('inf')
                
        #for i in range(self.r):
        #    for j in range(self.c):
        for loc in graph:
            graph[loc]['distF'] = bfs('F', loc) 
            graph[loc]['distS'] = bfs('S', loc)

        self.graph = dict(graph.items())

        return self.graph


    def setup(self):
        '''
        once we have the parameters and random variate generation methods from
        __init__, we can proceed to create instances of: people and bottlenecks
        '''
        self.precompute()
        
        av_locs = []
        bottleneck_locs = []
        fire_locs = []
        
        r, c = 0, 0
        for loc, attrs in self.graph.items():
            r = max(r, loc[0])
            c = max(c, loc[1])
            if attrs['P']: av_locs += [loc] 
            elif attrs['B']: bottleneck_locs += [loc]
            elif attrs['F']: fire_locs += [loc]

        assert len(av_locs) > 0, 'ERR: no people placement locations in input'
        for i in range(self.numpeople):
            p = Person(self.rate_generator(),
                       self.strategy_generator(),
                       self.location_sampler(av_locs))
            self.people += [p]

        for loc in bottleneck_locs:
            b = Bottleneck(loc)            
            self.bottlenecks[loc] = b
        self.fires.update(set(fire_locs))
        
        self.r, self.c = r+1, c+1
        
        print(
              '='*79,
              'initialized a {}x{} floor with {} people in {} locations'.format(
                    self.r, self.c, len(self.people), len(av_locs)
                  ),
              'initialized {} bottleneck(s)'.format(len(self.bottlenecks)),
              'detected {} fire zone(s)'.format(len([loc for loc in self.graph 
                                                     if self.graph[loc]['F']])),
              '\ngood luck escaping!', '='*79, 'LOGS', sep='\n'
              
             )


    def visualize(self, t=10000):
        '''
        '''
        try:
            from floorplan.floorplan import FloorGUI
            self.gui = FloorGUI(self.r, self.c)
            self.gui.setup()
            self.gui.window.Read(timeout=0)
            self.gui.load(self.graph)
            print('displaying for {}s. click X to close earlier.'.format(t/1e3))
            self.gui.window.Read(timeout=t)
        except ImportError:
            print('ERR: make sure you have the floorplan module containing '
                  'the FloorGUI class and try again')



    def update_bottlenecks(self):
        '''
        handles the bottleneck zones on the grid, where people cannot all pass
        at once. for simplicity, bottlenecks are treated as queues
        '''
        raise NotImplementedError


    def update_fire(self):
        '''
        docstring: TODO
        '''
        no_fire_nbrs = [] # list, not set because more neighbors = more likely
        for loc in self.fires:
            # gets the square at the computed location
            square = self.graph[loc]

            # returns the full list of nbrs of the square
            nbrs = [(coords, self.graph[coords]) for coords in square['nbrs']]

            # updates nbrs to exclude safe zones and spaces already on fire 
            no_fire_nbrs += [(loc, attrs) for loc, attrs in nbrs 
                             if attrs['S'] == attrs['F'] == 0]
            # more likely (twice) to spread to non-wall empty zone
            no_fire_nbrs += [(loc, attrs) for loc, attrs in nbrs 
                             if attrs['W'] == attrs['S'] == attrs['F'] == 0]
        #randomly choose a neighbor 
        #upper = len(no_fire_nbrs)-1
        # = random.sample(no_fire_nbrs, 1)
        try:
            # TODO replace with a randomgen stream draw
            [(choice, _)] = random.sample(no_fire_nbrs, 1)
        except ValueError as e:
            if 'Sample larger than population' in e.args:
                return
            else: 
                raise

        self.graph[choice]['F'] = 1
        self.fires.add(choice)

        self.precompute()
        self.sim.sched(self.update_fire, 
                       offset=len(self.graph)/max(1, len(self.fires))**2)

        return choice


    def update_person(self, person_ix):
        '''
        handles scheduling an update for each person, by calling move() on them.
        move will return a location decided by the person, and this method will
        handle the simulus scheduling part to keep it clean
        '''
        if self.maxtime and self.sim.now >= self.maxtime:
            return

        p = self.people[person_ix]
        if not p.alive:
            self.numdead += 1
            print('Person at {} is now DED'.format(p.loc))
            return
        if p.safe:
            self.numsafe += 1
            p.exit_time = self.sim.now
            self.exit_times += [p.exit_time]
            self.avg_exit += p.exit_time
            # print('Person at {} is now SAFE'.format(p.loc))
            return

        loc = p.loc
        square = self.graph[loc]
        nbrs = [(coords, self.graph[coords]) for coords in square['nbrs']]
        
        target = p.move(nbrs)
        square = self.graph[target]
        if square['B']:
            b = self.bottlenecks[target]
            b.enterBottleNeck(p)
        elif square['F']:
            p.alive = False
            return
        else:
            t = 1/p.rate
            if self.sim.now + t >= (self.maxtime or float('inf')):
                if square['S']: 
                    self.nummoving += 1
                else:
                    self.numdead += 1
            else:
                self.sim.sched(self.update_person, person_ix, offset=1/p.rate)
        
        # self.sim.show_calendar()


    def simulate(self, maxtime=None, spread_fire=False):
        '''
        sets up initial scheduling and calls the sim.run() method in simulus
        '''
        # set initial movements of all the people
        for i, p in enumerate(self.people):
            loc = tuple(p.loc)
            square = self.graph[loc]
            nbrs = square['nbrs']
            self.sim.sched(self.update_person, i, offset=1/p.rate)

        #updates fire initially
        if spread_fire:
            self.sim.sched(self.update_fire, 
                           offset=1)#len(self.graph)/max(1, len(self.fires)))
        else:
            print('INFO\t', 'fire won\'t spread around!')
       
        self.maxtime = maxtime
        self.sim.run()
   
        self.avg_exit /= max(self.numsafe, 1)


    def stats(self):
        '''
        '''
        print('\n\n', '='*79, sep='')
        print('STATS')

        def printstats(desc, obj):
            print('\t', 
                  (desc+' ').ljust(30, '.') + (' '+str(obj)).rjust(30, '.'))

        printstats('total # people', self.numpeople)
        printstats('# people safe', self.numsafe)
        printstats('# people dead', self.numpeople -self.numsafe-self.nummoving)
        printstats('# people gravely injured', self.nummoving)
        print()
        printstats('total simulation time', '{:.3f}'.format(self.sim.now))
        printstats('average time to safe', '{:.3f}'.format(self.avg_exit))

        print()


def main():
    '''
    driver method for this file
    '''
    # set up and parse commandline arguments
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='in/singleexit.txt',
                        help='input floor plan file (default:floor.txt.pkl)')
    parser.add_argument('-n', '--numpeople', type=int, default=10,
                        help='number of people in the simulation (default:10)')
    parser.add_argument('-s', '--random_state', type=int, default=8675309,
                        help='aka. seed (default:8675309)')
    parser.add_argument('-t', '--max_time', type=float, default=None,
                        help='the building collapses at this clock tick. people'
                             ' beginning movement before this will be assumed'
                             ' to have moved away sufficiently (safe)')
    parser.add_argument('-f', '--no_spread_fire', action='store_true',
                        help='disallow fire to spread around?')
    args = parser.parse_args()
    # output them as a make-sure-this-is-what-you-meant
    print('commandline arguments:', args, '\n')


    # load the graph representation of some floor plan
    #with open(args.input, 'rb') as f:
    #    graph = pickle.load(f)

    # set up random streams
    streams = [Generator(PCG64(args.random_state, i)) for i in range(5)]
    loc_strm, strat_strm, rate_strm, pax_strm, fire_strm = streams
    
    location_sampler = loc_strm.choice # used to make initial placement of pax
    strategy_generator = lambda: strat_strm.uniform(.5, 1) # used to pick move
    rate_generator = lambda: max(.1, abs(rate_strm.normal(1, .1))) # used to
                                                                   # decide
                                                                   # strategies
    person_mover = lambda: pax_strm.uniform() # 
    fire_mover = lambda: fire_strm.uniform() # 

    # create an instance of Floor
    floor = Floor(args.input, args.numpeople, location_sampler, 
                  strategy_generator, rate_generator, person_mover, fire_mover)

    # floor.visualize(t=5000)
    # call the simulate method to run the actual simulation
    floor.simulate(maxtime=args.max_time, spread_fire=not args.no_spread_fire) 
    
    floor.stats() 

if __name__ == '__main__':
    main()
