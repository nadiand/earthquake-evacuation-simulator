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
from argparse import ArgumentParser
try:
    from randomgen import PCG64, RandomGenerator as Generator
except ImportError:
    from randomgen import PCG64, Generator

# local project imports
from person import Person
from bottleneck import Bottleneck
from floorparse import FloorParser
import numpy as np

pp = pprint.PrettyPrinter(indent=4).pprint

class FireSim:
    sim = None
    graph = None # dictionary (x,y) --> attributes
    gui = False
    r = None
    c = None

    numpeople = 0
    numdead = 0
    numsafe = 0
    nummoving = 0

    bottlenecks = dict()
    fires = set()
    graves = set()
    risky = set()
    people = []

    exit_times = []
    avg_exit = 0 # tracks sum first, then we divide

    def __init__(self, input, n, location_sampler=random.sample,
                 strategy_generator=lambda: random.uniform(.5, 1.),
                 rate_generator=lambda: abs(random.normalvariate(1, .5)),
                 person_mover=random.uniform, fire_mover=random.sample,
                 fire_rate=2, bottleneck_delay=1, animation_delay=.1,
                 verbose=False,
                 **kwargs):
        '''
        constructor method
        ---
        graph (dict): a representation of the floor plan as per our
                      specification
        n (int): number of people in the simulation
        '''
        self.sim = simulus.simulator()
        self.parser = FloorParser() 
        self.animation_delay = animation_delay
        self.verbose = verbose

        with open(input, 'r') as f:
            self.graph = self.parser.parse(f.read())
        self.numpeople = n

        self.location_sampler = location_sampler
        self.strategy_generator = strategy_generator
        self.rate_generator = rate_generator
        self.person_mover = person_mover
        self.fire_mover = fire_mover
        
        self.fire_rate = fire_rate
        self.bottleneck_delay = bottleneck_delay
        self.kwargs = kwargs

        self.setup()


    def precompute(self):
        '''
        precompute stats on the graph, e.g. nearest safe zone, nearest fire
        '''
        graph = self.graph

        def bfs(target, pos):
            if graph[pos]['W'] or graph[pos]['G']: return float('inf')
            q = [(pos, 0)]
            visited = set()
            while q:
                node, dist = q.pop()
                if node in visited: continue
                visited.add(node)

                #TODO: figure out how to take dangerous tiles into account for pathfinding
                node = graph[node]
                if node['W'] or node['G']: continue
                if node[target]: return dist

                for n in node['nbrs']:
                    if n in visited: continue
                    q = [(n, dist+1)] + q

            # unreachable
            return float('inf')

        # for each location, we do breath first search to find the nearest safe zone (distS) and the nearest fire zone (distF).
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

        
        bottleneck_locs = []
        fire_locs = []
        risky_locs = []

        # get lists of fire locations, bottleneck locations and people
        r, c = 0, 0
        for loc, attrs in self.graph.items():
            r = max(r, loc[0])
            c = max(c, loc[1])
            
            if attrs['B']: bottleneck_locs += [loc]
            elif attrs['F']: fire_locs += [loc]
            elif attrs['R']: risky_locs += [loc]


        
        # initialise all people
        for i in range(self.numpeople):
            loc = random.randint(0, r-1), random.randint(0, c-1)
            while loc not in self.graph or self.graph[loc]['W'] == 1 or self.graph[loc]['S'] == 1:
                # sample a random location that is not a wall nor a safe location
                loc = random.randint(0, r-1), random.randint(0, c-1)

            # initilase boldness
            scaredness = random.randint(0,1)

            p = Person(i, self.rate_generator(),
                       self.strategy_generator(),
                       loc, scaredness=scaredness)
            self.people += [p]

        # initialise bottlenecks
        for loc in bottleneck_locs:
            b = Bottleneck(loc)
            self.bottlenecks[loc] = b

        # update the fire locations
        self.fires.update(set(fire_locs))
        self.risky.update(set(risky_locs))

        self.r, self.c = r+1, c+1

        print(
              '='*79,
              'initialized a {}x{} floor with {} people'.format(
                    self.r, self.c, len(self.people)
                  ),
              'initialized {} bottleneck(s)'.format(len(self.bottlenecks)),
              'detected {} fire zone(s)'.format(len([loc for loc in self.graph
                                                     if self.graph[loc]['F']])),
              '\ngood luck escaping!', '='*79, 'LOGS', sep='\n'
             )


    def visualize(self, t):
        '''
        '''
        if self.gui:
            self.plotter.visualize(self.graph, self.people, t)


    def update_bottlenecks(self):
        '''
        handles the bottleneck zones on the grid, where people cannot all pass
        at once. for simplicity, bottlenecks are treated as queues
        '''

        for key in self.bottlenecks:
            personLeaving = self.bottlenecks[key].exitBottleNeck()
            if(personLeaving != None):
                self.sim.sched(self.update_person, personLeaving.id, offset=0)

        # stop if the simulation is over
        if self.numsafe + self.numdead >= self.numpeople:
            return

        # check if the simulation is overtime, if not, update the bottlenecks
        if self.maxtime and self.sim.now >= self.maxtime:
            return
        else:
            self.sim.sched(self.update_bottlenecks, 
                           offset=self.bottleneck_delay)

    def update_grave(self):
        '''
        Makes G (grave) locations randomly appear
        '''

        if len(self.risky) > 0 and np.random.uniform(0,1) < 0.3:
            loc = random.sample(self.risky, 1)[0]
            randcol = loc[0]
            randrow = loc[1]
        else:
            randcol = np.random.randint(0, self.c)
            randrow = np.random.randint(0, self.r)
            

        # set the random square to grave and set all other values to false
        self.graph[(randrow, randcol)].update({'G': True})
        self.graves.add(((randrow, randcol), self.sim.now))
        self.graph[(randrow, randcol)].update({'R': False, 'D': False, 'N': False})

        # turn all neigbours to risky
        for neighbour in self.graph[(randrow, randcol)]['nbrs']:
            if self.graph[neighbour]['R']:
                self.graph[neighbour].update({'D': True})
            elif self.graph[neighbour]['W']:
                self.graph[neighbour].update({'R': True})
            elif self.graph[neighbour]['G']:
                pass
            else:
                self.graph[neighbour].update({'R': True})
    
    def update(self):
        if self.numsafe + self.numdead >= self.numpeople:
            print('INFO:', 'people no longer moving, so stopping updating the debris')
            return
        if self.maxtime and self.sim.now >= self.maxtime:
            return

        # chance of 0.1 for a grave to form
        if np.random.uniform(0,1) < 0.7:
            self.update_grave()

        # update graves turning into damaged
        new_graves = set()
        for loc, time in self.graves:
            if (time + 2) <= self.sim.now:
                self.graph[loc].update({'D': True, 'G': False})
            else:
                new_graves.add((loc, time))
        self.graves = new_graves

        # risky cells become damaged with probability
        if np.random.uniform(0,1) < 0.6 and len(self.risky) > 0:
            loc = random.sample(self.risky, 1)[0]
            self.graph[loc].update({'D': True, 'R': False})

        self.precompute()
        rt = self.fire_rate
        self.sim.sched(self.update,
                       offset=len(self.graph)/max(1, len(self.risky))**rt)

        self.visualize(self.animation_delay/max(1, len(self.risky))**rt)



    def update_fire(self):
        '''
        method that controls the spread of fire. we use a rudimentary real-world
        model that spreads fire exponentially faster proportional to the amount
        of fire already on the floor. empty nodes are more likely to get set
        on fire the more lit neighbors they have. empty plain nodes are more
        likely to burn than walls.
        fire spreads proportional to (grid_area)/(fire_area)**exp
        the method uses the 'fire_rate' instance variable as the 'exp' in the
        expression above.
        fire stops spreading once it's everywhere, or when all the people have
        stopped moving (when all are dead or safe, not moving)
        '''

        if self.numsafe + self.numdead >= self.numpeople:
            print('INFO:', 'people no longer moving, so stopping fire spread')
            return
        if self.maxtime and self.sim.now >= self.maxtime:
            return

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

        try:
            (choice, _) = self.fire_mover(no_fire_nbrs)
        except ValueError as e:
            print('INFO:', 'fire is everywhere, so stopping fire spread')
            return

        self.graph[choice]['F'] = 1
        self.fires.add(choice)

        # self.precompute()
        # rt = self.fire_rate
        # self.sim.sched(self.update_fire,
        #                offset=len(self.graph)/max(1, len(self.fires))**rt)

        # self.visualize(self.animation_delay/max(1, len(self.fires))**rt)

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

        # check if the person is safe
        if self.graph[p.loc]['S']:
            p.safe = True
    
        # when a grave appears on top of a person, then count them as dead
        if self.graph[p.loc]['G'] or not p.alive:
            p.alive = False
            p.rate = 0
            self.numdead += 1
            if self.verbose:
                print('{:>6.2f}\tPerson {:>3} at {} died from falling debris'.format(
                                                                  self.sim.now,
                                                                  p.id, p.loc))
            return
        
        # when a person is in risky cell, reduce rate by 5%
        if self.graph[p.loc]['R']:
            p.rate *= 0.95

        # when a person is in damaged cell, reduce rate by 20%
        if self.graph[p.loc]['D']:
            p.rate *= 0.8

        # if rate drops below 0.6 they are considered as injured
        if p.rate < 0.8:
            p.injured = True

        # when a persons rate drops below 0.4 they die
        if p.rate < 0.4:
            p.alive = False
            self.numdead += 1
            if self.verbose:
                print('{:>6.2f}\tPerson {:>3} at {} died due to injury'.format(
                                                                  self.sim.now,
                                                                  p.id, p.loc))
            return
        
        # check if the person made it out safely, if so, update some stats
        if p.safe:
            self.numsafe += 1
            p.exit_time = self.sim.now
            self.exit_times += [p.exit_time]
            self.avg_exit += p.exit_time
            if self.verbose:
                print('{:>6.2f}\tPerson {:>3} is now SAFE!'.format(self.sim.now, 
                                                               p.id))
            return

        loc = p.loc
        square = self.graph[loc]
        nbrs = [(coords, self.graph[coords]) for coords in square['nbrs']]
        target = p.move(nbrs)

        # if there is no target location, then consider the person dead
        if not target:
            p.alive = False
            self.numdead += 1
            if self.verbose:
                print('{:>6.2f}\tPerson {:>3} at {} got trapped in fire'.format(
                                                                   self.sim.now,
                                                                   p.id, p.loc))
            return
        
        # get the target square, and handle walking, going into a bottleneck, or going into fire.
        square = self.graph[target]
        if square['B']:
            b = self.bottlenecks[target]
            b.enterBottleNeck(p)
        elif square['F']:
            p.alive = False
            self.numdead += 1
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

        if (1+person_ix) % int(self.numpeople**.5) == 0:
            self.visualize(t=self.animation_delay/len(self.people)/2)

        # self.sim.show_calendar()

    

    def simulate(self, maxtime=None, spread_fire=False, gui=False):
        '''
        sets up initial scheduling and calls the sim.run() method in simulus
        '''
        self.gui = gui
        if self.gui: 
            from viz import Plotter
            self.plotter = Plotter()

        # set initial movements of all the people
        for i, p in enumerate(self.people):
            loc = tuple(p.loc)
            square = self.graph[loc]
            nbrs = square['nbrs']
            self.sim.sched(self.update_person, i, offset=1/p.rate)

        #updates fire initially
        if spread_fire:
            self.sim.sched(self.update,
                           offset=1)
            # self.sim.sched(self.update_fire,
            #                offset=1) #len(self.graph)/max(1, len(self.fires)))
        else:
            print('INFO\t', 'fire won\'t spread around!')
        self.sim.sched(self.update_bottlenecks, offset=self.bottleneck_delay)

        self.maxtime = maxtime
        self.sim.run()

        self.avg_exit /= max(self.numsafe, 1)


    def stats(self):
        '''
        computes and outputs useful stats about the simulation for nice output
        '''
        print('\n\n', '='*79, sep='')
        print('STATS')

        def printstats(desc, obj):
            print('\t',
                  (desc+' ').ljust(30, '.') + (' '+str(obj)).rjust(30, '.'))

        # find how many people are injured
        numinjured = len([1 for i in self.people if i.injured])
        numdead = self.numpeople-self.numsafe-self.nummoving
        numdiedinjury = len([1 for i in self.people if (not i.alive) and i.rate > 0])
        numdied = len([1 for i in self.people if (not i.alive) and i.rate == 0])
        avg_rate = sum([p.rate for p in self.people if p.alive])/len([p.rate for p in self.people if p.alive])

        printstats('total # people', self.numpeople)
        print()
        printstats('# people safe', self.numsafe)
        printstats('# people gravely injured', numinjured-numdead)
        printstats('average rate of survivors', avg_rate)

        print()

        printstats('# people dead', numdead)
        printstats('# people died from injuries', numdiedinjury)
        printstats('# people died from falling debris', numdied)
        print()
        
        # printstats('total simulation time', '{:.3f}'.format(self.sim.now))
        if self.avg_exit:
            printstats('average time to safe', '{:.3f}'.format(self.avg_exit))
        else:
            printstats('average time to safe', 'NA')
        print()
        print("Id\tsafe\tinjured\trate\tstrat\tscaredness")
        for p in self.people:
            print(p.id, "\t", round(p.exit_time, 2), "\t", p.injured, "\t", round(p.rate, 2), "\t", round(p.strategy, 2), "\t", p.scaredness)

        # print(self.parser.tostr(self.graph))
        self.visualize(4)


def main():
    '''
    driver method for this file. the firesim class can be used via imports as
    well, but this driver file provides a comprehensive standalone interface
    to the simulation
    '''
    # set up and parse commandline arguments
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='in/twoexitbottleneck.txt',
                        help='input floor plan file (default: '
                             'in/twoexitbottleneck.py)')
    parser.add_argument('-n', '--numpeople', type=int, default=10,
                        help='number of people in the simulation (default:10)')
    parser.add_argument('-r', '--random_state', type=int, default=8675309,
                        help='aka. seed (default:8675309)')
    parser.add_argument('-t', '--max_time', type=float, default=None,
                        help='the building collapses at this clock tick. people'
                             ' beginning movement before this will be assumed'
                             ' to have moved away sufficiently (safe)')
    parser.add_argument('-f', '--no_spread_fire', action='store_true',
                        help='disallow fire to spread around?')
    parser.add_argument('-g', '--no_graphical_output', action='store_true',
                        help='disallow graphics?')
    parser.add_argument('-o', '--output', action='store_true',
                        help='show excessive output?')
    parser.add_argument('-d', '--fire_rate', type=float, default=2,
                        help='rate of spread of fire (this is the exponent)')
    parser.add_argument('-b', '--bottleneck_delay', type=float, default=1,
                        help='how long until the next person may leave the B')
    parser.add_argument('-a', '--animation_delay', type=float, default=1,
                        help='delay per frame of animated visualization (s)')
    args = parser.parse_args()
    # output them as a make-sure-this-is-what-you-meant
    print('commandline arguments:', args, '\n')

    # set up random streams
    streams = [np.random.Generator(PCG64(args.random_state, i)) for i in range(5)]
    loc_strm, strat_strm, rate_strm, pax_strm, fire_strm = streams

    location_sampler = loc_strm.choice # used to make initial placement of pax
    strategy_generator = lambda: strat_strm.uniform(.5, 1) # used to pick move
    rate_generator = lambda: rate_strm.uniform(.9, 1.1)
    person_mover = lambda: pax_strm.uniform() #
    fire_mover = lambda a: fire_strm.choice(a) #

    # create an instance of Floor
    floor = FireSim(args.input, args.numpeople, location_sampler,
                    strategy_generator, rate_generator, person_mover,
                    fire_mover, fire_rate=args.fire_rate,
                    bottleneck_delay=args.bottleneck_delay,
                    animation_delay=args.animation_delay, verbose=args.output)

    # floor.visualize(t=5000)
    # call the simulate method to run the actual simulation
    floor.simulate(maxtime=args.max_time, spread_fire=not args.no_spread_fire,
                   gui=not args.no_graphical_output)

    floor.stats()

if __name__ == '__main__':
    main()
