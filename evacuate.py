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
import random
import pprint
from argparse import ArgumentParser
from randomgen import PCG64

# local project imports
from person import Person
from bottleneck import Bottleneck
from floorparse import FloorParser
import numpy as np

pp = pprint.PrettyPrinter(indent=4).pprint

class EarthquakeSim:
    sim = None
    graph = None # dictionary (x,y) --> attributes
    gui = False
    r = None
    c = None

    def __init__(self, input, n, location_sampler=random.sample,
                 strategy_generator=lambda: random.uniform(.5, 1.),
                 rate_generator=lambda: abs(random.normalvariate(1, .5)),
                 person_mover=random.uniform,
                 damage_rate=2, bottleneck_delay=1, animation_delay=.1,
                 verbose=False, scaredness_rate=0.5, follower_rate=0.2, 
                 run_id=0,
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
        self.numwaiting = 0 
        self.run_id = run_id

        with open(input, 'r') as f:
            self.graph = self.parser.parse(f.read())
        self.numpeople = n

        self.fov = {}
        self.location_sampler = location_sampler
        self.strategy_generator = strategy_generator
        self.rate_generator = rate_generator
        self.person_mover = person_mover
        
        self.damage_rate = damage_rate
        self.bottleneck_delay = bottleneck_delay
        self.kwargs = kwargs

        self.gui = False
        self.r = None
        self.c = None

        self.people = []
        self.bottlenecks = dict()
        self.graves = set()
        self.risky = set()

        self.numdead = 0
        self.numsafe = 0
        self.nummoving = 0

        self.exit_times = []
        self.avg_exit = 0

        self.scaredness_rate = scaredness_rate
        self.follower_rate = follower_rate

        self.setup()


    def bfs(self, target, pos):
        if self.graph[pos]['W'] or self.graph[pos]['G']: return float('inf')
        if self.graph[pos]['S']: return 0.0
        q = [(pos, 0)]
        visited = set()
        while q:
            node, dist = q.pop()
            if node in visited: continue
            visited.add(node)

            node = self.graph[node]
            if node['W'] or node['G']: continue
            if node[target]: return dist

            for n in node['nbrs']:
                if n in visited: continue
                q = [(n, dist+1)] + q

        # unreachable
        return float('inf')

    def precompute(self):
        '''
        precompute stats on the graph, e.g. nearest safe zone, nearest bottleneck
        '''
        
        # for each location, we do breath first search to find the nearest safe zone (distS) and the nearest bottleneck (distB).
        for loc in self.graph:
            self.graph[loc]['distS'] = self.bfs('S', loc)
            self.graph[loc]['distB'] = self.bfs('B', loc)

        self.graph = dict(self.graph.items())

        return self.graph


    def setup(self):
        '''
        once we have the parameters and random variate generation methods from
        __init__, we can proceed to create instances of: people and bottlenecks
        '''
        self.precompute()
        
        bottleneck_locs = []
        risky_locs = []
        r, c = 0, 0

        # get lists of bottleneck locations and risky locations
        for loc, attrs in self.graph.items():
            r = max(r, loc[0])
            c = max(c, loc[1])
            
            if attrs['B']: bottleneck_locs += [loc]
            elif attrs['R']: risky_locs += [loc]

        self.risky.update(set(risky_locs))

        # initialise all people
        for i in range(self.numpeople):
            loc = random.randint(0, r-1), random.randint(0, c-1)
            while loc not in self.graph or self.graph[loc]['W'] == 1 or self.graph[loc]['S'] == 1:
                # sample a random location that is not a wall nor a safe location
                loc = random.randint(0, r-1), random.randint(0, c-1)

            # initilase scaredness and strategy
            scaredness = 0
            follower = 0
            if random.uniform(0,1) < self.scaredness_rate:
                scaredness = 1
            if random.uniform(0,1) < self.follower_rate:
                follower = 1

            p = Person(i, self.rate_generator(), loc,
                       strategy=follower,
                       scaredness=scaredness,
                       graph=self.graph)
            self.people += [p]

        # initialise bottlenecks
        for loc in bottleneck_locs:
            b = Bottleneck(loc)
            self.bottlenecks[loc] = b

        # get the dimensions of the graph
        dims = list(self.graph.keys())[-1]
        dims = np.subtract(dims, (-1,-1))

        # get locations of the walls for vield of view calculations
        walls = np.zeros(dims) 
        for loc in self.graph:
            if self.graph[loc]['W']:
                walls[loc] = 1
            else:
                walls[loc] = 0
        self.visibility(walls)

        self.r, self.c = r+1, c+1

        print(
              '='*79,
              'initialized a {}x{} floor with {} people'.format(
                    self.r, self.c, len(self.people)
                  ),
              'initialized {} bottleneck(s)'.format(len(self.bottlenecks)),
              '\ngood luck escaping!', '='*79, 'LOGS', sep='\n'
             )

    def visibility(self, walls):
        '''
        Calcualtes field of view for each cell in the graph.
        '''

        vision = 15
        n = 6
        for loc in self.graph:
            if not (self.graph[loc]['W'] or self.graph[loc]['S']):
                # initialise and add itself
                self.fov[loc] = []
                self.fov[loc].append(loc)

                # calculate boudaries of vision
                x_min = loc[0]-vision if loc[0] > vision else 0
                y_min = loc[1]-vision if loc[1] > vision else 0
                x_max = loc[0]+vision if loc[0] < len(walls) - vision else len(walls)
                y_max = loc[1]+vision if loc[1] < len(walls[0]) - vision else len(walls[0])

                # check if there is a wall in between, if not it is added to the fov
                for i in range(x_min, x_max):
                    for j in range(y_min, y_max):
                        dxy = (abs(i - loc[0]) + abs(j - loc[1])) * n
                        x = np.rint(np.linspace(i, loc[0], dxy)).astype(int)
                        y = np.rint(np.linspace(j, loc[1], dxy)).astype(int)
                        has_collision = np.any(walls[x, y])

                        if not has_collision:
                            self.fov[loc].append((i, j))            


    def visualize(self, t):
        if self.gui:
            self.plotter.visualize(self.graph, self.people, t)


    def update_bottlenecks(self):
        '''
        Handles the bottleneck zones on the grid, where people cannot all pass
        at once. For simplicity, bottlenecks are treated as queues
        '''

        for key in self.bottlenecks:
            personLeaving = self.bottlenecks[key].exitBottleNeck()
            if(personLeaving != None):
                self.sim.sched(self.update_person, personLeaving.id, offset=0)

        # check if the simulation is over, if not, update the bottlenecks
        if self.maxtime and self.sim.now >= self.maxtime:
            return
        elif self.numsafe + self.numwaiting + self.numdead >= self.numpeople:
            return
        else:
            self.sim.sched(self.update_bottlenecks, 
                           offset=self.bottleneck_delay)

    def update_grave(self):
        '''
        Makes G (grave) locations randomly appear.
        '''

        # take random row and column either from the risky cells or randomly
        if len(self.risky) > 0 and np.random.uniform(0,1) < 0.3:
            loc = random.sample(self.risky, 1)[0]
            randcol = loc[1]
            randrow = loc[0]
        else:
            randcol = np.random.randint(0, self.c)
            randrow = np.random.randint(0, self.r)
            
        # add the new grave cell to the list
        if (np.random.uniform(0,1) < 0.1):
            self.graves.add(((randrow, randcol), self.sim.now + float('inf')))
        else:
            self.graves.add(((randrow, randcol), self.sim.now))
        
        # set the random square to grave and set all other values to false
        self.graph[(randrow, randcol)].update({'G': True})
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
        '''
        Handles all updates regarding the environment.
        '''

        if self.numsafe + self.numwaiting + self.numdead >= self.numpeople:
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

        # only precompute again if the number of graves changes
        if self.graves != new_graves and self.sim.now > 10:
            self.precompute()
        self.graves = new_graves

        # risky cells become damaged with probability
        if np.random.uniform(0,1) < 0.6 and len(self.risky) > 0:
            loc = random.sample(self.risky, 1)[0]
            self.graph[loc].update({'D': True, 'R': False})

        # the offset is basically the more unstable the faster the damage spreads
        rt = self.damage_rate
        if (self.sim.now > 10): # after a time we add some additional time between updates
            self.sim.sched(self.update, offset=len(self.graph)/max(1, len(self.risky))**rt + 5)
        else:
            self.sim.sched(self.update, offset=len(self.graph)/max(1, len(self.risky))**rt)

        self.visualize(self.animation_delay/max(1, len(self.risky))**rt)
    

    def update_person(self, person_ix):
        '''
        Handles scheduling an update for each person, by calling move() on them.
        Move will return a location decided by the person, and this method will
        handle the simulus scheduling part to keep it clean.
        '''
        if self.maxtime and self.sim.now >= self.maxtime:
            for p in self.people:
                if p.alive and not p.safe:
                    p.waiting_for_rescue = True
                    self.numwaiting += 1
            return

        p = self.people[person_ix]

        # check if the person is safe
        if self.graph[p.loc]['S']:
            p.safe = True
    
        # when a grave appears on top of a person, then count them as dead
        if self.graph[p.loc]['G'] or not p.alive:
            p.alive = False
            p.exit_time = self.sim.now
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

        # when a person is in damaged cell, reduce rate by 15%
        if self.graph[p.loc]['D']:
            p.rate *= 0.85

        # if rate drops below 0.6 they are considered as injured
        if p.rate < 0.6:
            p.injured = True

        # when a persons rate drops below 0.4 they die
        if p.rate < 0.4:
            p.alive = False
            p.exit_time = self.sim.now
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

        # find the square with the most people for the followers
        max_people = 0
        loc_max_people = None        
        if p.strategy > self.follower_rate:
            for loc in self.fov[p.loc]:
                if not (loc[0] == p.loc[0] and loc[1] == p.loc[1]):
                    num_people = 0
                    for person in self.people:
                        if person.loc == loc and person.alive: # only add if alive
                            num_people += 1
                    if num_people > max_people:
                        max_people = num_people
                        loc_max_people = loc

        target = p.move(nbrs, self.fov[p.loc], loc_max_people, loc)

        # if there is no target location, then consider the person dead
        if not target:
            p.alive = False
            p.exit_time = self.sim.now
            self.numdead += 1
            if self.verbose:
                print('{:>6.2f}\tPerson {:>3} at {} got trapped'.format(
                                                                   self.sim.now,
                                                                   p.id, p.loc))
            return
        
        # get the target square, and handle walking or going into a bottleneck.
        square = self.graph[target]
        if square['B']:
            b = self.bottlenecks[target]
            b.enterBottleNeck(p)
        else:
            t = 1/p.rate
            if self.sim.now + t >= (self.maxtime or float('inf')):
                if square['S']:
                    self.numsafe += 1
                else:
                    self.numdead += 1
            else:
                num_peeps = sum([1 for peep in self.people if p.loc == peep.loc])
                # offset depends on how many people are in the square, to model pushing and obstacles of fallen peeps
                self.sim.sched(self.update_person, person_ix, offset=num_peeps/p.rate)

        if (1+person_ix) % int(self.numpeople**.5) == 0:
            self.visualize(t=self.animation_delay/len(self.people)/2)


    def simulate(self, maxtime=None, spread_damage=False, gui=False):
        '''
        sets up initial scheduling and calls the sim.run() method in simulus
        '''
        self.gui = gui
        if self.gui: 
            from viz import Plotter
            self.plotter = Plotter()

        # set initial movements of all the people
        for i, p in enumerate(self.people):
            self.sim.sched(self.update_person, i, offset=1/p.rate+10)

        # updates earthquake initially
        if spread_damage:
            self.sim.sched(self.update, offset=1)
        else:
            print('INFO\t', 'damage won\'t spread around!')

        # schedule initial bottlenecks
        self.sim.sched(self.update_bottlenecks, offset=self.bottleneck_delay)

        self.maxtime = maxtime

        # Termination condition: All people are either safe, dead, or waiting for rescue
        while self.numsafe + self.numwaiting + self.numdead < self.numpeople:
            self.sim.step()
        self.avg_exit /= max(self.numsafe, 1)

        return


    def stats(self):
        '''
        computes and outputs useful stats about the simulation for nice output
        '''
        print('\n\n', '='*79, sep='')
        print('STATS')

        def printstats(desc, obj):
            print('\t',
                  (desc+' ').ljust(30, '.') + (' '+str(obj)).rjust(30, '.'))

        # calculate some statistics
        numinjured = len([1 for i in self.people if i.injured])
        numdead = self.numpeople-self.numsafe-self.numwaiting
        numdiedinjury = len([1 for i in self.people if (not i.alive) and i.rate > 0])
        numdied = len([1 for i in self.people if (not i.alive) and i.rate == 0])
        avg_rate = sum([p.rate for p in self.people if p.alive])/len([p.rate for p in self.people if p.alive])

        # print some overall statistics
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

        # print table of individual statistics
        print('INDIVIDUALS')
        if self.avg_exit:
            printstats('average time to safe', '{:.3f}'.format(self.avg_exit))
        else:
            printstats('average time to safe', 'NA')
        print()
        print("Id\tsafe\tinjured\tstartR\trate\tstrat\tscaredness")

        # make string of results to return
        results = ""
        for p in self.people:
            print(self.run_id, "\t", p.id, "\t", round(p.exit_time, 2), "\t", p.injured, "\t", round(p.starting_rate, 2), "\t", round(p.rate, 2), "\t", round(p.strategy, 2), "\t", p.scaredness, "\t", p.waiting_for_rescue)
            results += str(self.run_id) + " " + str(p.id) + " " + str(p.alive) + " " + \
            str(round(p.exit_time, 3)) + " " + str(round(p.starting_rate, 3)) + " " + \
            str(round(p.rate, 3)) + " " + str(p.injured) + " " + str(round(p.strategy, 3)) + " " + \
            str(p.scaredness) + " " + str(p.waiting_for_rescue) + "\n"

        self.visualize(4)

        return results


def main(raw_args=None):
    '''
    driver method for this file. the EarthquakeSim class can be used via imports as
    well, but this driver file provides a comprehensive standalone interface
    to the simulation
    '''
    # set up and parse commandline arguments
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='in/twoexitbottleneck.txt',
                        help='input floor plan file (default: '
                             'in/twoexitbottleneck.py)')
    parser.add_argument('-n', '--numpeople', type=int, default=30,
                        help='number of people in the simulation (default:10)')
    parser.add_argument('-r', '--random_state', type=int, default=8675309,
                        help='aka. seed (default:8675309)')
    parser.add_argument('-t', '--max_time', type=float, default=None,
                        help='the building collapses at this clock tick. people'
                             ' beginning movement before this will be assumed'
                             ' to have moved away sufficiently (safe)')
    parser.add_argument('-f', '--no_spread_damage', action='store_true',
                        help='disallow damage to spread around?')
    parser.add_argument('-g', '--no_graphical_output', action='store_true',
                        help='disallow graphics?')
    parser.add_argument('-o', '--output', action='store_true',
                        help='show excessive output?')
    parser.add_argument('-d', '--damage_rate', type=float, default=2,
                        help='rate of spread of damage (this is the exponent)')
    parser.add_argument('-b', '--bottleneck_delay', type=float, default=1,
                        help='how long until the next person may leave the B')
    parser.add_argument('-a', '--animation_delay', type=float, default=1,
                        help='delay per frame of animated visualization (s)')
    parser.add_argument('-S', '--scaredness_rate', type=float, default=0.5,
                        help='number of people who are scared to walk through danger')
    parser.add_argument('-F', '--follower_rate', type=float, default=0.2,
                        help='number of people who have the follower strategy')
    parser.add_argument('-I', '--run_id', type=int, default=0,
                        help='keeps track of the run id')
    args = parser.parse_args(raw_args)
    # output them as a make-sure-this-is-what-you-meant
    print('commandline arguments:', args, '\n')

    # set up random streams
    streams = [np.random.Generator(PCG64(args.random_state, i)) for i in range(4)]
    loc_strm, strat_strm, rate_strm, pax_strm = streams

    location_sampler = loc_strm.choice # used to make initial placement of pax
    strategy_generator = lambda: strat_strm.uniform(.5, 1) 
    rate_generator = lambda: rate_strm.uniform(.9, 1.1)
    person_mover = lambda: pax_strm.uniform() 

    # create an instance of Floor
    floor = EarthquakeSim(args.input, args.numpeople, location_sampler,
                    strategy_generator, rate_generator, person_mover,
                    damage_rate=args.damage_rate,
                    bottleneck_delay=args.bottleneck_delay,
                    animation_delay=args.animation_delay, verbose=args.output, 
                    scaredness_rate=args.scaredness_rate, follower_rate=args.follower_rate, 
                    run_id=args.run_id)

    # call the simulate method to run the actual simulation
    floor.simulate(maxtime=args.max_time, spread_damage=not args.no_spread_damage,
                   gui=not args.no_graphical_output)

    stats = floor.stats()
    return stats

if __name__ == '__main__':
    stats = main()
    