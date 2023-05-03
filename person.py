'''
This file accompanies other files in the evacuation simulation project.
people: Nick B., Matthew J., Aalok S.

In this file we define a useful class for the agent, 'Person'
'''

import numpy as np

class Person:
    id = None
    rate = None # how long it takes to move one unit distance
    strategy = None # probability with which agent picks closes exit
    loc = None, None # variable tracking this agent's location (xy coordinates)

    alive = True 
    safe = False # mark safe once successfully exited. helps track how many
                 # people still need to finish

    exit_time = 0 # time it took this agent to get to the safe zone from its
                  # starting point
    scaredness = None


    def __init__(self, id, rate:float=1.0, strategy:float=.7, loc:tuple=None, scaredness:int=0):
        '''
        constructor method
        ---
        rate
        strategy
        loc
        '''
        self.id = id
        self.rate = rate
        self.strategy = strategy
        self.loc = tuple(loc)
        self.scaredness = scaredness

    def move(self, nbrs, rv=None):
        '''
        when this person has finished their current movement, we must schedule
        the next one
        ---
        graph (dict): a dictionary-like graph storing the floor plan according
                      to our specification

        return: tuple, location the agent decided to move to
        '''
        # decide safe neighbours
        nbrs = [(loc, attrs) for loc, attrs in nbrs
                if not(attrs['G'] or attrs['W'])]
        #TODO nadia: have a chance of adding a damaged / risky cell into the neighbours depending on scaredness, (e.g. risky: 30%, damaged: 5%)
        #TODO nadia: graves cannot be neighbours

        # neighbors = [] 
        # for loc, attrs in nbrs:
        #     if attrs['N'] or attrs['B'] or attrs['S']:
        #         neighbors.append((loc,attrs))
        #     if attrs['R']:
        #         # scared people will choose R with 50/50 chance
        #         if self.scaredness and np.random.uniform(0,1) < 0.5:
        #             neighbors.append((loc,attrs))
        #         # non scared people will always consider R as a neighbor
        #         elif not self.scaredness:
        #             neighbors.append((loc,attrs))
        #     if attrs['D']:
        #         # scared people will choose R with 30% chance
        #         if self.scaredness and np.random.uniform(0,1) < 0.3:
        #             neighbors.append((loc,attrs))
        #         # non scared people will choose R with 80% chance
        #         elif (not self.scaredness) and np.random.uniform(0,1) < 0.7:
        #             neighbors.append((loc,attrs))

        if not nbrs: return None

        # find the neighbour that is the shortest distance from the door
        # loc, attrs = min(nbrs, key=lambda tup: tup[1]['distS'])
        nbrs.sort(key=lambda tup: tup[1]['distS'])
        ind = 0
        loc, attrs = nbrs[ind] # or nbrs[ind][0], im sorry i cant debug :(
        while (attrs['D'] or attrs['R']): 
            # if the best location is damaged, rethink
            if attrs['D']:
                # scared people will choose R with 30% chance
                if self.scaredness and np.random.uniform(0,1) < 0.3:
                    break
                # non scared people will choose R with 80% chance
                elif (not self.scaredness) and np.random.uniform(0,1) < 0.7:
                    break
            # if the best location is risky, scared people rethink (choose to continue with 50/50 chance)
            if attrs['R'] and self.scaredness and np.random.uniform(0,1) < 0.5:
                break
            # if the ifs didn't happen, the current location is bad, get the next one
            ind += 1
            loc, attrs = nbrs[ind] # or nbrs[ind][0], im sorry i cant debug :(


        #TODO: strategy: follow other people
        #TODO: strategy: move away from danger

        # print('Person {} at {} is moving to {}'.format(self.id, self.loc, loc))
        # print('Person {} is {} away from safe'.format(self.id, attrs['distS']))
        self.loc = loc
        if attrs['S']:
            self.safe = True
        elif attrs['F']:
            self.alive = False

        return loc
