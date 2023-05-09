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

    def closestExit(self, nbrs):
        # find the neighbour that is the shortest distance from the door
        nbrs.sort(key=lambda tup: tup[1]['distS'])
        ind = 0
        loc, attrs = nbrs[ind]
        while ((attrs['D'] or attrs['R']) and len(nbrs) > 0):
            # if the best location is damaged, rethink
            if attrs['R']:
                # scared people will choose R with 70% chance
                if self.scaredness and np.random.uniform(0,1) < 0.7:
                    nbrs.append((loc,attrs))
                    break
                # non scared people will choose R with 90% chance
                elif (not self.scaredness) and np.random.uniform(0,1) < 0.9:
                    nbrs.append((loc,attrs))
                    break
            # if the best location is risky, scared people rethink (choose to continue with 50/50 chance)
            if attrs['D'] and (not self.scaredness) and np.random.uniform(0,1) < 0.5:
                nbrs.append((loc,attrs))
                break
            # if the ifs didn't happen, the current location is bad, get the next one
            ind += 1
            loc, attrs = nbrs[ind]
        return loc, attrs
    
    def followPeople(self, nbrs):
        ind = 0
        loc, attrs = nbrs[ind]

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
        if not nbrs: return None

        if self.strategy is "closest exit":
            loc, attrs = self.closestExit(nbrs)
        else:
            loc, attrs = self.followPeople(nbrs)



        #TODO: strategy: follow other people
        #TODO: strategy: move away from danger

        # print('Person {} at {} is moving to {}'.format(self.id, self.loc, loc))
        # print('Person {} is {} away from safe'.format(self.id, attrs['distS']))
        self.loc = loc
        if attrs['S']:
            self.safe = True

        return loc
