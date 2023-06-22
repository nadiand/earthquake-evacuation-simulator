'''
This file accompanies other files in the evacuation simulation project.
people: Nick B., Matthew J., Aalok S.

In this file we define a useful class for the agent, 'Person'
'''

import numpy as np
import math

class Person:
    id = None
    rate = None # how long it takes to move one unit distance
    starting_rate = None
    strategy = None # probability with which agent picks closes exit
    loc = None, None # variable tracking this agent's location (xy coordinates)

    alive = True 
    safe = False # mark safe once successfully exited. helps track how many
                 # people still need to finish
    injured = False # mark people that are injured i.e. their rate falls below a threshold

    exit_time = 0 # time it took this agent to get to the safe zone from its starting point
    scaredness = None
    waiting_for_rescue = False
    


    def __init__(self, id, rate:float=1.0, loc:tuple=None, strategy:int=0, scaredness:int=0, graph=None):
        '''
        constructor method
        '''
        self.id = id
        self.rate = rate
        self.strategy = strategy
        self.loc = tuple(loc)
        self.scaredness = scaredness
        self.starting_rate = rate
        self.graph = graph
        self.loc_history = []  # Initialize location history

    def closestExit(self, nbrs):
        # find the neighbour that is the shortest distance from the door
        nbrs.sort(key=lambda tup: tup[1]['distS'])
        ind = 0
        loc, _ = nbrs[ind]
        for location, attributes in nbrs:
            if not (attributes['D'] or attributes['R']):
                loc = location
                break
            # if the best location is risky, rethink
            if attributes['R']:
                # scared people will choose R with 70% chance
                if self.scaredness and np.random.uniform(0,1) < 0.9:
                    loc = location
                    break
                # non scared people will choose R with 90% chance
                elif (not self.scaredness) and np.random.uniform(0,1) < 0.95:
                    loc = location
                    break
            # if the best location is damaged, scared people rethink (choose to continue with 50/50 chance)
            if attributes['D']:
                if self.scaredness and np.random.uniform(0,1) < 0.5:
                    loc = location
                    break
                elif (not self.scaredness) and np.random.uniform(0,1) < 0.7:
                    loc = location
                    break
        
        if loc is None:
            return self.loc
        return loc
    

    def followPeople(self, nbrs, fov, loc_max_people):
        move = None

        # check if person can see a safe space, if so, move toward closest exit
        for loc in fov:
            if self.graph[loc]['S']:
                return self.closestExit(nbrs)

        shortest_dist = float('inf')
        # find direction to loc with the most people
        if loc_max_people is not None:
            for nbr_loc, _ in nbrs:
                dist = math.dist(nbr_loc, loc_max_people)
                if dist < shortest_dist:
                    shortest_dist = dist
                    move = nbr_loc
        else: # go to closest bottleneck
            nbrs.sort(key=lambda tup: tup[1]['distB'])
            loc, _ = nbrs[0]
            return loc

        return move


    def move(self, nbrs, fov, loc_max_people, loc, rv=None):
        '''
        when this person has finished their current movement, we must schedule
        the next one
        ---
        graph (dict): a dictionary-like graph storing the floor plan according
                      to our specification

        return: tuple, location the agent decided to move to
        '''   

        # Decide safe neighbors
        safe_nbrs = [(loc, attrs) for loc, attrs in nbrs if not (attrs['G'] or attrs['W'])]
        if not safe_nbrs:
            return None
        
        loc = None
        if self.strategy == 1:
            loc = self.followPeople(safe_nbrs, fov, loc_max_people)
        else: 
            loc = self.closestExit(safe_nbrs)

        self.loc = loc

        return loc
