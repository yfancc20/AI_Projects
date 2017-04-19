# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# P2-1
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    
    "[Project 2] YOUR CODE HERE"
    "[Command] python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs"
    from util import Stack

    # stack instance, route list, visit list
    stackDFS = Stack()
    route = []
    visit = []

    # step: current step
    # stackStep: return the step encountering intersection
    step = 0
    stackStep = []

    # start point
    start = problem.getStartState()
    stackDFS.push(start)
    first = True

    # DFS function
    # DFS search when routing
    while not stackDFS.isEmpty():
        # step increasing and popping from the stackDFS
        if first: # start point is independent
            currentPos = stackDFS.pop()
            first = False
        else:
            step += 1
            currentState = stackDFS.pop()
            currentPos = currentState[0]
            currentAction = currentState[1]
            # building the route
            route.append(currentAction)

        # check isGoal or not
        if problem.isGoalState(currentPos):
            break

        # getting the successors
        successors = problem.getSuccessors(currentPos)

        # push successors
        # pos[0]: position
        legalSuccessors = 0
        if len(successors) > 0:
            for pos in successors:
                if not pos[0] in visit:
                    stackDFS.push(pos)
                    visit.append(pos[0])
                    legalSuccessors += 1

        # if there is a intersection, recording this step(checkpoint)
        # ex: two directions, recording once; three directions, recording twice.
        if legalSuccessors > 1:
            for i in range(0, legalSuccessors-1):
                stackStep.append(step)

        # No way to go
        if legalSuccessors == 0:
            step = stackStep.pop() # back to the checkpoint
            route = route[0:step]  # delete the route after last checkpoint

    return route
    
    util.raiseNotDefined()

# P2-2
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    "[Project 2] YOUR CODE HERE"
    "[Command] python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs"
    from util import Queue

    # Queue Instance and some lists
    queueBFS = Queue()
    route = []
    visit = []
    predecessorsList = []

    # deal with start point independent
    start = problem.getStartState()
    queueBFS.push(start)
    visit.append(start)

    # VFS function
    # BFS search
    while not queueBFS.isEmpty():
        # popping from the queueBFS
        currentPos = queueBFS.pop()

        # getting the successors
        successors = problem.getSuccessors(currentPos)

        # check isGoal or not
        if problem.isGoalState(currentPos):
            goal = currentPos
            break

        # push successors
        # pos[0]: position, pos[1]: action
        # predecessorsList: [indexPos, indexPos's predecessor, this action]
        if len(successors) > 0:
            for pos in successors:
                if not pos[0] in visit:
                    queueBFS.push(pos[0])
                    visit.append(pos[0])
                    predecessorsList.append([pos[0], currentPos, pos[1]])

    # Build the route from the goal
    indexPos = goal
    while 1:
        for i in range(0, len(predecessorsList)):
            if indexPos == predecessorsList[i][0]:
                # always inserting to the first
                route.insert(0, predecessorsList[i][2])
                indexPos = predecessorsList[i][1]
        # when reaching start, done
        if indexPos == start:
            break

    return route
    
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# P2-3
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "print heuristic(problem.getStartState(), problem)"
    
    "[Project 2] YOUR CODE HERE"
    "[Command] python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic"

    def getG(preG, cost):
        return preG + cost

    def getH(position):
        return heuristic(position, problem)

    def getF(g, h):
        return g + h

    from util import PriorityQueue

    queueAstar = PriorityQueue()
    predecessorsList = {}

    start = problem.getStartState()
    route = []
    openList = []
    closeList = []
    valueList = {}
    predecessorsList = {}

    currentG = 0
    currentH = heuristic(start, problem)
    currentF = getF(currentG, currentH)
    valueList[start] = {'F': currentF, 'G': currentG, 'H': currentH}
    queueAstar.push(start, currentF)
    openList.append(start)
    first = True

    while not queueAstar.isEmpty():
        # pop from the queue, put into cloesList
        currentPos = queueAstar.pop()

        """
        check is removed and add into closeList or not
        because queueAstar cannot change the priority
        it can only add new items with same pos but different priority
        so we should check here if this one is already in closeList
        """
        if currentPos in closeList:
            continue

        # reaching goal, break the loop(searching)
        if problem.isGoalState(currentPos):
            goal = currentPos
            break


        currentG = valueList[currentPos]['G']
        currentH = valueList[currentPos]['H']
        currentF = valueList[currentPos]['F']


        closeList.append(currentPos)
        openList.remove(currentPos)

        # get successors
        successors = problem.getSuccessors(currentPos)

        for pos in successors:
            if pos[0] in openList: # already be found
                # check G and update valueList or not
                nextG = getG(valueList[currentPos]['G'], pos[2])
                if nextG < valueList[pos[0]]['G']:
                    valueList[pos[0]]['G'] = nextG
                    nextF = getF(nextG, valueList[pos[0]]['H'])
                    valueList[pos[0]]['F'] = nextF
                    queueAstar.push(pos[0], nextF)
                    # update the parent
                    predecessorsList[pos[0]] = {'parent': currentPos, 'action': pos[1]}

            elif not pos[0] in closeList: # first be found
                openList.append(pos[0]) # add to openList
                # predecessorList[index]:{'parent', 'action'}
                predecessorsList[pos[0]] = {'parent': currentPos, 'action': pos[1]}
                # calculate F = G + H, and store it in valueList
                nextG = getG(valueList[currentPos]['G'], pos[2])
                nextH = heuristic(pos[0], problem)
                nextF = getF(nextG, nextH)
                valueList[pos[0]] = {'F': nextF, 'G': nextG, 'H': nextH}
                queueAstar.push(pos[0], nextF)

    indexPos = goal
    while 1:
        action = predecessorsList[indexPos]['action']
        route.insert(0, action)
        indexPos = predecessorsList[indexPos]['parent']
        # when reaching start, break
        if indexPos == start:
            break  

    return route
    util.raiseNotDefined()


# Abbreviations
astar = aStarSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
