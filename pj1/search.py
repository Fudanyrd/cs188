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
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    return  [s, s, w, s, w, w, s, w]

class DFSHelper():
    """ An auxiliary class for depthFirstSearch function below. """
    def __init__(
        self, 
        next : list
    ):
       self.Next = next  # Next : list
       self.visited = False
    
    def hasNext(self) -> bool:
        return len(self.Next) != 0

    def getNext(self):
        if len(self.Next) == 0:
            util.raiseNotDefined()
        return self.Next[0]
    def pop(self):
        if len(self.Next) == 0:
            util.raiseNotDefined()
        self.visited = False
        del self.Next[0]

    def addState(
        self,
        state : tuple
    ) -> None:
        self.Next.append(state)

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Get directions.
    visited = set()
    path = util.Stack()
    node = {"state": problem.getStartState(), "path": []}
    path.push(node)
    while True:
        if path.isEmpty():
            return None
        node = path.pop()
        if problem.isGoalState(node["state"]):
            return node["path"]
        if node["state"] not in visited:
            visited.add(node['state'])
            for successor in problem.getSuccessors(node['state']):
                child = {"state": successor[0], "path": node['path'] + [successor[1]]}
                path.push(child)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    node = {"state": problem.getStartState(), "path": []}
    queue = util.Queue()
    queue.push(node)
    # visited.add(node['state'])
    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node['state']):
            return node['path']
        if node['state'] in visited:
            continue  # ignore expanded nodes.
        visited.add(node["state"])
        successors = problem.getSuccessors(node['state'])
        for successor in successors:
            # if successor[0] in visited:
            #    continue
            # visited.add(successor[0])
            child = {'state': successor[0], 'path': node['path'] + [successor[1]]}
            queue.push(child)
    return None

def func(d):
    return d['cost']

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    queue = util.PriorityQueueWithFunction(func)
    node = {'state': problem.getStartState(), 'cost': 0, 'path': []}
    queue.push(node)
    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node['state']):
            return node['path']
        if node['state'] in visited:
            continue
        visited.add(node['state'])
        successors = problem.getSuccessors(node['state'])
        for successor in successors:
            child = {'state': successor[0], 'cost': successor[2] + node['cost'], 'path': node['path'] + [successor[1]]}
            queue.push(child)
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = {'state': problem.getStartState(), 'cost': heuristic(problem.getStartState(), problem), 'path': [],
            'heuristic': heuristic(problem.getStartState(), problem)}
    queue = util.PriorityQueueWithFunction(func)
    visited = set()
    queue.push(node)
    while not queue.isEmpty():
        node = queue.pop()
        if problem.isGoalState(node['state']):
            return node['path']
        if node['state'] in visited:  # ignore already expanded states.
            continue
        visited.add(node['state'])  # OK, mark current state as expanded.
        successors = problem.getSuccessors(node['state'])
        for successor in successors:
            h = heuristic(successor[0], problem)
            cost = h + node['cost'] + successor[2] - node['heuristic']
            child = {'state': successor[0], 'cost': cost, 'path': node['path'] + [successor[1]],
                     'heuristic': h}
            queue.push(child)
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
