import heapq

class Node:
    def __init__(self, state, parent, cost, heuristic):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def astar(start, goal, graph, max_nodes):
    heap = []
    heapq.heappush(heap, (0, Node(start, None, 0, 0)))
    visited = set()
    node_counter = 0

    while heap and node_counter < max_nodes:
        (cost, current) = heapq.heappop(heap)

        if current.state == goal:
            path = []
            while current is not None:
                path.append(current.state)
                current = current.parent
            return path[::-1]

        if current.state in visited:
            continue

        visited.add(current.state)
        node_counter += 1

        for state, cost in graph[current.state].items():
            if state not in visited:
                heuristic = 0  # Replace with your heuristic function if needed
                total_cost = current.cost + cost + heuristic
                heapq.heappush(heap, (total_cost, Node(state, current, current.cost + cost, heuristic)))

    return None

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start = 'A'
goal = 'D'
max_nodes = 10

result = astar(start, goal, graph, max_nodes)
print(result)

# OUTPUT:
# ['A', 'B', 'C', 'D']
