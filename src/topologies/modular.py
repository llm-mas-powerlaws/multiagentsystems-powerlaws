class ModularTopology:

    def initialize(self, agents: list[str], seed: int = 0):
        self.agents = agents
        n = len(agents)
        self.adjacency = {a: [] for a in agents}

        # Determine communities based on scale
        if n <= 8:
            n_communities = 2
        else:
            n_communities = 3

        # Split agents into communities
        self.communities = []
        size = n // n_communities
        for i in range(n_communities):
            start = i * size
            end = start + size if i < n_communities - 1 else n
            self.communities.append(agents[start:end])

        # Fully connect within each community
        for community in self.communities:
            for a in community:
                for b in community:
                    if a != b:
                        self.adjacency[a].append(b)

        # Add fixed bridge edges between communities
        # One bridge agent per community connected in a chain
        for i in range(len(self.communities) - 1):
            bridge_a = self.communities[i][1]    # second agent in community i
            bridge_b = self.communities[i + 1][0]  # first agent in community i+1
            if bridge_b not in self.adjacency[bridge_a]:
                self.adjacency[bridge_a].append(bridge_b)
            if bridge_a not in self.adjacency[bridge_b]:
                self.adjacency[bridge_b].append(bridge_a)

    def neighbors(self, agent_id: str) -> list[str]:
        return self.adjacency.get(agent_id, [])

    def name(self) -> str:
        return "modular"