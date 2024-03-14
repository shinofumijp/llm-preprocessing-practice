class UnionFind():
    def __init__(self, elements: list[str]) -> None:
        self.element_map = {element: index for index, element in enumerate(elements)}
        self.root = [-1] * len(elements)
        self.rank = [0] * len(elements)  # Initialize rank

    def find(self, element: str) -> int:
        x = self.element_map[element]
        if self.root[x] < 0:
            return x
        else:
            self.root[x] = self.find(list(self.element_map.keys())[self.root[x]])  # Path compression
            return self.root[x]

    def union(self, element1: str, element2: str):
        root1: int = self.find(element1)
        root2: int = self.find(element2)

        if root1 == root2:
            return

        # Attach smaller rank tree under root of higher rank tree
        if self.rank[root1] < self.rank[root2]:
            self.root[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.root[root2] = root1
        else:
            self.root[root2] = root1
            self.rank[root1] += 1  # Increase rank if both trees have same rank

    def size(self, element):
        return -self.root[self.find(element)]

    def same(self, element1, element2):
        return self.find(element1) == self.find(element2)

    def members(self, element):
        root = self.find(element)
        return [elem for elem, idx in self.element_map.items() if self.find(elem) == root]

    def roots(self):
        return [list(self.element_map.keys())[i] for i, x in enumerate(self.root) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        return {r: self.members(r) for r in self.roots()}

    def groups(self):
        return [self.members(r) for r in self.roots()]
