# training/core/transposition.py

class TranspositionTable:
    def __init__(self):
        self.table = {}
        self.hits = 0

    def get(self, key: int):
        if key in self.table:
            self.hits += 1
            return self.table[key]
        return None

    def put(self, key: int, score: int):
        self.table[key] = score

    def reset(self):
        self.table.clear()
        self.hits = 0