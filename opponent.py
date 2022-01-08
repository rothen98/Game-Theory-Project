class Opponent:
    def __init__(self, attitude, belief, nashParameter):
        self.attitude = attitude
        self.belief = belief
        self.nashParameter = nashParameter

    def readable(self):
        return f"Opponent has attitude {self.attitude}, belief {self.belief}, and nash parameter {self.nashParameter}"
