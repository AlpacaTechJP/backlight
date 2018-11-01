from enum import Enum


class Action(Enum):
    TakeShort = -1
    Donothing = 0
    TakeLong = 1

    def act_on_amount(self):
        return self.value
