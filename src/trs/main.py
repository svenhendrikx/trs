from __future__ import annotations
from rich import print

from abc import ABC, abstractmethod
from typing import List, Union, Dict
from functools import reduce
from pathlib import Path
# from sys import argv

from typing import Iterator
from dataclasses import dataclass
import re
from re import Match, match

script_file = Path('./script.t')


class RegexDict(dict):

    def by_match(self, pattern: str):
        regex_matches = {re.compile("^"+key+"$").match(
            pattern
            ): val for key, val in self.items()}
        # lambda match: match is not None,
        return next(filter(lambda keyval: keyval[0] is not None,
                           regex_matches.items(),
                           )
                    )[1]

OPERATIONS_MAP = {'state': RegexDict({r'S': 'Set',
                                      r'G': 'Get',
                                      r'P': 'Print',
                                      r'\+': 'Add',
                                      r'-': 'Subtract',
                                      r'\*': 'Multiply',
                                      r'/': 'Divide',
                                      }),
                  'value': RegexDict({r'\d': 'Number',
                                      r'[a-z]': 'Variable',
                                      }
                                     )
                  }


def isint(x):
    try:
        int(x)
        return True
    except:
        return False

def is_state(command: str):
    print(re.match(command[0], r'\d'))


class Operation(ABC):

    @abstractmethod
    def execute(self):
        raise NotImplementedError()

    @classmethod
    def from_string(cls,
                    input_string: str,
                    **kwargs
                    ):
        if command:
            op = get_operation(command[0])

            

            op.from_string(command[2: -1],
                           **kwargs,
                           )
        else:
            return None


def get_operation(input_string: str) -> Operation:

    return globals()[OPERATIONS_MAP[input_string]]


@dataclass(frozen=True)
class Number(Operation):
    x: int | float

    def execute(self) -> int | float:
        return self.x

    @classmethod
    def from_string(cls,
                    input_string: str,
                    ):
        return cls(x=int(input_string))


@dataclass(frozen=True)
class Variable(Operation):
    x: str

    def execute(self) -> str:
        return self.x


State = Dict[Variable, Number | Operation]


class StateOperation(Operation):

    def __init__(self,
                 state: State,
                 ) -> None:
        self.state = state

    @abstractmethod
    def execute(self):
        raise NotImplementedError()


class MathOperation(StateOperation):
    numbers: List[Number]

    @abstractmethod
    def execute(self) -> Number:
        raise NotImplementedError()


class BinaryMathOperation(MathOperation):

    def __post_init__(self):
        if len(self.numbers) != 2:  # Example: Enforcing exactly 3 elements
            raise ValueError(f"{self.__name__} requires exactly 2 arguments.")

    @abstractmethod
    def execute(self) -> Number:
        raise NotImplementedError()


class Set(StateOperation):

    def __init__(self,
                 update: State,
                 **kwargs,
                 ) -> None:
        self.update = update
        super().__init__(**kwargs)

    def execute(self) -> None:
        self.state.update(self.update)

    @classmethod
    def from_string(cls,
                    input_string: str,
                    state: State,
                    ):

        key = Variable(x=input_string[0])

        v = input_string[1]
        if v in OPERATIONS_MAP:
            op = get_operation(v)
            val = op.from_string(input_string[2:-1], state)
        else:
            val = Number.from_string(input_string=input_string[1:])
        return cls(state=state,
                   update={key: val},
                   )


class Get(StateOperation):

    def __init__(self,
                 symbol: str,
                 **kwargs,
                 ) -> None:
        self.symbol = symbol
        super().__init__(**kwargs)

    def execute(self) -> Number | Operation:
        return self.state.get(self.symbol, None)

    @classmethod
    def from_string(cls,
                    input_string: str,
                    state: State,
                    ):

        return cls(symbol=Variable(input_string))


@dataclass(frozen=True)
class Print(StateOperation):

    def __init__(self,
                 msg: Operation,
                 **kwargs,
                 ) -> None:
        self.msg = msg
        super().__init__(**kwargs)

    def execute(self):
        print(self.msg.execute())

    @classmethod
    def from_string(cls,
                    input_string: str,
                    state: State,
                    ):

        return cls(msg=Operation.from_string(input_string,
                                             state,
                                             ))




class Add(BinaryMathOperation):
    def execute(self) -> Number:
        return self.numbers[0] + self.numbers[1]


class Multiply(BinaryMathOperation):
    def execute(self) -> Number:
        return self.numbers[0] * self.numbers[1]


class Subtract(BinaryMathOperation):
    def execute(self) -> Number:
        return self.numbers[0] - self.numbers[1]


class Divide(BinaryMathOperation):
    def execute(self) -> Number:
        return self.numbers[0] / self.numbers[1]


def read_script(path: Path) -> Iterator[str]:
    with open(script_file, 'r') as f:
        return map(lambda line: line.strip(),
                   reduce(lambda acc, val: acc + val,
                          map(lambda line: line.split(';'),
                              filter(lambda line: line[0] != '#',
                                     f.readlines(),
                                     ),
                              ),
                          [],
                          )
                   )


script = read_script(script_file)

state = {}
program = []
for command in script:
    program.append(Operation.from_string(command,
                                         state=state,
                                         )
                   )

print(program)
for operation in program:
    operation.execute()
print(state)
