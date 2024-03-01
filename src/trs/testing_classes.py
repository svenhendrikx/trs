from dataclasses import dataclass


@dataclass
class foo:
    a: int


@dataclass
class bar(foo):
    b: int


bar(b='a', a=1)
