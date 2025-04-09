from typing import List, Tuple, TypeVar, Generic
import math


# Conditions for T:
# - Overloaded add operator
# - Overloaded sub operator
# - Overloaded mul operator
# - Overloaded abs
# - Overload < > = operators
T = TypeVar('T')


class Vector(Generic[T]):
    """
    Class representing a vector of T.
    """

    def __init__(self, coordinates: List[T]):
        if len(coordinates) == 0:
            raise ValueError("Vector cannot be empty")

        self._type = type(coordinates[0])
        self._size = len(coordinates)
        self._coordinates = coordinates

    def __str__(self) -> str:
        string = ""
        for coord in self._coordinates:
            string += f'{coord}   '

        return string

    def __len__(self) -> int:
        return self._size

    def __add__(self, other) -> 'Vector[T]':
        coord = []
        for c in range(self._size):
            coord.append(self._coordinates[c] + other.get_coordinates()[c])
        return Vector(coord)

    def __sub__(self, other) -> 'Vector[T]':
        coord = []
        for c in range(self._size):
            coord.append(self._coordinates[c] - other.get_coordinates()[c])
        return Vector(coord)

    def __mul__(self, scalar: T) -> 'Vector[T]':
        coord = []
        for c in range(self._size):
            coord.append(self._coordinates[c] * scalar)
        return Vector(coord)

    def __mul__(self, v: 'Vector[T]') -> float:
        if self._size != len(v):
            raise ValueError('Vector does not have the same size')

        res = 0.
        for i in range(self._size):
            res += self._coordinates[i] * v.get_coordinates()[i]

        return res

    def __abs__(self) -> float:
        return self.norm_1()

    def get_coordinates(self) -> List[T]:
        return self._coordinates

    def get_type(self):
        return self._type

    def norm(self) -> float:
        results = 0.
        # CHECK: If T is a vector, this will be the scalar product between c and c
        for c in self._coordinates:
            results += c * c
        return math.sqrt(results)

    def norm_1(self) -> float:
        results = 0.
        for c in self._coordinates:
            results += abs(c)

        return results

    def norm_inf(self) -> float:
        abslist = [abs(c) for c in self._coordinates]
        return max(abslist)


def linear_combination(u: List[Vector[T]], coefs: List[T]) -> Vector[T]:
    if len(u) != len(coefs) or len(u) == 0:
        raise ValueError("Arguments must have the same size greater than 0")

    ttype = type(u[0].get_type())
    vec = Vector([ttype()] * len(u[0]))

    for i in range(len(coefs)):
        vec += (u[i] * coefs[i])
    return vec


def linear_interpolation(u: T, v: T, t: float) -> T:
    diff = v - u
    diff = diff * t
    result = u + diff
    return result






