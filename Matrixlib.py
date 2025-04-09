from typing import List, Tuple, TypeVar, Generic

# Conditions for T:
# - Overloaded + operator
# - Overloaded - operator
# - Overloaded * operator
T = TypeVar('T')


class Vector(Generic[T]):
    """
    Class representing a vector of T.
    """

    def __init__(self, coordinates: List[T]):
        self._size = len(coordinates)
        self._coordinates = coordinates

    def __str__(self):
        string = f'Vector size: {self._size}\n'f'Coordinates:\n'
        for coord in self._coordinates:
            string += f'{coord} \n'

        return string

    def __sizeof__(self):
        return self._size

    def __add__(self, other):
        coord = []
        for c in range(self._size):
            coord.append(self._coordinates[c] + other.getcoordinates()[c])
        return Vector(coord)

    def __sub__(self, other):
        coord = []
        for c in range(self._size):
            coord.append(self._coordinates[c] - other.getcoordinates()[c])
        return Vector(coord)

    def __mul__(self, scalar: T):
        coord = []
        for c in range(self._size):
            coord.append(self._coordinates[c] * scalar)
        return Vector(coord)

    def getcoordinates(self) -> List[T]:
        return self._coordinates


def linear_combination(u: List[Vector[T]], coefs: List[T]) -> Vector[T]:
    if len(u) != len(coefs):
        raise ValueError("Arguments must have the same size")

    vec = Vector([0] * len(u))
    for i in range(len(u)):
        vec += (u[i] * coefs[i])
    return vec











