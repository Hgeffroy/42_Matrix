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

    def tomatrix(self):
        return Matrix([self._coordinates])

    def getcoordinates(self) -> List[T]:
        return self._coordinates


class Matrix(Generic[T]):
    """
    Class representing a matrix of int.
    """

    def __init__(self, values: List[List[T]]):
        for val in values:
            if len(val) != len(values[0]):
                raise ValueError("Matrix columns must have the same length.")

        self._columnsNb = len(values)
        self._rowsNb = len(values[0])
        self._values = values

    def __str__(self):
        string = ""
        for col in self._values:
            for val in col:
                string += f'{val} '
            string += '\n'

        return string

    def __sizeof__(self):
        return self._columnsNb * self._rowsNb

    def __add__(self, other):
        values = []
        for c in range(self._columnsNb):
            col = []
            for r in range(self._rowsNb):
                col.append(self._values[r][c] + other.getValues()[r][c])
            values.append(col)
        return Matrix(values)

    def __sub__(self, other):
        values = []
        for c in range(self._columnsNb):
            col = []
            for r in range(self._rowsNb):
                col.append(self._values[r][c] - other.getValues()[r][c])
            values.append(col)
        return Matrix(values)

    def __mul__(self, scalar: T):
        values = []
        for c in range(self._columnsNb):
            col = []
            for r in range(self._rowsNb):
                col.append(self._values[r][c] * scalar)
            values.append(col)
        return Matrix(values)

    def tovector(self):
        coordinates = []
        for col in self._values:
            coordinates.extend(col)
        return Vector(coordinates)

    def getvalues(self) -> List[List[T]]:
        return self._values

    def shape(self) -> Tuple[int, int]:
        return self._rowsNb, self._columnsNb

    def issquare(self) -> bool:
        return self._rowsNb == self._columnsNb


def linear_combination(u: List[Vector[T]], coefs: List[T]) -> Vector[T]:
    if len(u) != len(coefs):
        raise ValueError("Arguments must have the same size")

    vec = Vector([0] * len(u))
    for i in range(len(u)):
        vec += (u[i] * coefs[i])
    return vec











