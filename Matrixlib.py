from typing import List, Tuple, TypeVar, Generic
import math


# Conditions for T:
# - Overloaded add operator
# - Overloaded sub operator
# - Overloaded mul operator
# - Overloaded abs
# - Overload < > = operators for abs values ? (for norm)
T = TypeVar('T')


class Vector(Generic[T]):
    """
        Class representing a vector of T.
    """

    def __init__(self, coordinates: List[T]):
        if len(coordinates) == 0:
            raise ValueError('Vector cannot be empty')

        self._type = type(coordinates[0])
        self._size = len(coordinates)
        self._coordinates = coordinates

    def __str__(self) -> str:
        string = ''
        for coord in self._coordinates:
            string += f'{coord}   '

        return string + '\n'

    def __len__(self) -> int:
        return self._size

    def __add__(self, other) -> 'Vector[T]':
        return Vector(
            [self._coordinates[c] + other.get_coordinates()[c] for c in range(self._size)])

    def __sub__(self, other) -> 'Vector[T]':
        return Vector([self._coordinates[c] - other.get_coordinates()[c] for c in range(self._size)])

    def __mul__(self, scalar: T) -> 'Vector[T]':
        return Vector([self._coordinates[c] * scalar for c in range(self._size)])

    def __abs__(self) -> float:
        return self.norm_1()

    def get_coordinates(self) -> List[T]:
        return self._coordinates

    def get_type(self):
        return self._type

    def get_size(self) -> int:
        return self._size

    def dot(self, v: 'Vector[T]') -> float:
        if self._size != len(v):
            raise ValueError('Vector does not have the same size')

        res = 0.
        for i in range(self._size):
            res += self._coordinates[i] * v.get_coordinates()[i]

        return res

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
        return max([abs(c) for c in self._coordinates])


class Matrix(Generic[T]):
    """
        Class representing a matrix constituted of Vectors
    """

    def __init__(self, columns: List[Vector[T]]):
        if not all(len(columns[i]) == len(columns[0]) for i in range(len(columns))):
            raise ValueError('All vectors must have the same size')

        self._columns = columns
        self._nbcolumns = len(columns)
        self._nbrows = len(columns[0])
        self._type = columns[0].get_type()

    def __str__(self) -> str:
        string = ''
        for i in range(self._nbrows):
            for j in range(self._nbcolumns):
                string += f'{self._columns[j].get_coordinates()[i]}   '
            string += '\n'
        return string

    def __add__(self, other: 'Matrix[T]') -> 'Matrix[T]':
        if self.get_shape() != other.get_shape():
            raise ValueError('Matrix cannot be added with different shapes')

        return Matrix([self._columns[i] + other.get_columns()[i] for i in range(self._nbcolumns)])

    def __sub__(self, other: 'Matrix[T]') -> 'Matrix[T]':
        if self.get_shape() != other.get_shape():
            raise ValueError('Matrix cannot be added with different shapes')

        return Matrix([self._columns[i] - other.get_columns()[i] for i in range(self._nbcolumns)])

    def __mul__(self, scalar: T) -> 'Matrix[T]':
        return Matrix([self._columns[i] * scalar for i in range(self._nbcolumns)])

    def get_shape(self) -> Tuple[int, int]:
        return self._nbrows, self._nbcolumns

    def get_columns(self) -> List[Vector[T]]:
        return self._columns

    def mul_vec(self, vec: 'Vector[T]') -> 'Vector[T]':
        if self._nbcolumns != len(vec):
            raise ValueError('Matrix must have as many columns as vector dimension')

        return linear_combination(self._columns, vec.get_coordinates())

    def mul_mat(self, mat: 'Matrix[T]') -> 'Matrix[T]':
        if self._nbcolumns != mat.get_shape()[1]:
            raise ValueError('First matrix must have as many columns as second matrix has rows')

        return Matrix([self.mul_vec(mat.get_columns()[i]) for i in range(mat.get_shape()[1])])

    def trace(self) -> T:
        if self._nbcolumns != self._nbrows:
            raise ValueError('Matrix must be square to compute trace')

        result = self._type()
        for i in range(self._nbcolumns):
            result += self._columns[i].get_coordinates()[i]
        return result

    def transpose(self) -> 'Matrix[T]':
        return Matrix([Vector([self._columns[i].get_coordinates()[j]
                               for i in range(self._nbcolumns)]) for j in range(self._nbrows)])


def linear_combination(u: List[Vector[T]], coefs: List[T]) -> Vector[T]:
    if len(u) != len(coefs) or len(u) == 0:
        raise ValueError('Arguments must have the same size greater than 0')

    ttype = type(coefs[0])
    vec = Vector([ttype()] * len(u[0]))

    for i in range(len(coefs)):
        vec += u[i] * coefs[i]
    return vec


def linear_interpolation(u: T, v: T, t: float) -> T:
    diff = v - u
    diff = diff * t
    result = u + diff
    return result


def angle_cos(u: Vector[T], v: Vector[T]) -> float:
    if u.get_size() != v.get_size() or u.get_size() == 0:
        raise ValueError('Arguments must have the same size greater than 0')

    # CHECK: Division induces approx
    return (u.dot(v))/(u.norm() * v.norm())


def cross_product(u: Vector[T], v: Vector[T]) -> Vector[T]:
    if u.get_size() != 3 or v.get_size() != 3:
        raise ValueError('Vectors must be three dimensional')

    ucoord = u.get_coordinates()
    vcoord = v.get_coordinates()

    return Vector([ucoord[1] * vcoord[2] - ucoord[2] * vcoord[1],
                   ucoord[2] * vcoord[0] - ucoord[0] * vcoord[2],
                   ucoord[0] * vcoord[1] - ucoord[1] * vcoord[0]])






