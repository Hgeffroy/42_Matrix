from typing import List, Tuple, TypeVar, Generic
import math
import copy


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

    def __eq__(self, other) -> bool:
        return all(self._coordinates[i] - other.get_coordinates()[i] < 1e-10 for i in range(self._size))

    def __abs__(self) -> float:
        return self.norm_1()

    def get_coordinates(self) -> List[T]:
        return self._coordinates

    def get_type(self):
        return self._type

    def get_size(self) -> int:
        return self._size

    def is_null(self) -> bool:
        return all(self._coordinates[i] == self._type() for i in range(self._size))

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

    def __eq__(self, other) -> bool:
        return all(self._columns == other.get_columns() for i in range(self._nbcolumns))

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

    def row_echelon(self) -> 'Matrix[T]':
        m_col = self.transpose().get_columns()
        h = 0
        k = 0

        while h < self._nbrows and k < self._nbcolumns:
            lst_pivots = [m_col[i].get_coordinates()[k] for i in range(h, self._nbrows)]
            i_max = lst_pivots.index(max(lst_pivots)) + h
            if m_col[i_max].get_coordinates()[k] == 0:
                k += 1
            else:
                m_col = swap_vectors(m_col, h, i_max)
                for i in range(h + 1, self._nbrows):
                    ratio = m_col[i].get_coordinates()[k] / m_col[h].get_coordinates()[k]
                    m_col[i] = m_col[i] - m_col[h] * ratio

                h += 1
                k += 1

        return Matrix(m_col).transpose()

    def determinant(self) -> T:
        if self._nbcolumns != self._nbrows:
            raise ValueError('Matrix must be square to compute determinant')
        if self._nbcolumns < 2:
            raise ValueError('Cannot compute determinant of matrix with less than 2 columns')

        if self._nbcolumns == 2:
            det = (self._columns[0].get_coordinates()[0] * self._columns[1].get_coordinates()[1] -
                   self._columns[0].get_coordinates()[1] * self._columns[1].get_coordinates()[0])
            return det

        else:
            det = self._type()
            lst = [self._columns[j].get_coordinates() for j in range(1, self._nbcolumns)]
            for i in range(0, self._nbcolumns):
                lst_vec = []
                for j in range(len(lst)):
                    lst_tmp = copy.deepcopy(lst)
                    lst_tmp[j].pop(i)
                    lst_vec.append(Vector(lst_tmp[j]))
                det += Matrix(lst_vec).determinant() * pow(-1, i) * self._columns[0].get_coordinates()[i]
            return det

    def _gauss_descent(self) -> Tuple['Matrix[T]', 'Matrix[T]']:
        m_id = identity_matrix(self._nbcolumns, self._type).get_columns()  # No need to transpose since transpose(I) = I
        m_col = self.transpose().get_columns()
        h = 0
        k = 0

        while h < self._nbrows and k < self._nbcolumns:
            lst_pivots = [m_col[i].get_coordinates()[k] for i in
                          range(h, self._nbrows)]
            i_max = lst_pivots.index(max(lst_pivots)) + h
            if m_col[i_max].get_coordinates()[k] == 0:
                k += 1
            else:
                m_col = swap_vectors(m_col, h, i_max)
                m_id = swap_vectors(m_id, h, i_max)
                for i in range(h + 1, self._nbrows):
                    ratio = m_col[i].get_coordinates()[k] / m_col[h].get_coordinates()[k]
                    m_col[i] = m_col[i] - m_col[h] * ratio
                    m_id[i] = m_id[i] - m_id[h] * ratio

                h += 1
                k += 1

        return Matrix(m_col), Matrix(m_id)

    def _gauss_ascent(self, m: 'Matrix[T]', identity: 'Matrix[T]') -> Tuple['Matrix[T]', 'Matrix[T]']:
        m_col = m.get_columns()
        m_id = identity.get_columns()

        for i in range(self._nbcolumns - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                ratio = m_col[j].get_coordinates()[i] / m_col[i].get_coordinates()[i]
                m_col[j] = m_col[j] - m_col[i] * ratio
                m_id[j] = m_id[j] - m_id[i] * ratio
            ratio = 1 / m_col[i].get_coordinates()[i]
            m_col[i] = m_col[i] * ratio
            m_id[i] = m_id[i] * ratio

        return Matrix(m_col).transpose(), Matrix(m_id).transpose()

    def inverse(self) -> 'Matrix[T]':
        if self._nbcolumns != self._nbrows:
            raise ValueError('Matrix must be square to be invertible')
        if self.determinant() == 0:
            raise Exception('Matrix is singular')

        m, m_id = self._gauss_descent()
        m, m_id = self._gauss_ascent(m, m_id)

        return m_id

    def rank(self) -> int:
        m = self.row_echelon()
        m_col = m.transpose().get_columns()
        count = 0
        for i in range(len(m_col)):
            if not m_col[i].is_null():
                count += 1

        return count


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


def swap_vectors(vectors: List[Vector[T]], row1: int, row2: int) -> List[Vector[T]]:
    tmp = vectors[row1]
    vectors[row1] = vectors[row2]
    vectors[row2] = tmp
    return vectors


def identity_matrix(sz: int, tp: type) -> Matrix[T]:
    lst_identity = []
    for i in range(sz):
        lst_identity.append([tp()] * sz)
        lst_identity[i][i] = tp(1)
    return Matrix([Vector(lst_identity[i]) for i in range(sz)])





