from Matrixlib import *


def test00():
    print('Testing Addition for vector:')
    a = Vector([1., 2., 3.])
    print(f'Vector a: \n{str(a)}')
    b = Vector([4., 5., 6.])
    print(f'Vector b: \n{str(b)}')
    c = a + b
    print(f'Vector c = a + b: \n{str(c)}')

    print('Testing Subtraction for vector:')
    c = a - b
    print(f'Vector c = a - b: \n{str(c)}')

    print('Testing Scalar Multiplication for vector:')
    c = a * 3.
    print(f'Vector c = a * 3: \n{str(c)}')


def test01():
    print('Testing linear combination')
    e1 = Vector([1., 0., 0.])
    e2 = Vector([0., 1., 0.])
    e3 = Vector([0., 0., 1.])
    v1 = Vector([1., 2., 3.])
    v2 = Vector([0., 10., -100.])

    vec1 = linear_combination([e1, e2, e3], [10., -2., 0.5])
    print(f'{vec1}')
    vec2 = linear_combination([v1, v2], [10., -2.])
    print(f'{vec2}')


def test02():
    print('Testing linear interpolation')
    print('\nFloats: ')
    print(linear_interpolation(0., 1., 0.))
    print(linear_interpolation(0., 1., 1.))
    print(linear_interpolation(0., 1., 0.5))
    print(linear_interpolation(21., 42., 0.3))

    print('\nVectors: ')
    v1 = Vector([2., 1.])
    v2 = Vector([4., 2.])
    print(linear_interpolation(v1, v2, 0.3))


def test03():
    print('Testing dot product')
    v1 = Vector([0., 0.])
    v2 = Vector([1., 1.])
    v3 = Vector([-1., 6.])
    v4 = Vector([3., 2.])
    print(v1.dot(v2))
    print(v2.dot(v2))
    print(v3.dot(v4))


def test04():
    print('Testing norm')
    v1 = Vector([0., 0., 0.])
    v2 = Vector([1., 2., 3.])
    v3 = Vector([-1., 2.])
    print('v1: ' + str(v1))
    print(f'norm: {v1.norm()} norm_1: {v1.norm_1()} norm_inf: {v1.norm_inf()}')
    print('v2: ' + str(v2))
    print(f'norm: {v2.norm()} norm_1: {v2.norm_1()} norm_inf: {v2.norm_inf()}')
    print('v3: ' + str(v3))
    print(f'norm: {v3.norm()} norm_1: {v3.norm_1()} norm_inf: {v3.norm_inf()}')


def test05():
    print('Testing cosine of angle between vectors')
    v1 = Vector([1., 0.])
    v2 = Vector([0., 1.])
    v3 = Vector([-1., 1.])
    v4 = v3 * -1
    v5 = Vector([2., 1.])
    v6 = v5 * 2
    v7 = Vector([1., 2., 3.])
    v8 = Vector([4., 5., 6.])
    print(angle_cos(v1, v1))
    print(angle_cos(v1, v2))
    print(angle_cos(v3, v4))
    print(angle_cos(v5, v6))
    print(angle_cos(v7, v8))


def test06():
    print('Testing cross product')
    v1 = Vector([0., 0., 1.])
    v2 = Vector([1., 0., 0.])
    v3 = Vector([1., 2., 3.])
    v4 = Vector([4., 5., 6.])
    v5 = Vector([4., 2., -3.])
    v6 = Vector([-2., -5., 16.])
    print(cross_product(v1, v2))
    print(cross_product(v3, v4))
    print(cross_product(v5, v6))


def test07():
    print('Testing matrices multiplication')
    v1 = Vector([4., 2.])
    m1 = Matrix([Vector([1., 0.]), Vector([0., 1.])])
    m2 = m1 * 2
    m3 = Matrix([Vector([2., -2.]), Vector([-2., 2.])])
    m4 = Matrix([Vector([2., 1.]), Vector([4., 2.])])
    m5 = Matrix([Vector([3., -5.]), Vector([6., 8.])])
    print('Matrix * Vector:')
    print(m1.mul_vec(v1))
    print(m2.mul_vec(v1))
    print(m3.mul_vec(v1))

    print('Matrix * Matrix:')
    print(m1.mul_mat(m1))
    print(m1.mul_mat(m4))
    print(m4.mul_mat(m5))


def test08():
    print('Testing matrix trace computation')
    m1 = Matrix([Vector([1., 0.]), Vector([0., 1.])])
    m2 = Matrix([Vector([2., -5., 0.]),
                 Vector([4., 3., 7.]),
                 Vector([-2., 3., 4.])])
    m3 = Matrix([Vector([-2., -8., 4.]),
                 Vector([1., -23., 4.]),
                 Vector([0., 6., 4.])])
    print(m1.trace())
    print(m2.trace())
    print(m3.trace())


def test09():
    print('Testing matrix transposition')
    m1 = Matrix([Vector([1., 0.]), Vector([0., 1.])])
    m2 = Matrix([Vector([2., -5., 0.]),
                 Vector([4., 3., 7.]),
                 Vector([-2., 3., 4.])])
    m3 = Matrix([Vector([-2., -8., 4.]),
                 Vector([1., -23., 4.])])
    print(m1)
    print(m1.transpose())
    print(m2)
    print(m2.transpose())
    print(m3)
    print(m3.transpose())

def test10():
    m1 = Matrix([Vector([1., 0., 0.]), Vector([0., 1., 0.]), Vector([0., 0., 1.])])
    m2 = Matrix([Vector([1., 2.]), Vector([3., 4.])])
    m3 = Matrix([Vector([1., 2.]), Vector([2., 4.])])
    m4 = Matrix([Vector([2., -5., 0.]),
                 Vector([4., 3., 7.]),
                 Vector([-2., 3., 4.])])
    m5 = Matrix([Vector([8., 5., -2., 4., 28.]),
                 Vector([4., 2.5, 20., 4., -4.]),
                 Vector([8., 5., 1., 4., 17.])])

    print(m1.row_echelon())
    print(m2.row_echelon())
    print(m3.row_echelon())
    print(m4.row_echelon())
    print(m5.row_echelon())


def test11():
    m1 = Matrix([Vector([2., 0., 0.]), Vector([0., 2., 0.]), Vector([0., 0., 2.])])
    m2 = Matrix([Vector([8., 4., 7.]), Vector([5., 7., 6.]), Vector([-2., 20., 1.])])
    m3 = Matrix([Vector([8., 4., 8., 28.]), Vector([5., 2.5, 5., -4.]), Vector([-2., 20., 1., 17.]), Vector([4., 4., 4., 1.])])

    print(m1.determinant())
    print(m2.determinant())
    print(m3.determinant())


def main():
    # test00()
    # test01()
    # test02()
    # test03()
    # test04()
    # test05()
    # test06()
    # test07()
    # test08()
    # test09()
    # test10()
    test11()
    return


if __name__ == '__main__':
    main()
