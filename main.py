from Matrixlib import *


def test00():
    print("Testing Addition for vector:")
    a = Vector([1., 2., 3.])
    print(f"Vector a: \n{str(a)}")
    b = Vector([4., 5., 6.])
    print(f"Vector b: \n{str(b)}")
    c = a + b
    print(f"Vector c = a + b: \n{str(c)}")

    print("Testing Subtraction for vector:")
    c = a - b
    print(f"Vector c = a - b: \n{str(c)}")

    print("Testing Scalar Multiplication for vector:")
    c = a * 3.
    print(f"Vector c = a * 3: \n{str(c)}")


def test01():
    print("Testing linear combination")
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
    print("Testing linear interpolation")
    print("\nFloats: ")
    print(linear_interpolation(0., 1., 0.))
    print(linear_interpolation(0., 1., 1.))
    print(linear_interpolation(0., 1., 0.5))
    print(linear_interpolation(21., 42., 0.3))

    print("\nVectors: ")
    v1 = Vector([2., 1.])
    v2 = Vector([4., 2.])
    print(linear_interpolation(v1, v2, 0.3))


def test03():
    print("Testing dot product")
    v1 = Vector([0., 0.])
    v2 = Vector([1., 1.])
    v3 = Vector([-1., 6.])
    v4 = Vector([3., 2.])
    print(v1 * v2)
    print(v2 * v2)
    print(v3 * v4)

def test04():
    print("Testing norm")
    v1 = Vector([0., 0., 0.])
    v2 = Vector([1., 2., 3.])
    v3 = Vector([-1., 2.])
    print("v1: " + str(v1))
    print(f"norm: {v1.norm()} norm_1: {v1.norm_1()} norm_inf: {v1.norm_inf()}")
    print("v2: " + str(v2))
    print(f"norm: {v2.norm()} norm_1: {v2.norm_1()} norm_inf: {v2.norm_inf()}")
    print("v3: " + str(v3))
    print(f"norm: {v3.norm()} norm_1: {v3.norm_1()} norm_inf: {v3.norm_inf()}")

def test05():
    pass

def main():
    test04()
    return


if __name__ == '__main__':
    main()
