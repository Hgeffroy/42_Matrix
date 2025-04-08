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
    # vec2 = linear_combination([v1, v2], [10., -2.])
    # print(f'{vec2}')




def main():
    test01()
    return


if __name__ == '__main__':
    main()
