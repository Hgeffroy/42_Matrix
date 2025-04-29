#!/bin/python3

from Matrixlib import *


class TestMatrixlib:
    i = Vector([1., 0., 0.])
    j = Vector([0., 1., 0.])
    k = Vector([0., 0., 1.])
    v0 = Vector([0., 0., 0.])
    v1 = Vector([1., 2., 3.])
    v2 = Vector([4., 5., 6.])
    v3 = Vector([0., 10., -100.])
    v4 = Vector([4., 2., -3.])
    v5 = Vector([-2., -5., 16.])

    v10 = Vector([0., 0.])
    v11 = Vector([4., 2.])
    v12 = Vector([2., 1.])
    v13 = Vector([1., 1.])
    v14 = Vector([-1., 6.])
    v15 = Vector([3., 2.])
    v16 = Vector([1., 0.])
    v17 = Vector([0., 1.])
    v18 = Vector([-1., 1.])
    v19 = Vector([-1., -2.])

    m1 = Matrix([Vector([-2., -8., 4.]),
                 Vector([1., -23., 4.])])

    m20 = Matrix([Vector([1., 0.]), Vector([0., 1.])])
    m21 = Matrix([Vector([2., 1.]), Vector([3., 4.])])
    m22 = Matrix([Vector([20., 10.]), Vector([30., 40.])])
    m23 = Matrix([Vector([2., -2.]), Vector([-2., 2.])])
    m24 = Matrix([Vector([2., 1.]), Vector([4., 2.])])
    m25 = Matrix([Vector([3., -5.]), Vector([6., 8.])])

    m30 = Matrix([Vector([1., 0., 0.]),
                  Vector([0., 1., 0.]),
                  Vector([0., 0., 1.])])
    m31 = Matrix([Vector([2., -5., 0.]),
                  Vector([4., 3., 7.]),
                  Vector([-2., 3., 4.])])
    m32 = Matrix([Vector([-2., -8., 4.]),
                  Vector([1., -23., 4.]),
                  Vector([0., 6., 4.])])

    def test00(self):
        assert self.v1 + self.v2 == Vector([5., 7., 9.])
        assert self.v1 - self.v2 == Vector([-3., -3., -3.])
        assert self.v1 * 3. == Vector([15., 21., 27.])

    def test01(self):
        assert linear_combination([self.i, self.j, self.k], [10., -2., 0.5]) == Vector([10., -2., 0.5])
        assert linear_combination([self.v1, self.v3], [10., -2.]) == Vector([10., 0., 230.])

    def test02(self):
        assert linear_interpolation(self.v12, self.v11, 0.3) == Vector([2.6, 1.3])
        assert linear_interpolation(self.m21, self.m22, 0.5) == Matrix([Vector([11., 5.5]), Vector([16.5, 22.])])

    def test03(self):
        assert self.v10.dot(self.v13) == 0.0
        assert self.v13.dot(self.v13) == 2.0
        assert self.v14.dot(self.v15) == 9.0

    def test04(self):
        assert self.v0.norm_1() == 0.0
        assert self.v0.norm() == 0.0
        assert self.v0.norm_inf() == 0.0

        assert self.v1.norm_1() == 6.0
        assert round(self.v1.norm(), 8) == 3.74165739
        assert self.v1.norm_inf() == 3.0

        assert self.v19.norm_1() == 3.0
        assert round(self.v19.norm(), 9) == 2.236067977
        assert self.v19.norm_inf() == 2.0

    def test05(self):
        assert round(angle_cos(self.v16, self.v16), 10) == 1.0
        assert round(angle_cos(self.v16, self.v17), 10) == 0.0
        assert round(angle_cos(self.v18, self.v18 * -1), 10) == -1.0
        assert round(angle_cos(self.v12, self.v11), 10) == 1.0
        assert round(angle_cos(self.v1, self.v2), 9) == 0.974631846

    def test06(self):
        assert cross_product(self.k, self.i) == Vector([0., 1., 0.])
        assert cross_product(self.v1, self.v2) == Vector([-3., 6., -3.])
        assert cross_product(self.v4, self.v5) == Vector([17., -58., -16.])

    def test07(self):
        assert self.m20.mul_vec(self.v11) == Vector([4., 2.,])
        assert (self.m20 * 2).mul_vec(self.v11) == Vector([8., 4.,])
        assert self.m23.mul_vec(self.v11) == Vector([4., -4.,])
        assert self.m20.mul_mat(self.m20) == self.m20
        assert self.m20.mul_mat(self.m24) == self.m24
        assert self.m24.mul_mat(self.m25) == Matrix([Vector([-14., -7.]), Vector([44., 22.])]) # Sujet faux ici !

    def test08(self):
        assert self.m20.trace() == 2.0
        assert self.m31.trace() == 9.0
        assert self.m32.trace() == -21.0

    def test09(self):
        assert self.m30.transpose() == self.m30
        assert self.m31.transpose() == Matrix([Vector([2., 4., -2.]),
                                               Vector([-5., 3., 3.]),
                                               Vector([0., 7., 4.])])
        assert self.m1.transpose() == Matrix([Vector([-2., 1.]),
                                              Vector([-8., -23.]),
                                              Vector([4., 4.])])
    #
    # def test10(self):
    #     m1 = Matrix([Vector([1., 0., 0.]), Vector([0., 1., 0.]), Vector([0., 0., 1.])])
    #     m2 = Matrix([Vector([1., 3.]), Vector([2., 4.])])
    #     m3 = Matrix([Vector([1., 2.]), Vector([2., 4.])])
    #     m4 = Matrix([Vector([2., -5., 0.]),
    #                  Vector([4., 3., 7.]),
    #                  Vector([-2., 3., 4.])])
    #     m5 = Matrix([Vector([8., 5., -2., 4., 28.]),
    #                  Vector([4., 2.5, 20., 4., -4.]),
    #                  Vector([8., 5., 1., 4., 17.])])
    #
    #
    # def test11(self):
    #     m1 = Matrix([Vector([2., 0., 0.]), Vector([0., 2., 0.]), Vector([0., 0., 2.])])
    #     m2 = Matrix([Vector([8., 4., 7.]), Vector([5., 7., 6.]), Vector([-2., 20., 1.])])
    #     m3 = Matrix([Vector([8., 4., 8., 28.]), Vector([5., 2.5, 5., -4.]), Vector([-2., 20., 1., 17.]), Vector([4., 4., 4., 1.])])
    #
    #
    # def test12(self):
    #     m1 = identity_matrix(3, float)
    #     m2 = m1 * 2
    #     m3 = Matrix([Vector([8., 5., -2]), Vector([4., 7., 20.]), Vector([7., 6., 1.])])
    #     m4 = Matrix([Vector([1., 0., 1]), Vector([0., 1., 0.]), Vector([0., 0., 1.])])
    #
    #
    # def test13(self):
    #     m1 = identity_matrix(3, float)
    #     m2 = Matrix([Vector([1., 2., 0., 0.]), Vector([2., 4., 0., 0.]), Vector([-1., 2., 1., 1.])])
    #     m3 = Matrix([Vector([8., 5., -2.]), Vector([4., 7., 20.]), Vector([7., 6., 1.]), Vector([21., 18., 7.])])

