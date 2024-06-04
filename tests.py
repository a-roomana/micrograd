import unittest

import torch

from micrograd.engine import Value


class ValueTest(unittest.TestCase):
    def test_add(self):
        av = Value(2.0, label="a")
        bv = Value(1.0, label="b")
        cv = av + bv
        cv.backward_children()

        self.assertEqual(cv.data, 3.0)
        self.assertEqual(av.grad, 1.0)
        self.assertEqual(bv.grad, 1.0)

        at = torch.tensor([2.0], requires_grad=True)
        bt = torch.tensor([1.0], requires_grad=True)
        ct = at + bt
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)
        self.assertEqual(bt.grad, bv.grad)

    def test_mul(self):
        av = Value(2.0, label="a")
        bv = Value(3.0, label="b")
        cv = av * bv
        cv.backward_children()

        self.assertEqual(cv.data, cv.data)
        self.assertEqual(av.grad, av.grad)
        self.assertEqual(bv.grad, bv.grad)

        at = torch.tensor([2.0], requires_grad=True)
        bt = torch.tensor([3.0], requires_grad=True)
        ct = at * bt
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)
        self.assertEqual(bt.grad, bv.grad)

    def test_pow(self):
        av = Value(2.0, label="a")
        bv = Value(3.0, label="b")
        cv = av**bv
        cv.backward_children()

        self.assertEqual(cv.data, 8.0)
        self.assertEqual(av.grad, 12.0)
        self.assertEqual(bv.grad, 5.545177444479562)

        at = torch.tensor([2.0], requires_grad=True)
        bt = torch.tensor([3.0], requires_grad=True)
        ct = at**bt
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)
        self.assertEqual(bt.grad, bv.grad)

    def test_neg(self):
        av = Value(2.0, label="a")
        cv = -av
        cv.backward_children()

        self.assertEqual(cv.data, -2.0)
        self.assertEqual(av.grad, -1.0)

        at = torch.tensor([2.0], requires_grad=True)
        ct = -at
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)

    def test_div(self):
        av = Value(8.0, label="a")
        bv = Value(2.0, label="b")
        cv = av / bv
        cv.backward_children()

        self.assertEqual(cv.data, 4.0)
        self.assertEqual(av.grad, 0.5)
        self.assertEqual(bv.grad, -4.0)

        at = torch.tensor([8.0], requires_grad=True)
        bt = torch.tensor([2.0], requires_grad=True)
        ct = at / bt
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)
        self.assertEqual(bt.grad, bv.grad)

    def test_ln(self):
        av = Value(2.0, label="a")
        cv = av.ln()
        cv.backward_children()

        self.assertEqual(cv.data, 0.6931471805599453)
        self.assertEqual(av.grad, 0.5)

        at = torch.tensor([2.0], requires_grad=True)
        ct = at.log()
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)

    def test_tanh(self):
        av = Value(0.0, label="a")
        cv = av.tanh()
        cv.backward_children()

        self.assertEqual(cv.data, 0.0)
        self.assertEqual(av.grad, 1.0)

        at = torch.tensor([0.0], requires_grad=True)
        ct = at.tanh()
        ct.backward()

        self.assertEqual(ct.data, cv.data)
        self.assertEqual(at.grad, av.grad)
