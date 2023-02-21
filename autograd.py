
class Param:
    def __init__(self, val):
        self.val = val
        self.grad = 0
    def backward(self, grad):
        self.grad += grad

class Branch:
    def __init__(self, val, backward_fn):
        self.val = val
        self.backward = backward_fn

def mul(p1, p2):
    def backward(g = 1):
        p1.backward(g * p2.val)
        p2.backward(g * p1.val)
    return Branch(p1.val * p2.val, backward)

def test_mul():
    x = Param(3)
    y = Param(4)
    z = mul(x, y)
    t = mul(z, z)
    t.backward()
    assert x.grad == 2 * x.val * y.val**2
    assert y.grad == 2 * y.val * x.val**2

def add(p1, p2):
    def backward(g = 1):
        p1.backward(g)
        p2.backward(g)
    return Branch(p1.val + p2.val, backward)

def test_add():
    v0, v1, v2, v3 = [Param(i) for i in range(2, 6)]
    u = add(add(add(v0, v1), v2), v3)
    u.backward()
    assert v0.grad == v1.grad == v2.grad == v3.grad == 1

Param.__add__ = add
Param.__mul__ = mul
Branch.__add__ = add
Branch.__mul__ = mul

def dot(us, vs):
    return sum((u * v for u, v in zip(us, vs)), start=Param(0))

def test_dot():
    us = [Param(i) for i in range(10)]
    vs = [Param(i) for i in range(10)]
    z = dot(us, vs)
    assert z.val == sum(u.val * v.val for u, v in zip(us, vs))
    z.backward()
    for u, v in zip(us, vs):
        assert u.grad == v.val
        assert v.grad == u.val

def mv_dot(matrix, vec):
    return [dot(row, vec) for row in matrix]

def relu(p):
    def backward(g = 1):
        if p.val > 0:
            p.backward(g)
    return Branch(max(p.val, 0), backward)

def test_relu():
    x = Param(3)
    y = Param(-3)
    z = relu(x) + relu(y)
    z.backward()
    assert x.grad == 1
    assert y.grad == 0

import random
def linear(cols, rows):
    return [[Param(random.random()-.5) for _ in range(cols)] for _ in range(rows)]

def bias(n):
    return [Param(random.random()-.5) for _ in range(n)]

def vec_add(us, vs):
    return [u + v for u, v in zip(us, vs)]

def vrelu(vs):
    return [relu(v) for v in vs]

def main():
    from sklearn.datasets import make_moons, make_blobs
    X_moon, y_moon = make_moons(n_samples=100, noise=0.1)
    Xy = list(zip(X_moon.tolist(), y_moon.tolist()))

    M1, B1 = linear(2, 16), bias(16)
    M2, B2 = linear(16, 16), bias(16)
    M3, B3 = linear(16, 1), bias(1)
    params = [p for mat in [M1, M2, M3] for row in mat for p in row]
    params += [p for bias in [B1, B2, B3] for p in bias]

    def compute(inp):
        h1 = vrelu(vec_add(mv_dot(M1, inp), B1))
        h2 = vrelu(vec_add(mv_dot(M2, h1), B2))
        out = vec_add(mv_dot(M3, h2), B3)
        return out[0]

    Xy = list(zip(X_moon.tolist(), y_moon.tolist()))
    random.shuffle(Xy)
    for epoch in range(1000):
        total = 0
        correct = 0
        for Xi, yi in Xy:
            out = compute([Param(xi) for xi in Xi])
            # Compute loss
            diff = (out + Param(-yi))
            loss = diff * diff
            total += loss.val
            correct += bool(out.val > .5) == yi
            # Backprop
            for p in params:
                p.grad = 0
            loss.backward()
            # Gradient descent
            for p in params:
                p.val -= 1e-3 * p.grad
        print(f'Epoch: {epoch}, Loss: {total/len(Xy)}, Acc: {correct/len(Xy)}')

    import matplotlib.pyplot as plt
    import numpy as np
    X, y = X_moon, y_moon
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Param, xrow)) for xrow in Xmesh]
    Z = np.array([compute(inp).val for inp in inputs])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

if __name__ == '__main__':
    main()
