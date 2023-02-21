
class Param:
    def __init__(self, val):
        self.val = val
        self.grad = 0
        self.children = []

    def push_grad(self):
        pass

class Branch:
    def __init__(self, val, children):
        self.val = val
        self.grad = 0
        self.children = children

def backward(root, top=True):
    if top:
        root.grad = 1
    root.push_grad()
    root.grad = 0
    for child in root.children:
        if isinstance(child, Branch):
            backward(child, top=False)

def fast_backward(root):
    def walk(node, source):
        # If I want to batch gradients, I probably need
        # a different way to zero grads. Or just not do it on leaves.
        node.grad = 0
        if not hasattr(node, 'sources'):
            node.sources = 1
        else:
            node.sources += 1
            return
        for child in node.children:
            walk(child, node)
    walk(root, None)

    root.grad = 1
    todo = [root]
    while todo:
        node = todo.pop()
        node.push_grad()
        for child in node.children:
            child.sources -= 1
            if not child.sources:
                todo.append(child)


def backward_kuhn(self, v=np.array([[1]])):
    self.grad = v
    sources = {}
    def walk(node, source):
        if node not in sources:
            sources[node] = 0
            for child in node.children:
                walk(child, node)
        sources[node] += 1
    walk(self, None)
    todo = [self]
    while todo:
        node = todo.pop()
        node.push_grad(node.grad)
        for child in node.children:
            sources[child] -= 1
            if not sources[child]:
                todo.append(child)


def test_fast():
    x = Param(3)
    y = x * x # x^2
    z = y * y # x^4
    t = z * z # x^8
    fast_backward(t)
    assert t.grad == 1
    assert z.grad == 2 * 3**4
    assert y.grad == 4 * 3**6
    assert x.grad == 8 * 3**7

def mul(p1, p2):
    node = Branch(p1.val * p2.val, [p1, p2])
    def push_grad():
        p1.grad += node.grad * p2.val
        p2.grad += node.grad * p1.val
    node.push_grad = push_grad
    return node

def test_mul():
    x = Param(3)
    y = Param(4)
    z = mul(x, y)
    t = mul(z, z)
    fast_backward(t)
    assert x.grad == 2 * x.val * y.val**2
    assert y.grad == 2 * y.val * x.val**2

def add(p1, p2):
    node = Branch(p1.val + p2.val, [p1, p2])
    def push_grad():
        p1.grad += node.grad
        p2.grad += node.grad
    node.push_grad = push_grad
    return node

def test_add():
    v0, v1, v2, v3 = [Param(i) for i in range(2, 6)]
    u = add(add(add(v0, v1), v2), v3)
    fast_backward(u)
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
    fast_backward(z)
    for u, v in zip(us, vs):
        assert u.grad == v.val
        assert v.grad == u.val

def mv_dot(matrix, vec):
    return [dot(row, vec) for row in matrix]

def relu(p):
    node = Branch(max(p.val, 0), [p])
    def push_grad():
        if p.val > 0:
            p.grad += node.grad
    node.push_grad = push_grad
    return node

def test_relu():
    x = Param(3)
    y = Param(-3)
    z = relu(x) + relu(y)
    fast_backward(z)
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
        vec = [Param(x) for x in inp]
        h1 = vrelu(vec_add(mv_dot(M1, vec), B1))
        h2 = vrelu(vec_add(mv_dot(M2, h1), B2))
        out = vec_add(mv_dot(M3, h2), B3)
        return out[0]

    Xy = list(zip(X_moon.tolist(), y_moon.tolist()))
    random.shuffle(Xy)
    def step(epoch):
        total = 0
        correct = 0
        for Xi, yi in Xy:
            out = compute(Xi)
            # Compute loss
            diff = (out + Param(-yi))
            loss = diff * diff
            total += loss.val
            correct += bool(out.val > .5) == yi
            # Backprop
            fast_backward(loss)
            # Gradient descent
            for p in params:
                p.val -= 1e-3 * p.grad
        print(f'Epoch: {epoch}, Loss: {total/len(Xy)}, Acc: {correct/len(Xy)}')

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    def animate(i):
        step(i)
        X, y = X_moon, y_moon
        h = 0.25

        # Create a meshgrid of points to evaluate the function over
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        import numpy as np
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()]

        # Compute function values for each point in the meshgrid
        Z = np.array([compute(x).val for x in Xmesh])
        Z = Z.reshape(xx.shape)

        # Plot the contour and scatter plots
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    for i in range(500):
        animate(i)
        plt.savefig(f'plots/moons_{i:03d}.png', dpi=96)
        plt.gca()

    #ani = animation.FuncAnimation(fig, animate, interval=1)
    #plt.show()

if __name__ == '__main__':
    main()
