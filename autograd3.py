import numpy as np

class Tensor:
    def __init__(self, val, children=(), push_grad=lambda g:None):
        self.val = val if isinstance(val, np.ndarray) else np.array([[val]])
        self.grad = 0
        self.children = children
        self.push_grad = push_grad

    def backward(self, v=np.array([[1]])):
        self.grad = v
        visited, order = set(), []
        def walk(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    walk(child)
                order.append(node)
        walk(self)
        for node in reversed(order):
            node.push_grad(node.grad)

    @staticmethod
    def broadcast_tensors(*tensors):
        shape = np.broadcast_shapes(*(t.val.shape for t in tensors))
        return [t.broadcast_to(shape) for t in tensors]

    def broadcast_to(self, shape):
        t = tuple(i for i, s in enumerate(self.val.shape) if s == 1)
        return self.apply(np.broadcast_to(self.val, shape), lambda g: g.sum(axis=t, keepdims=True))

    def __add__(p1, p2):
        p1, p2 = Tensor.broadcast_tensors(p1, p2)
        def push_grad(grad):
            p1.grad += grad
            p2.grad += grad
        return Tensor(p1.val + p2.val, [p1, p2], push_grad)

    def __matmul__(p1, p2):
        def push_grad(grad):
            p1.grad += grad @ p2.val.T
            p2.grad += p1.val.T @ grad
        return Tensor(p1.val @ p2.val, [p1, p2], push_grad)

    def __mul__(p1, p2):
        p1, p2 = Tensor.broadcast_tensors(p1, p2)
        def push_grad(grad):
            p1.grad += p2.val * grad
            p2.grad += p1.val * grad
        return Tensor(p1.val * p2.val, [p1, p2], push_grad)

    def __truediv__(p1, p2):
        return p1 * p2.pow(-1)

    def apply(self, new_val, jvp):
        # Helper method for unitary functions
        def push_grad(grad):
            self.grad += jvp(grad)
        return Tensor(new_val, [self], push_grad)

    @property
    def T(self):
        return self.apply(self.val.T, lambda g: g.T)

    def relu(self):
        return self.apply(np.maximum(self.val, 0), lambda g: (self.val > 0) * g)

    def pow(self, n):
        return self.apply(self.val ** n, lambda g: n * self.val ** (n-1) * g)

    def log(self):
        return self.apply(np.log(self.val), lambda g: g / self.val)

    def abs(self):
        return self.apply(np.abs(self.val), lambda g: np.sign(self.val) * g)

    def exp(self):
        e = np.exp(self.val)
        return self.apply(e, lambda g: e * g)

    def softmax2(self, axis=0):
        # Jacobian = I - pp^T
        # No, actually not, it's diag(p) - pp^T. I found this bug with the Jac test
        exps = np.exp(self.val - np.max(self.val, axis=axis, keepdims=True))
        p = exps / exps.sum(axis=axis, keepdims=True)
        return self.apply(p, lambda g: p*g - np.einsum('db,eb,eb->db', p, p, g))

    def softmax(self, axis=0):
        return self.exp() / self.exp().sum(axis=axis)

    def logsumexp2(self, axis=0):
        return self.exp().sum(axis=axis).log()

    def logsumexp(self, axis=0):
        top = np.max(self.val, axis=axis, keepdims=True)
        exps = np.exp(self.val - top)
        expsum = exps.sum(axis=axis, keepdims=True)
        return self.apply(top + np.log(expsum), lambda g: (exps / expsum) * g)

    def __neg__(self):
        return self.apply(-self.val, lambda g: -g)

    def sum(self, axis=0):
        return self.apply(self.val.sum(axis=axis, keepdims=True),
                          lambda g: np.broadcast_to(g, self.val.shape))

    def dot(self, other, axis=0):
        return (self * other).sum(axis=axis)

    def cross_entropy(q, p):
        # Returns -sum_i(p_i * log(q_i))
        return -p.dot(q.log()).sum(axis=1)

    def select(self, labels):
        d, b = self.val.shape
        def jvp(grad):
            expanded = np.zeros((d, b))
            expanded[labels, range(b)] = grad[0]
            return expanded
        return self.apply(self.val[labels, range(b)][None], jvp)

    def label_cross_entropy(q, labels):
        # also known as the multiclass cross-entropy
        return -q.select(labels).log().sum(axis=1)

    def norm(self, axis, norm=2):
        if norm == 'exp':
            return self.logsumexp(axis=axis)
        return self.abs().pow(norm).sum(axis=axis).pow(1/norm)

    def label_cross_entropy_on_logits(self, labels, norm=None):
        norm_term = self.norm(axis=0, norm='exp' if norm is None else norm)
        return (-self.select(labels) + norm_term).sum(axis=1)

def test_mul():
    np.random.seed(0)
    x = Tensor(np.random.randn(4,3))
    y = Tensor(np.random.randn(4,3))
    z = (x * y).sum(axis=0).sum(axis=1)
    z.backward()
    np.testing.assert_allclose(x.grad, y.val)
    np.testing.assert_allclose(y.grad, x.val)
    # Test broadcasting
    x = Tensor(np.arange(12).reshape(4,3))
    y = Tensor(np.arange(4).reshape(4,1))
    z = (x * y).sum(axis=0).sum(axis=1)
    z.backward()
    np.testing.assert_allclose(*np.broadcast_arrays(x.grad, y.val))
    np.testing.assert_allclose(*np.broadcast_arrays(y.grad, x.val.sum(axis=1, keepdims=True)))
    # Broadcast on first coordinate
    x = Tensor(np.random.randn(4,3))
    y = Tensor(np.random.randn(1,3))
    z = (x * y).sum(axis=0).sum(axis=1)
    z.backward()
    np.testing.assert_allclose(*np.broadcast_arrays(x.grad, y.val))
    np.testing.assert_allclose(*np.broadcast_arrays(y.grad, x.val.sum(axis=0, keepdims=True)))

def jacobian_test(x0, f, eps=1e-8):
    d0, o = x0.val.shape
    assert o == 1
    y0 = f(x0)
    d1, o = y0.val.shape
    assert o == 1
    x1 = x0 + Tensor(np.eye(d0) * eps)
    y1 = f(x1)
    J0 = (y1 + -y0).val / eps
    # The forward here is wrong, but not enough to matter.
    y1.backward(v = np.eye(d1))
    J1 = x1.grad
    if not np.allclose(J0, J1):
        print(f'{x0.val=}')
        print(f'{x1.val=}')
        print(f'{y0.val=}')
        print(f'{y1.val=}')
    np.testing.assert_allclose(J0, J1.T, rtol=1e-4, atol=1e-6)

def test_softmax():
    np.random.seed(0)
    #x = np.random.randn(4,1)
    x = np.arange(4)[:, None]
    x0 = Tensor(x)
    x1 = Tensor(x)
    y0 = x0.softmax()
    y1 = x1.softmax2()
    np.testing.assert_allclose(y0.val, y1.val, rtol=1e-4, atol=1e-6)
    y0.sum(axis=0).backward()
    y1.sum(axis=0).backward()
    np.testing.assert_allclose(x0.grad, x1.grad, rtol=1e-4, atol=1e-6)

def test_mean():
    np.random.seed(0)
    #x = np.random.randn(4,1)
    x = Tensor(np.arange(4, dtype=float)[:, None])
    y = (x / x.sum(axis=0)).sum(axis=0)
    np.testing.assert_allclose(y.val, np.array([[1]]))
    y.backward()
    np.testing.assert_allclose(x.grad, np.zeros((4,1)))

def test_sum():
    np.random.seed(0)
    x0 = Tensor(np.arange(1,5,dtype=float).reshape(4,1))
    jacobian_test(x0, lambda x: x + x.sum(axis=0))
    jacobian_test(x0, lambda x: x * x.sum(axis=0))
    jacobian_test(x0, lambda x: x / x.sum(axis=0))

def test_jacobians():
    np.random.seed(0)
    #x0 = Tensor(np.random.randn(4,1))
    x0 = Tensor(np.arange(1, 5, dtype=float)[:, None])
    jacobian_test(x0, lambda x: x.relu())
    jacobian_test(x0, lambda x: x.abs())
    jacobian_test(x0, lambda x: x.pow(2))
    jacobian_test(x0, lambda x: x.pow(-1))
    jacobian_test(x0, lambda x: x.softmax())
    jacobian_test(x0, lambda x: x.softmax2())
    x1 = Tensor(np.abs(np.random.randn(4,1)))
    jacobian_test(x1, lambda x: x.log())

def test_logsumexp():
    np.random.seed(0)
    d, b = 100, 20
    x = np.random.randn(d, b)
    x0, x1 = Tensor(x), Tensor(x)
    y0 = x0.logsumexp(axis=0)
    y1 = x1.exp().sum(axis=0).log()
    np.testing.assert_allclose(y0.val, y1.val)
    y0.backward()
    y1.backward()
    np.testing.assert_allclose(x0.grad, x1.grad)

def test_label_cross_entropy():
    np.random.seed(0)
    d0, d1, b = 4, 4, 1
    x = np.random.randn(d0, b)
    x0 = Tensor(x)
    x1 = Tensor(x)
    labels = np.random.randint(d1, size=(b,))
    c0 = x0.pow(2).softmax().label_cross_entropy(labels)
    c1 = x1.pow(2).label_cross_entropy_on_logits(labels)
    np.testing.assert_allclose(c0.val, c1.val)
    c0.backward()
    c1.backward()
    np.testing.assert_allclose(x0.grad, x1.grad)

class Linear:
    def __init__(self, a, b):
        self.M = Tensor(np.random.randn(b, a) / (a + b)**.5)
        self.B = Tensor(np.random.randn(b, 1) / (b + 1)**.5)

    def __call__(self, x):
        return self.M @ x + self.B

class MLP:
    def __init__(self, sizes):
        self.layers = [Linear(a, b) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

    def parameters(self):
        return [l.M for l in self.layers] + [l.B for l in self.layers]

from numpy.lib.stride_tricks import as_strided

class Convolution:
    def __init__(self, w, h, c1, c2):
        # https://jessicastringham.net/2017/12/31/stride-tricks/
        # https://jessicastringham.net/2018/01/01/einsum/
        self.K = Tensor(np.random.randn(w*h*c1, c2) / (w*h*c1 + c2)**.5)
        self.M = Tensor(np.random.randn(b, a) / (a + b)**.5)
        self.B = Tensor(np.random.randn(b, 1) / (b + 1)**.5)

    def __call__(self, x):
        return self.M @ x + self.B


def mnist():
    from sklearn import datasets, utils, preprocessing
    from sklearn.model_selection import train_test_split
    X, y = datasets.load_digits(return_X_y=True)
    X = X.reshape(-1, 64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    batch_size = 32
    def batches(X, y):
        X, y = utils.shuffle(X, y)
        for i in range(0, len(X), batch_size):
            X_batch = Tensor(X[i:i+batch_size].T) # We use column-vector format
            y_batch = y[i:i+batch_size]
            yield X_batch, y_batch

    # X = X.reshape(-1, 64)
    # X -= X.mean(axis=1, keepdims=True)
    # X /= np.linalg.norm(X, axis=1, keepdims=True)
    # X, y = utils.shuffle(X, y)

    mlp = MLP([64] + [256]*4 + [10])

    def step(epoch):
        total_loss, correct = 0, 0
        for X_batch, y_batch in batches(X_train, y_train):
            out = mlp(X_batch)
            # Compute loss
            #loss = out.softmax().label_cross_entropy(y_batch)
            #loss = out.label_cross_entropy_on_logits(y_batch)
            #loss = out.label_cross_entropy_on_logits(y_batch, norm=2)
            loss = -(out / out.norm(axis=0, norm=2)).select(y_batch).sum(axis=1)
            #loss = -(out / out.norm(axis=0, norm='exp')).select(y_batch).sum(axis=1)
            #loss = out.softmax2().label_cross_entropy(y_batch)
            # Measure accuracy
            total_loss += loss.val
            correct += (out.val.argmax(axis=0) == y_batch).sum()
            # Backprop

            #reg = sum((p.pow(2).sum(axis=0).sum(axis=1) for p in mlp.parameters()), start=Tensor(0))
            #(loss + reg).backward()
            loss.backward()
            # Gradient descent
            for p in mlp.parameters():
                p.val -= 1e-2 * p.grad / batch_size
                p.grad = 0
        test_acc = 0
        for X_batch, y_batch in batches(X_test, y_test):
            out = mlp(X_batch)
            test_acc += (out.val.argmax(axis=0) == y_batch).sum()
        print(f'Epoch: {epoch}, Loss: {float(total_loss)/len(X_train):.3}, Acc: {correct/len(X_train):.3}, Test: {test_acc/len(X_test):.3}')

    for i in range(100):
        step(i)



def moons():
    from sklearn.datasets import make_moons, make_blobs
    X, y = make_moons(n_samples=100, noise=0.1)

    # mlp = MLP([2, 100, 100, 1])
    mlp = MLP([2] + [16]*2 + [1])

    def step(epoch):
        total = 0
        correct = 0
        batch_size = 32
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].T # We use column-vector format
            y_batch = y[i:i+batch_size].reshape(1, -1)
            out = mlp(Tensor(X_batch))
            # Compute loss
            diff = (out + Tensor(-y_batch))
            sum_loss = diff @ diff.T
            # Measure accuracy
            total += sum_loss.val
            correct += ((out.val > .5) == y_batch).sum()
            # Backprop
            sum_loss.backward()
            # Gradient descent
            for p in mlp.parameters():
                p.val -= 1e-2 * p.grad / batch_size
                p.grad = 0
        print(f'Epoch: {epoch}, Loss: {total_loss/len(X)}, Acc: {correct/len(X)}')

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    def plot(i):
        h = 0.25
        # Create a meshgrid of points to evaluate the function over
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()]

        # Compute function values for each point in the meshgrid
        Z = mlp(Tensor(Xmesh.T)).val.T
        #Z = mlp(Tensor(Xmesh.T)).val.T > .5
        Z = Z.reshape(xx.shape)

        # Plot the contour and scatter plots
        plt.contourf(xx, yy, Z, 20, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    def animate(i):
        step(i)
        plot(i)

    #for i in range(500):
    #    animate(i)
        #plt.savefig(f'plots/moons_{i:03d}.png', dpi=96)
        #plt.gca()

    for i in range(500):
        step(i)
    plot(0)


    #ani = animation.FuncAnimation(fig, animate, interval=1)
    plt.show()




if __name__ == '__main__':
    mnist()
