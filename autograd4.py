import numpy as np
import tqdm

# Alternative tensor implementation that allows computing higher derivatives.
# Slower than autograd3 though, about half as fast.
class Tensor:
    def __init__(self, val, children=(), push_grad=lambda g:None):
        self.val = val if isinstance(val, np.ndarray) else np.array([[val]])
        self.grad = None
        self.children = children
        self.push_grad = push_grad

    def backward(self, v=np.array([[1]])):
        self.zero_grad()
        self.grad = Tensor(v)
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

    def zero_grad(self):
        self.grad = Tensor(np.zeros(self.val.shape))
        for child in self.children:
            child.zero_grad()

    @staticmethod
    def broadcast_tensors(*tensors):
        shape = np.broadcast_shapes(*(t.val.shape for t in tensors))
        return [t.broadcast_to(shape) for t in tensors]

    def broadcast_to(self, shape):
        t = tuple(i for i, s in enumerate(self.val.shape) if s == 1)
        return self.apply(np.broadcast_to(self.val, shape), lambda g: g.sum(axis=t))

    def __add__(p1, p2):
        p1, p2 = Tensor.broadcast_tensors(p1, p2)
        def push_grad(grad):
            p1.grad += grad
            p2.grad += grad
        return Tensor(p1.val + p2.val, [p1, p2], push_grad)

    def __matmul__(p1, p2):
        def push_grad(grad):
            p1.grad += grad @ p2.T
            # If we wanted to compute the per-batch grad-norms, we could do this:
            # ib,jb->ijb then take the norms of the outer products and sum to size b.
            # In practice the norm of an outer product is just the product of the norms, so
            # p1.gnorms += grad.pow(2).sum(axis=0) * p2.pow(2).sum(axis=0)
            # That's cheap enough!
            p2.grad += p1.T @ grad
        return Tensor(p1.val @ p2.val, [p1, p2], push_grad)

    def __mul__(p1, p2):
        p1, p2 = Tensor.broadcast_tensors(p1, p2)
        def push_grad(grad):
            p1.grad += p2 * grad
            p2.grad += p1 * grad
        return Tensor(p1.val * p2.val, [p1, p2], push_grad)

    def apply(self, new_val, vjp):
        def push_grad(grad):
            self.grad += vjp(grad)
        return Tensor(new_val, [self], push_grad)

    def __neg__(self): return self.apply(-self.val, lambda g: -g)
    def __sub__(p1, p2): return p1 + -p2
    def __truediv__(p1, p2): return p1 * p2.pow(-1)
    def relu(self): return self.apply(np.maximum(self.val, 0), lambda g: Tensor(self.val > 0) * g)
    def pow(self, n): return self.apply(self.val ** n, lambda g: Tensor(n) * self.pow(n - 1) * g)
    def log(self): return self.apply(np.log(self.val), lambda g: g / self)
    def abs(self): return self.apply(np.abs(self.val), lambda g: Tensor(np.sign(self.val)) * g)
    def exp(self): return self.apply(np.exp(self.val), lambda g: self.exp() * g)
    def logsumexp(self, axis=0): return self.exp().sum(axis=axis).log()
    def sum(self, axis=0): return self.apply(self.val.sum(axis=axis, keepdims=True), lambda g: g)
    def norm(self, axis, norm=2): return self.abs().pow(norm).sum(axis=axis).pow(1/norm)

    def expand(self, labels, d):
        o, b = self.val.shape
        assert o == 1
        new_val = np.zeros((d, b))
        new_val[labels, range(b)] = self.val[0]
        return self.apply(new_val, lambda g: g.select(labels))

    def select(self, labels):
        d, b = self.val.shape
        return self.apply(self.val[labels, range(b)][None], lambda g: g.expand(labels, d))

    def label_cross_entropy_on_logits(self, labels):
        return (-self.select(labels) + self.logsumexp(axis=0)).sum(axis=1)

    ## Not needed

    @property
    def T(self): return self.apply(self.val.T, lambda g: g.T)
    def softmax(self, axis=0): return self.exp() / self.exp().sum(axis=axis)
    # also known as the multiclass cross-entropy
    def label_cross_entropy(q, labels): return -q.select(labels).log().sum(axis=1)
    def dot(self, other, axis=0): return (self * other).sum(axis=axis)
    # Returns -sum_i(p_i * log(q_i))
    def cross_entropy(q, p): return -p.dot(q.log()).sum(axis=1)

def test_double_grad():
    x0, N = 2, 5
    x = Tensor(x0)
    y = x.pow(N)
    c = 1
    for n in range(N, 0, -1):
        y.backward()
        c *= n
        assert x.grad.val == c * x0 ** (n-1)
        y = x.grad

def test_mul():
    np.random.seed(0)
    x = Tensor(np.random.randn(4,3))
    y = Tensor(np.random.randn(4,3))
    z = (x * y).sum(axis=0).sum(axis=1)
    z.backward()
    np.testing.assert_allclose(x.grad.val, y.val)
    np.testing.assert_allclose(y.grad.val, x.val)
    # Test broadcasting
    x = Tensor(np.arange(12).reshape(4,3))
    y = Tensor(np.arange(4).reshape(4,1))
    z = (x * y).sum(axis=0).sum(axis=1)
    z.backward()
    np.testing.assert_allclose(*np.broadcast_arrays(x.grad.val, y.val))
    np.testing.assert_allclose(*np.broadcast_arrays(y.grad.val, x.val.sum(axis=1, keepdims=True)))
    # Broadcast on first coordinate
    x = Tensor(np.random.randn(4,3))
    y = Tensor(np.random.randn(1,3))
    z = (x * y).sum(axis=0).sum(axis=1)
    z.backward()
    np.testing.assert_allclose(*np.broadcast_arrays(x.grad.val, y.val))
    np.testing.assert_allclose(*np.broadcast_arrays(y.grad.val, x.val.sum(axis=0, keepdims=True)))

def jacobian_test(x0, f, eps=1e-6):
    J0 = numerical_jacobian(x0, f, eps)
    J1 = jacobian(x0, f).val
    np.testing.assert_allclose(J0, J1)

def test_mean():
    np.random.seed(0)
    #x = np.random.randn(4,1)
    x = Tensor(np.arange(4, dtype=float)[:, None])
    y = (x / x.sum(axis=0)).sum(axis=0)
    np.testing.assert_allclose(y.val, np.array([[1]]))
    y.backward()
    np.testing.assert_allclose(x.grad.val, np.zeros((4,1)))

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
    np.testing.assert_allclose(x0.grad.val, x1.grad.val)

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
    np.testing.assert_allclose(x0.grad.val, x1.grad.val)

def jacobian(x, f):
    d0, o = x.val.shape
    assert o == 1, "Can't compute Jacobians of batched input"
    x1 = x.broadcast_to((d0,d0))
    y1 = f(x1)
    d1, _ = y1.val.shape
    y1.backward(v = np.eye(d1))
    return x1.grad.T

def numerical_jacobian(x, f, eps=1e-4):
    d0, o = x.val.shape
    assert o == 1, "Can't compute batch Jacobians, sorry"
    y0 = f(x - Tensor(np.eye(d0) * eps)).val
    y1 = f(x + Tensor(np.eye(d0) * eps)).val
    return (y1 - y0) / (2 * eps)

def hessian(x, f):
    d0, o = x.val.shape
    assert o == 1
    xb = x.broadcast_to((d0, d0))
    y = f(xb)
    assert y.val.shape == (1, d0)
    y.backward()
    g = xb.grad
    g.backward(v=np.eye(d0))
    return xb.grad

def numerical_hessian(x, f, eps=1e-4):
    d0, o = x.val.shape
    assert o == 1
    H = np.zeros((d0, d0))
    for i, v in enumerate(eps * np.eye(d0)):
        e = Tensor(v[:, None])
        g0 = numerical_jacobian(x - e, f, eps=eps)
        g1 = numerical_jacobian(x + e, f, eps=eps)
        H[i, :] = (g1 - g0).flatten() / (2 * eps)
    return H

def test_hessian():
    x = Tensor(np.array([1.,2.,3.])[:, None])
    f = lambda x: x.softmax().dot(Tensor(np.array([1.,2.,3.])[:, None]))
    H = hessian(x, f)
    H1 = numerical_hessian(x, f)
    np.testing.assert_allclose(H.val, H1, rtol=1e-6, atol=1e-8)

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
            #x = x / x.norm(axis=0)
            #x = layer(x / x.norm(axis=0))
        return self.layers[-1](x)

    def parameters(self):
        return [l.M for l in self.layers] + [l.B for l in self.layers]

def mnist():
    from sklearn import datasets, utils, preprocessing
    from sklearn.model_selection import train_test_split
    X, y = datasets.load_digits(return_X_y=True)
    X = X.reshape(-1, 64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #from sklearn.datasets import fetch_openml
    #cifar10 = fetch_openml('CIFAR_10', version=1)
    #X, y = cifar10['data'], cifar10['target']
    #X = X.to_numpy().reshape(-1, 3*32*32)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Seems we don't really need normalization
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    batch_size = 32
    def batches(X, y):
        # X, y = utils.shuffle(X, y)
        for i in range(0, len(X), batch_size):
            X_batch = Tensor(X[i:i+batch_size].T) # We use column-vector format
            y_batch = y[i:i+batch_size]
            yield X_batch, y_batch

    # X = X.reshape(-1, 64)
    # X -= X.mean(axis=1, keepdims=True)
    # X /= np.linalg.norm(X, axis=1, keepdims=True)
    # X, y = utils.shuffle(X, y)

    mlp = MLP([64] + [256]*4 + [10])

    def step(pb, epoch):
        total_loss, correct = 0, 0
        for X_batch, y_batch in batches(X_train, y_train):
            out = mlp(X_batch)
            # Compute loss
            #loss = out.label_cross_entropy_on_logits(y_batch)
            loss = -(out / out.norm(axis=0)).select(y_batch).sum(axis=1)
            # Measure accuracy
            total_loss += loss.val
            correct += (out.val.argmax(axis=0) == y_batch).sum()
            # Backprop
            loss.backward()
            # Gradient descent
            for p in mlp.parameters():
                p.val -= 1e-2 * p.grad.val / batch_size
        test_acc = 0
        for X_batch, y_batch in batches(X_test, y_test):
            test_acc += (mlp(X_batch).val.argmax(axis=0) == y_batch).sum()
        pb.set_description(
                f'Loss: {float(total_loss)/len(X_train):.3}, '
                f'Acc: {correct/len(X_train):.3}, '
                f'Test: {test_acc/len(X_test):.3}')

    with tqdm.tqdm(range(100)) as pb:
        for i in pb:
            step(pb, i)



if __name__ == '__main__':
    mnist()
