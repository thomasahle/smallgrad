import numpy as np
import tqdm
import itertools

from numpy.lib.stride_tricks import as_strided
# https://jessicastringham.net/2017/12/31/stride-tricks/
# https://agustinus.kristia.de/techblog/2016/07/16/convnet-conv-layer/
def im2win(image, kh, kw):
    (bs, c, ih, iw), s = image.shape, image.strides
    return as_strided(image,
                      shape=(bs, c, ih - kh + 1, iw - kw + 1, kh, kw),
                      strides=(s[0], s[1]) + (s[2], s[3]) * 2)

class Tensor:
    def __init__(self, val, children=(), push_grad=lambda g:None):
        self.val = val if isinstance(val, np.ndarray) else np.array([[val]])
        self.grad = np.zeros(self.val.shape)
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

    def select(self, labels):
        bs, d = self.val.shape
        def push_grad(grad):
            self.grad[range(bs), labels] += grad[0]
        return Tensor(self.val[range(bs), labels][:, None], [self], push_grad)

    def label_cross_entropy_on_logits(self, labels):
        return (-self.select(labels) + self.logsumexp(axis=1)).sum(axis=0)

    def apply(self, new_val, jvp):
        # Helper method for unitary functions
        def push_grad(grad):
            self.grad += jvp(grad)
        return Tensor(new_val, [self], push_grad)

    def __neg__(self): return self.apply(-self.val, lambda g: -g)
    def __sub__(p1, p2): return p1 + -p2
    def __truediv__(p1, p2): return p1 * p2.pow(-1)
    def relu(self): return self.apply(np.maximum(self.val, 0), lambda g: (self.val > 0) * g)
    def pow(self, n): return self.apply(self.val ** n, lambda g: n * self.val ** (n-1) * g)
    def log(self): return self.apply(np.log(self.val), lambda g: g / self.val)
    def abs(self): return self.apply(np.abs(self.val), lambda g: np.sign(self.val) * g)
    def exp(self): return self.apply(np.exp(self.val), lambda g: np.exp(self.val) * g)
    def logsumexp(self, axis): return self.exp().sum(axis=axis).log()
    def sum(self, axis): return self.apply(self.val.sum(axis=axis, keepdims=True), lambda g: g)
    def norm(self, axis, norm=2): return self.abs().pow(norm).sum(axis=axis).pow(1/norm)

    def convolve(self, kernel):
        (kh, kw, c1, c2), (_, _c1, ih, iw) = kernel.val.shape, self.val.shape
        assert c1 == _c1, f'image shape={self.val.shape}, kernel shape={kernel.val.shape}'
        cols = self.im2win(kh, kw).reshape(-1, c1 * kh * kw)
        matmul = cols @ kernel.reshape(-1, c2)
        return matmul.reshape(-1, c2, ih-1, iw-1)

    def im2win(self, kh, kw):
        def push_grad(grad):
            blocks = im2win(self.grad, kh, kw)
            # It doesn't work to just say "blocks += grad", because
            # overlapping values will disappear
            np.add.at(blocks, (slice(None),)*4, grad)
        return Tensor(im2win(self.val, kh, kw), [self], push_grad)

    def reshape(self, *shape):
        return self.apply(self.val.reshape(*shape), lambda g: g.reshape(*self.val.shape))


    ## Not needed

    @property
    def T(self):
        return self.apply(self.val.T, lambda g: g.T)

    def softmax(self, axis):
        return self.exp() / self.exp().sum(axis=axis)

    def label_cross_entropy(q, labels):
        # also known as the multiclass cross-entropy
        return -q.select(labels).log().sum(axis=1)

    def dot(self, other, axis):
        return (self * other).sum(axis=axis)

    def cross_entropy(q, p):
        # Returns -sum_i(p_i * log(q_i))
        return -p.dot(q.log()).sum(axis=1)

    def logsumexp2(self, axis=0):
        top = np.max(self.val, axis=axis, keepdims=True)
        exps = np.exp(self.val - top)
        expsum = exps.sum(axis=axis, keepdims=True)
        return self.apply(top + np.log(expsum), lambda g: (exps / expsum) * g)

    def softmax2(self, axis):
        # Jacobian = I - pp^T
        # No, actually not, it's diag(p) - pp^T. I found this bug with the Jac test
        exps = np.exp(self.val - np.max(self.val, axis=axis, keepdims=True))
        p = exps / exps.sum(axis=axis, keepdims=True)
        return self.apply(p, lambda g: p*g - np.einsum('db,eb,eb->db', p, p, g))

def test_reshape():
    x = Tensor(np.arange(16).reshape(16, 1))
    for size in [(1, 1, 1, 16), (1, 1, 2, 8), (1, 1, 4, 4), (1, 2, 2, 4), (2, 2, 2, 2)]:
        for perm in itertools.permutations(size):
            np.testing.assert_allclose(x.reshape(*perm).val, x.val.reshape(perm))
            def f(x0):
                dim, bs = x0.val.shape
                return x0.reshape(*perm, bs).reshape(dim, bs)

def test_convolve():
    np.random.seed(0)
    images = Tensor(np.ones((1, 1, 3, 3)))
    kernel = Tensor(np.ones((2, 2, 1, 1)))
    np.testing.assert_allclose(
            images.convolve(kernel).val[0,:,:,0],
            np.array([[4, 4], [4, 4]]))
    kernel2 = Tensor(np.ones((2, 2, 1, 2)))
    np.testing.assert_allclose(
            images.convolve(kernel2).val[0,:,:,:],
            np.ones((2,2,2))*4)

    images = Tensor(np.arange(9).reshape(1, 1, 3, 3))
    kernel = Tensor(np.arange(4).reshape(2, 2, 1, 1))
    np.testing.assert_allclose(
            images.convolve(kernel).val[0,:,:,0],
            np.array([[19, 25], [37, 43]]))

    x0 = Tensor(np.arange(9).reshape(9, 1))
    jacobian_test(x0, lambda x: x.T.reshape(-1, 1, 3, 3).convolve(kernel).reshape(-1, 4).T)
    kernel2 = Tensor(np.arange(8).reshape(2, 2, 1, 2))
    jacobian_test(x0, lambda x: x.T.reshape(-1, 1, 3, 3).convolve(kernel2).reshape(-1, 8).T)

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

def numerical_jacobian(x0, f, eps=1e-4):
    d0, o = x0.val.shape
    assert o == 1
    y0 = f(x0 - Tensor(np.eye(d0) * eps))
    y1 = f(x0 + Tensor(np.eye(d0) * eps))
    return (y1 - y0).val / (2 * eps)

def jacobian(x0, f):
    d0, o = x0.val.shape
    d1, _o = f(x0).val.shape
    assert o == _o == 1, "Can't compute Jacobians of batched input"
    x1 = x0.broadcast_to((d0,d1))
    y1 = f(x1)
    d1, _ = y1.val.shape
    y1.backward(v = np.eye(d1))
    return x1.grad.T

def jacobian_test(x0, f, eps=1e-4):
    J0 = numerical_jacobian(x0, f, eps)
    J1 = jacobian(x0, f)
    if not np.allclose(J0, J1):
        print(f'{x0.val=}')
        print(f'{x1.val=}')
        print(f'{y0.val=}')
        print(f'{y1.val=}')
    np.testing.assert_allclose(J0, J1)

def test_softmax():
    np.random.seed(0)
    #x = np.random.randn(4,1)
    x = np.arange(4).reshape(4,1)
    x0 = Tensor(x)
    x1 = Tensor(x)
    y0 = x0.softmax(axis=0)
    y1 = x1.softmax2(axis=0)
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
    x0 = Tensor(np.arange(1, 5, dtype=float).reshape(4, 1))
    jacobian_test(x0, lambda x: x.relu())
    jacobian_test(x0, lambda x: x.abs())
    jacobian_test(x0, lambda x: x.pow(2))
    jacobian_test(x0, lambda x: x.pow(-1))
    jacobian_test(x0, lambda x: x.softmax(axis=0))
    jacobian_test(x0, lambda x: x.softmax2(axis=0))
    x1 = Tensor(np.abs(np.random.randn(4,1)))
    jacobian_test(x1, lambda x: x.log())
    #jacobian_test(x1, lambda x: x.reshape(1,log())

def test_rectangular_jacobians():
    x0 = Tensor(np.arange(4).reshape(4, 1))
    # From 4 down to 2
    jacobian_test(x0, lambda x: x.reshape(2, 2, -1).sum(axis=0).reshape(2, -1))
    # From 4 up to 16
    jacobian_test(x0, lambda x: (x.reshape(4, 1, -1) + x.reshape(1, 4, -1)).reshape(16, -1))

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
    x = np.random.randn(b, d0)
    x0 = Tensor(x)
    x1 = Tensor(x)
    labels = np.random.randint(d1, size=(b,))
    c0 = x0.pow(2).softmax(axis=1).label_cross_entropy(labels)
    c1 = x1.pow(2).label_cross_entropy_on_logits(labels)
    np.testing.assert_allclose(c0.val, c1.val)
    c0.backward()
    c1.backward()
    np.testing.assert_allclose(x0.grad, x1.grad)

class Linear:
    def __init__(self, d1, d2):
        self.M = Tensor(np.random.randn(d1, d2) / (d1 + d2)**.5)
        self.B = Tensor(np.random.randn(1, d2) / (1 + d2)**.5)

    def __call__(self, x):
        return x @ self.M + self.B

    def parameters(self):
        return [self.M, self.B]

class Convolution:
    def __init__(self, h, w, c1, c2):
        self.K = Tensor(np.random.randn(h, w, c1, c2) / (w * h * c1 + c2)**.5)
        self.B = Tensor(np.random.randn(1, c2, 1, 1) / (c2 + 1)**.5)

    def __call__(self, x):
        return x.convolve(self.K) + self.B

    def parameters(self):
        return [self.K, self.B]

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

class ReLU:
    def __call__(self, x): return x.relu()
    def parameters(self): return []

class Reshape:
    def __init__(self, *shape): self.shape = shape
    def __call__(self, x): return x.reshape(*self.shape)
    def parameters(self): return []

from sklearn import datasets, utils, preprocessing
from sklearn.model_selection import train_test_split
import time
def get_mnist(full=False):
    if not full:
        X, y = datasets.load_digits(return_X_y=True)
        size = 8
    else:
        print('Downloading mnist...')
        X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
        print('Done.')
        X, y = X.to_numpy(), y.astype(int).to_numpy()
        size = 28
    return X.reshape(-1, 1, size, size), y, size

def mnist():
    X, y, size = get_mnist(full=False)
    X = X.reshape(-1, 1, size, size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    batch_size = 32
    def batches(X, y):
        X, y = utils.shuffle(X, y)
        for i in range(0, len(X), batch_size):
            yield Tensor(X[i:i+batch_size]), y[i:i+batch_size]

    ch = 8
    net = Sequential(
            Convolution(2, 2, 1, ch), ReLU(),
            Convolution(2, 2, ch, ch), ReLU(),
            Reshape(-1, (size-2)**2 * ch),
            Linear((size-2)**2*ch, 10),
        )

    def step(pb, epoch):
        total_loss, correct = 0, 0
        for X_batch, y_batch in tqdm.tqdm(batches(X_train, y_train), leave=False):
            #start = time.time()
            out = net(X_batch)
            #print('Forward:', time.time() - start)
            # Compute loss
            #loss = out.label_cross_entropy_on_logits(y_batch)
            loss = -(out / out.norm(axis=1)).select(y_batch).sum(axis=0)
            #loss = (-Tensor(2)*out.select(y_batch) + out.pow(2).sum(axis=1)).sum(axis=0)
            # Measure accuracy
            total_loss += loss.val
            #print(out.val.shape, out.val.arg y_batch)
            correct += (out.val.argmax(axis=1) == y_batch).sum()
            # Backprop
            #start = time.time()
            loss.backward()
            #print('Backwards:', time.time() - start)
            # Gradient descent
            #lr = 1e-2 if epoch < 100 else 1e-3
            lr = 1e-2
            for p in net.parameters():
                p.val -= lr * p.grad / batch_size
                p.grad = 0
        test_acc = 0
        for X_batch, y_batch in batches(X_test, y_test):
            test_acc += (net(X_batch).val.argmax(axis=1) == y_batch).sum()
        pb.set_description(
                f'Loss: {float(total_loss)/len(X_train):.3}, '
                f'Acc: {correct/len(X_train):.3}, '
                f'Test: {test_acc/len(X_test):.3}')

    with tqdm.tqdm(range(100)) as pb:
        for i in pb:
            step(pb, i)




if __name__ == '__main__':
    mnist()
