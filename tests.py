
def test_fast():
    x = Value(3)
    y = x * x # x^2
    z = y * y # x^4
    t = z * z # x^8
    backward(t)
    assert t.grad == 1
    assert z.grad == 2 * 3**4
    assert y.grad == 4 * 3**6
    assert x.grad == 8 * 3**7

def test_mul():
    x = Value(3)
    y = Value(4)
    z = mul(x, y)
    t = mul(z, z)
    backward(t)
    assert x.grad == 2 * x.val * y.val**2
    assert y.grad == 2 * y.val * x.val**2

def test_add():
    v0, v1, v2, v3 = [Value(i) for i in range(2, 6)]
    u = add(add(add(v0, v1), v2), v3)
    backward(u)
    assert v0.grad == v1.grad == v2.grad == v3.grad == 1

def dot(us, vs):
    return sum((u * v for u, v in zip(us, vs)), start=Value(0))

def test_dot():
    us = [Value(i) for i in range(10)]
    vs = [Value(i) for i in range(10)]
    z = dot(us, vs)
    assert z.val == sum(u.val * v.val for u, v in zip(us, vs))
    backward(z)
    for u, v in zip(us, vs):
        assert u.grad == v.val
        assert v.grad == u.val

def mv_dot(matrix, vec):
    return [dot(row, vec) for row in matrix]

def test_matmul():
    M = np.arange(9).reshape(3, 3)
    T = np.arange(3).reshape(3, 1)
    S = np.arange(3,6).reshape(1, 3)

    nM = Value(M)
    nT = Value(T)
    nS = Value(S)
    nZ = nS @ (nM @ nT)
    backward(nZ)

    pM = [[Value(i) for i in row] for row in M]
    pT = [Value(i) for i in T[:,0]]
    pS = [Value(i) for i in S[0]]
    pMT = mv_dot(pM, pT)
    pZ = dot(pS, pMT)
    backward(pZ)

    assert nM.grad.tolist() == [[v.grad for v in row] for row in pM]
    assert nT.grad.tolist() == [[v.grad] for v in pT]
    assert nS.grad.tolist() == [[v.grad for v in pS]]

def test_relu():
    x = Value(np.array([[3]]))
    y = Value(np.array([[-3]]))
    z = relu(x) + relu(y)
    backward(z)
    assert x.grad == 1
    assert y.grad == 0

