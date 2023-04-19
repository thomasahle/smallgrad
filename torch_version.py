import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import datasets, utils, preprocessing
from sklearn.model_selection import train_test_split
import tqdm, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--full", action="store_true")
parser.add_argument("-l", "--loss", type=str)
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-2)
args = parser.parse_args()

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

X, y, size = get_mnist(full=args.full)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

batch_size = 128
def batches(X, y):
    X, y = utils.shuffle(X, y)
    for i in range(0, len(X), batch_size):
        yield (torch.tensor(X[i:i+batch_size], dtype=torch.float),
               torch.tensor(y[i:i+batch_size]))

ch = 64
net = nn.Sequential(
    nn.Conv2d(1, ch, 2),
    nn.ReLU(),
    nn.Conv2d(ch, ch, 2),
    nn.ReLU(),
    nn.Flatten(1),
    nn.Linear((size-2)**2*ch, 10),
)

def step(pb, epoch):
    total_loss, correct = 0, 0
    for X_batch, y_batch in tqdm.tqdm(batches(X_train, y_train),
                                      leave=False, total=X.shape[0]//batch_size):
        out = net(X_batch)
        if args.loss == 'ce':
            loss = F.cross_entropy(out, y_batch)
        elif args.loss == 'spherical':
            loss = -(out / out.norm(dim=1, keepdim=True))[range(X_batch.shape[0]), y_batch].sum(dim=0)
        elif args.loss == 'brier':
            loss = (-2*out[range(X_batch.shape[0]), y_batch] + (out**2).sum(dim=1)).sum(dim=0)

        with torch.no_grad():
            # Measure accuracy
            total_loss += loss
            correct += (out.argmax(dim=1) == y_batch).sum()
            # Backprop
            #loss.zero_grad()
            loss.backward()
            # Gradient descent
            lr = args.learning_rate
            for p in net.parameters():
                p -= lr * p.grad / batch_size
                p.grad[:] = 0
    test_acc = 0
    for X_batch, y_batch in batches(X_test, y_test):
        test_acc += (net(X_batch).argmax(dim=1) == y_batch).sum()
    pb.set_description(
            f'Loss: {float(total_loss)/len(X_train):.3}, '
            f'Acc: {correct/len(X_train):.3}, '
            f'Test: {test_acc/len(X_test):.3}')

with tqdm.tqdm(range(1000)) as pb:
    for i in pb:
        step(pb, i)

