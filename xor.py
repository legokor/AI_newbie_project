import numpy as np

# The training data.
X = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])

# The labels for the training data.
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

w0=np.random.normal(0,1,(2,2))
w1=np.random.normal(0,1,(1,2))

b0=np.random.random((2,1))
b1=np.random.random((1,1))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    return z*(1-z)


def forward(x):
    z0=w0.dot(x.reshape(x.shape[0],1))+b0
    h=sigmoid(z0)
    z1=w1.dot(h)+b1
    return sigmoid(z1)

for j in range(7000):
    dw0=0
    dw1=0

    db0=0
    db1=0

    for i in range(4):        
        z0=w0.dot(X[i].reshape(X[i].shape[0],1))+b0
        h=sigmoid(z0)
        z1=w1.dot(h)+b1
        o=sigmoid(z1)
        
        dw1+=2*(o-y[i])* h.T
        db1+=2*(o-y[i])
        
        dw0+= (np.multiply((w1.T * 2 * (o-y[i])) , dsigmoid(h))).dot((X[i].reshape(X[i].shape[0],1)).T)
        db0+= (np.multiply((w1.T * 2 * (o-y[i])) , dsigmoid(h)))
    
    w0=w0-0.1*dw0/4
    w1=w1-0.1*dw1/4
    b0=b0-0.1*db0/4
    b1=b1-0.1*db1/4


print(forward(X[0]))
print(forward(X[1]))
print(forward(X[2]))
print(forward(X[3]))
