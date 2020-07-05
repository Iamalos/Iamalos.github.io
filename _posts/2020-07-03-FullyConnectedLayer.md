# Title



### Foward and backward passes

```python
# export
from exp.nb_MatrixExample import *

def get_data():
    """
    Grab MNIST data using path
    """
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        (x_train, y_train),(x_valid, y_valid), _ = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train, x_valid,y_valid))

def normalize(x,m,s): return (x-m) / s
```

Let's grab for this example [MNIST](http://yann.lecun.com/exdb/mnist/) data. MNIST data is a popular dataset for machine learning and deep learning and conists of handwritted digits from 0 to 9. By modern standards the dataset is considered to be trivial with many algorithms reaching >99% test accuracy.

 MNIST has a training set of 60,000 examples, and a test set of 10,000 examples.
 
 There are many different ways to grab MNIST dataset. We leverage FastAI `datasets`

```python
x_train, y_train, x_valid, y_valid = get_data()
```

```python
x_train.shape, x_valid.shape
```




    (torch.Size([50000, 784]), torch.Size([10000, 784]))



Our training set has 50 000 observations, each observation has 784 data points. This represents flattened 28 x 28 pixels of the resulting image. We can convert 784 vector back to a matrix form and plot it for vizualization

```python
mpl.rcParams['image.cmap'] = 'gray'

plt.imshow(x_train[0].view(28,28));
```


![png](2020-07-03-FullyConnectedLayer_files/output_6_0.png)


### Preparing the data

As a usual first step we normalize our data - subtract the mean and divide by the standard deviation of the __train dataset__. it is important to use the same mean and standard deviation while normalizing training and validation / test data

```python
train_mean, train_std = x_train.mean(), x_train.std()
train_mean, train_std
```




    (tensor(0.1304), tensor(0.3073))



```python
# normalizing the data
x_train = normalize(x_train, train_mean, train_std)
x_valid = normalize(x_valid,train_mean, train_std)
```

```python
# export
def test_near_zero(a, tol=1e-3): assert a.abs() < tol, f"Near zero: {a}"
```

```python
test_near_zero(x_train.mean())
test_near_zero(1 - x_train.std())
```

```python
n,m = x_train.shape
c = y_train.max() + 1
n,m,c
```




    (50000, 784, tensor(10))



### Starting version

Let's define the number of hidden layers

```python
nh = 50
```

Initialization of weights is __crucial__ to the training of the neural net. Using poor initialization could lead to very slow convergence of the loss function or even exploding. If weights are below 1, they will get smaller and smaller until they almost vanish and does not contribute to network learning.

We can look at a simplified example, inspired by [post](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/), to understand why weights are important. If our input for activation __z__ is near zero, sigmoid activation function degenerates to a linear one, which means it does not bring any new information. On the other hand, if **z** becomes too big or too small, activation function is flat at these areas and its gradient is approaching zero. We want the variance and bias for each layer to remain stable.

We  often don't worry about this as deep learning frameworks implement the necessary weights initialization under the hood, but it is still important to understand why it matters this much. I discuss weight initialization in greater details in separate posts.

The goal is for each layer to have mean of 0.0 and variance  of 1.0 for every layer in order to avoid gradient vanishing or exploding. This issues was tackled by Xavier in his paper [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html).

Let's briefly discuss the reasoning behind his approach. <br>
Consider a fully-connected linear layer: $ y = x * w + b$ which is $ y = x_{1}*w_{1} + x_{2}*w_{2} + ... + x_{N}*w_{N} + b $. Our goal is to make the variance of $y$ equal to $1$. Assuming independence between $x$ and $y$ we have the following formula: 

{% raw %}
$$Var(x*y) = Var(x)*Var(y) + (Var(x)*E(y))^2 + (Var(y)*E(x))^2$$
{% endraw %}

In our case $w_{i}$ was drawn from normal distrubution with zero mean and $x$ were normalized , thus for each i-th term we have:

{% raw %}
$$ Var(x_{i}*w_{i}) = Var(x_{i}) * Var(w_{i}) + (Var(x_{i})*0)^2 + (1 * 0) ^2  = Var(x_{i}) * Var(w_{i})) $$
{% endraw %}

We have N identically distributed elements, thus:
{% raw %}
$$ Var(y) = \sum_{i=1}^{N} Var(x_{i}) * Var(w_{i}) = N * Var(x_{i}) * Var(w_{i}))  $$
{% endraw %}

We want our input to the activation function ($y$) to have the same variance as the previous layer ($x$) in order to have stability in the system. This implies, that 

$$ N * Var(w_{i}) = 1 $$ 
$$or$$ 
{% raw %}
$$ Var(w_{i}) = 1 / N $$
{% endraw %}

This is what is called _Xavier_ initialization. Let's proceed with it below.

```python
# Xavier initialization
w1 = torch.randn(m, nh) / math.sqrt(m)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1) / math.sqrt(nh)
b2 = torch.zeros(1)
```

```python
# check that mean and variance are equal to 0 and 1
test_near_zero(w1.mean())
test_near_zero(w1.std() - 1/math.sqrt(m))
```

Let's define a simple linear layer:

```python
def lin(x,w,b): return x @ w + b
```

```python
t = lin(x_train, w1, b1)
```

```python
t.mean(), t.std()
```




    (tensor(0.0833), tensor(1.0343))



As we see, after applying linear layer, we succeeded in our goal - mean and variance of the input to activation function is 0 and 1 (almost). But this is not the end of the story - we have to actuall pass the results through the activation function. And the results itself will be the input to the next layer.

```python
def relu(x): return x.clamp_min(0)
```

```python
t = relu(lin(x_train, w1, b1))
t.mean(), t.std()
```




    (tensor(0.4491), tensor(0.6569))



After applying relu our mean and variance are distorted - relu clamps all negative values to zero, thus reducting variance by half and shifting mean by 0.5. We have to take into account when initializing the weigths. This is addressed by using _kaiming initialization_.

```python
# kaiming init for relu
w1 = torch.randn(m, nh) * math.sqrt(2/m)
```

```python
t = relu(lin(x_train, w1, b1))
t.mean(), t.std()
```




    (tensor(0.5875), tensor(0.8513))



This is still not perfect, but is much better than the inital approach. We could also subtract 0.5 to shift mean back to 0 (relu clamped the negative values and thus distorted the mean upwards). This initialization is implemented for us in PyTorch:

```python
#export
from torch.nn import init
```

```python
w1 = torch.zeros(m, nh)
init.kaiming_normal_(w1, mode='fan_out')
t = relu(lin(x_train,w1,b1))
```

```python
t.mean(), t.std()
```




    (tensor(0.5106), tensor(0.8187))



Unsirprisingly we get basically the same results. Note additional parameter - `fan_out`.This parameter specified either to preserve the magnitude of variance of the weights either in the forward pass (`fan_in`) or in the backwards pass (`fan_out`)

```python
# as discussed above, we can subtract 0.5 from our relu to get mean closer to zero
def relu(x): return x.clamp_min(0) - 0.5
```

Our simple forward pass can be specified as follows:

```python
def model(xb):
    l1 = lin(x_train, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3
```

```python
%timeit -n 10 model(x_train)
```

    24.2 ms ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


### Loss function: MSE

To keep things simple, we will consider MSE loss function, although in our case it does not make any practical sense

```python
#export
def mse(output, targ): return (output.squeeze(-1)-targ).pow(2).mean()
```

```python
preds=model(x_train)
mse(preds, y_train)
```




    tensor(26.2428)



### Gradients and backward pass

Now we approach the backward pass and need to calculate the gradients of each pass. Let's start from MSE

```python
def mse_grad(inp, targ):
    """
    we accumulate gradients during the backward pass
    First, calculate the gradient of loss with respect to its input, which is the output of the previous layer
    """
    inp.g = 2. * (inp.squeeze(-1) - targ).unsqueeze(-1) / inp.shape[0]
```

```python
def relu_grad(inp, out):
    # gradient of relu with respect to its input layer
    inp.g = (inp.float() > 0) * out.g
```

```python
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    b.g = out.g.sum(0)
```

```python
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # not necessary to compute loss
    loss = mse(out, targ)
    
    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```

```python
forward_and_backward(x_train, y_train)
```

```python
# save our gradients
w1g = w1.g.clone()
b1g = b1.g.clone()
w2g = w2.g.clone()
b2g = b2.g.clone()
inp = x_train.g.clone()
```

```python
w1g
```




    tensor([[-0.1282, -0.0313, -0.2001,  ..., -0.0689,  0.3500, -0.1771],
            [-0.1282, -0.0313, -0.2001,  ..., -0.0689,  0.3500, -0.1771],
            [-0.1282, -0.0313, -0.2001,  ..., -0.0689,  0.3500, -0.1771],
            ...,
            [-0.1282, -0.0313, -0.2001,  ..., -0.0689,  0.3500, -0.1771],
            [-0.1282, -0.0313, -0.2001,  ..., -0.0689,  0.3500, -0.1771],
            [-0.1282, -0.0313, -0.2001,  ..., -0.0689,  0.3500, -0.1771]])



```python
# we can check our results against pytorch
w12 = w1.clone().requires_grad_(True)
b12 = b1.clone().requires_grad_(True)
w22 = w2.clone().requires_grad_(True)
b22 = b2.clone().requires_grad_(True)
xt2 = x_train.clone().requires_grad_(True)
```

As pytorch calculates the backwards pass for us, we need only a forward pass

```python
def forward(inp, targ):
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    l3 = l2 @ w22 + b22
    return mse(l3, targ)
```

```python
loss = forward(xt2, y_train)
```

```python
loss.backward()
```

```python
test_near(w22.grad, w2g)
test_near(b22.grad, b2g)
test_near(w12.grad, w1g)
test_near(b12.grad, b1g)
test_near(xt2.grad, inp)
```

So we have just checked that our gradients are correct, but the code itself is very clunky. Let's refactor it by using classes.

## Refactor model

### Layers as classes

```python
class Relu():
    # __call__ allows us to call Relu directly as a function
    def __call__(self,inp):
        self.inp = inp
        self.out = self.inp.clamp_min(0.) - 0.5
        return self.out
    
    def backward(self):
        
```

```python
!nbdev_nb2md 2020-07-03-FullyConnectedLayer.ipynb
```
