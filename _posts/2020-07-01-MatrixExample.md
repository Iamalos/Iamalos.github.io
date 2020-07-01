# Matrix calculation in python



Let's consider two simple matrices A (3x3) and B (3x2). I am going to use torch two-dimensional `tensors` to represent matrices.

```python
A = tensor([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]])
B = tensor([[1.,1.], [2.,2.], [3.,3.]])
A.shape, B.shape
```




    (torch.Size([3, 3]), torch.Size([3, 2]))



We know from school calculus that to multiply two matrices together, they have to be compatible number of columns of matrix A has to be equal to the number of columns of matrix B

```python
assert(A.shape[1]==B.shape[0])
```

The results of A<sub>in</sub> x B<sub>nj</sub> is a matrix C <sub>ij</sub> and element C[i][j] is equal to the sum of dot product of row i of matrix A and column j of matrix B. [Vizualization of matrix multiply](http://matrixmultiplication.xyz/)

We can easliy implement this element-wise approach in code

#### Elementwise matrix multiplication

```python
def matmul(a: Tensor, b: Tensor):
    '''
    Multiply two matrices element-wise
    '''
    # check that matrices are compatible
    assert (a.shape[1] == b.shape[0])
    # get number of rows and columns of each matrix
    ar, ac = a.shape
    br, bc = b.shape
    
    # prepare the resulting matrix
    c = torch.zeros((ar,bc))
    
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i,j] += a[i][k] * b[k][j]
    return c
```

```python
matmul(A, B)
```




    tensor([[14., 14.],
            [32., 32.],
            [50., 50.]])



Let's check that our function works correctly. For this we will test against built-in matrix multiplication operator - `@`

```python
test_near(matmul(A,B), (A@B))
```

Great! We now have a way to multiply two matrices together. But have efficient is our approach? We have three loops, which would imply complexity $ O(n^3) $, which is pretty bad.

```python
A_large = torch.randn((5,700))
B_large = torch.randn((700,10))
```

```python
%time m1 = matmul(A_large, B_large)
```

    CPU times: user 1.24 s, sys: 3.33 ms, total: 1.25 s
    Wall time: 1.25 s


It took roughly 830 ms (results may vary depending on random seed) to multiply two matrices which seems rather quick, but in real life we often have to work with much larger matrices. For example, on a dataset with 50k rows, our function will take __more than a day__. We have to make it much faster.

#### Improving our function

An easy fix would be to speed up the outer loop. Insted of looping through each elements and then summing them (+= operator) we can do it in one go:

```python
(A[0,:] * B[:,0]).sum(0)
```




    tensor(14.)



```python
def matmul(a: Tensor, b: Tensor):
    '''
    Multiply two matrices element-wise
    '''
    # check that matrcices are compatible
    assert (a.shape[1] == b.shape[0])
    # get number of rows and columns of each matrix
    ar, ac = a.shape
    br, bc = b.shape
    
    # prepare the resulting matrix
    c = torch.zeros((ar,bc))
    
    for i in range(ar):
        for j in range(bc):
            c[i,j] = (a[i,:] * b[:,j]).sum(0)
    return c
```

```python
matmul(A, B)
```




    tensor([[14., 14.],
            [32., 32.],
            [50., 50.]])



```python
%time m2 = matmul(A_large, B_large)
```

    CPU times: user 4.22 ms, sys: 9 µs, total: 4.23 ms
    Wall time: 3.16 ms


Well, that is much better - actually more than __300__ times faster

### Broadcasting

To improve our function even more we have to talk about __broadcasting__. 

Beoadcasting describes how arrays with different shapes are treated during arithmetic operations. For example:

#### Broadcasting with a scalar

```python
A
```




    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])



```python
A > 0
```




    tensor([[True, True, True],
            [True, True, True],
            [True, True, True]])



```python
A + 2
```




    tensor([[ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]])



#### Broadcasting vector to a matrix

```python
v = tensor([1.,2.,1.]);v
```




    tensor([1., 2., 1.])



```python
# vector is broadcasted and added to each row
A + v
```




    tensor([[ 2.,  4.,  4.],
            [ 5.,  7.,  7.],
            [ 8., 10., 10.]])



To see what is going under the hood, when we try to add vector to a matrix, we can _expand_ the vector to the shape of a matrix

```python
v_exp = v.expand_as(A); v_exp
```




    tensor([[1., 2., 1.],
            [1., 2., 1.],
            [1., 2., 1.]])



So basically v is _copied_ three times. That's convenient, but it sounds really inefficient if we have to store additional copies in memory. Luckily, this is not the case and we can check how v is actually stored in memory by calling function `storage`:

```python
v_exp.storage()
```




     1.0
     2.0
     1.0
    [torch.FloatStorage of size 3]



Okay, but how does it actually know, that it has to _copy_ the row three times?

It knows because it has a stride of __(0,1)__ - it moves zero rows down and 1 column right.

```python
v_exp.stride()
```




    (0, 1)



What if we want to add vector to each column and not row of a matrix? We can add an additional dimension to the vector in a required place:

```python
v.shape, v.unsqueeze(0).shape, v.unsqueeze(1).shape
```




    (torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))



```python
v.unsqueeze(1)
```




    tensor([[1.],
            [2.],
            [1.]])



```python
A + v.unsqueeze(1)
```




    tensor([[ 2.,  3.,  4.],
            [ 6.,  7.,  8.],
            [ 8.,  9., 10.]])



We can also create new dimension by _indexing_ into it

```python
v.shape, v[None,:].shape, v[:,None].shape
```




    (torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))



#### Matmul with broadcasting

We get rid of the outer loopt by using broadcasting. We grab the ith row of the A matrix, convert it to column vector with `unsqueeze`, multiply by B and sum results by rows (meaning for each column)

```python
def matmul(a: Tensor, b: Tensor):
    '''
    Multiply two matrices element-wise
    '''
    # check that matrcices are compatible
    assert (a.shape[1] == b.shape[0])
    # get number of rows and columns of each matrix
    ar, ac = a.shape
    br, bc = b.shape
    
    # prepare the resulting matrix
    c = torch.zeros((ar, bc))
    
    for i in range(ar):
        c[i,:] = (a[i,:].unsqueeze(-1) * b).sum(0)
    return c
```

```python
%timeit -n 10 _=matmul(A_large, B_large)
```

    486 µs ± 101 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

