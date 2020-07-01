# Matrix calculation in python



Let's consider two simple matrices A (3x3) and B (3x2). I am going to use torch two-dimensional `tensors` to represent matrices.

```python
A = tensor([[1,2,3], [4,5,6], [7,8,9]])
B = tensor([[1,1], [2,2], [3,3]])
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
test_near(matmul(A,B), (A@B).float())
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-89-33e115f8b40b> in <module>
    ----> 1 test_near(matmul(A,B), (A@B).float())
    

    <ipython-input-88-41a0fa6fd4ef> in matmul(a, b)
         15         for j in range(bc):
         16             for k in range(ac):
    ---> 17                 c[i,k] += a[i][j] * b[j][k]
         18     return c


    IndexError: index 2 is out of bounds for dimension 1 with size 2


Great! We now have a way to multiply two matrices together. But have efficient is our approach? We have three loops, which would imply complexity $ O(n^3) $, which is pretty bad.

```python
A_large = torch.randn((5,700))
B_large = torch.randn((700,10))
```

```python
%time m1 =matmul(A_large, B_large)
```

    CPU times: user 838 ms, sys: 0 ns, total: 838 ms
    Wall time: 835 ms


It took roughly 830 ms to multiply two matrices which seems rather quick, but in real life we often have to work with much larger matrices. For example, on a dataset with 50k rows, our function will take __more than a day__. We have to make it much faster.

#### Improving our function

An easy fix would be to speed up the outer loop. Insted of looping through each elements and then summing them (+= operator) we can do it in one go:

```python
(A[0,:] * B[:,0]).sum(0)
```




    tensor(14)



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

    CPU times: user 0 ns, sys: 3.51 ms, total: 3.51 ms
    Wall time: 2.72 ms


Well, that is much better - actually more than __300__ times faster

```python
835/2.72
```




    306.985294117647



#### Broadcasting
