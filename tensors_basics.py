import torch
from torch import nn

'''
tensor is 2-dimensional,in this case having 3 rows and 4 columns
type of object returned in torch.Tensor(alias for torch.FloatTensor)
1-D dimensional tensor->vector, 2-D->Matrix,more than that->just tensor
'''
x = torch.empty(3,4)
print(type(x))
print(x)


zeros = torch.zeros(2,3)  #tensor with all entries = 0
print(zeros)

ones = torch.ones(2,3)    #tensor with all entries = 1
print(ones)

random = torch.rand(2,3)  #tensor with random values
print(random)


a = torch.tensor((2,3,4,5,6,7)) #shape 6(more like 1,6 if we see it in context of matrix i guess)
print(a)
print(a.shape)

b = torch.tensor(([2,4,6],[3,6,9])) #shape 2,3
print(b)
print(b.shape)


## default dtype==float32
t1 = torch.ones((2,3), dtype=torch.int16) #tensor with all values=1 but type=int
print(t1)

t2 = torch.rand((2,3), dtype=torch.float64) #tensor with random values type=>float
print(t2)

t3 = t2.to(torch.int32) #converts it to 32
print(t3)


ones = torch.zeros(3,3) + 1  #basic arithmetic 
twos = torch.ones(2,2) * 2
print(ones)
print(twos)


'''
Broadcasting
Broadcasting is a way to perform an operation between tensors that have similarities in their shapes.
In the example below, the one-row, four-column tensor is multiplied by both rows of the two-row, four-column tensor.

This is an important operation in Deep Learning.
The common example is multiplying a tensor of learning weights by a batch of input tensors,
applying the operation to each instance in the batch separately, and returning a tensor of identical shape - 
just like our (2, 4) * (1, 4) example above returned a tensor of shape (2, 4).

The rules for broadcasting are:

    ->Each tensor must have at least one dimension - no empty tensors.

    ->Comparing the dimension sizes of the two tensors, going from last to first:

        =>Each dimension must be equal, or

        =>One of the dimensions must be of size 1, or

        =>The dimension does not exist in one of the tensors

'''
rand = torch.rand(2,4)
doubled = rand * (torch.ones(2,4))
print(doubled)


b1 = torch.ones(4, 3, 2)

b2 = b1 * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b2)

b3 = b1 * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(b3)

b4 = b1 * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(b4)



if torch.cuda.is_available(): 
    print('GPU')
else:
    print('CPU')


'''
PyTorch models generally expect batches of input.

For example, imagine having a model that works on
3 x 226 x 226 images - a 226-pixel square with 3 color channels.
When you load and transform it, you’ll get a tensor of shape (3, 226, 226).
Your model, though, is expecting input of shape (N, 3, 226, 226),
where N is the number of images in the batch. So to make a batch we use unsqueeze.
The unsqueeze() method adds a dimension of extent 1.
unsqueeze(0) adds it as a new zeroth dimension
'''
i1 = torch.rand(3, 226, 226)
i2 = i1.unsqueeze(0)

print(i1.shape)
print(i2.shape)



output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20) #used for cnn mostly going from conv layer to linear layer
print(input1d.shape)
