import numpy as np


beam_width = 5

units = 4

batch_size = 3

data = np.array([np.ones(units)*(i+1)*10 for i in range(batch_size)])

print(data)

b = np.array([data]*beam_width)

print(b)

print('====')
c = b.swapaxes(0,1)
print(c)
print('dd====')

d = np.reshape(c,((beam_width*batch_size),-1))
print(d)

e = np.reshape(d,(batch_size,beam_width,-1))
print('e---')
print(e)

f = np.concatenate([np.expand_dims(e,-2) for _ in range(beam_width)],axis=-2)
print('fff')
print(f.shape)

g = np.reshape(f,(batch_size,beam_width*beam_width,-1))

print(g.shape)

# f =np.zeros((3,3))

h = np.reshape(f,(batch_size*beam_width,beam_width,-1))

print(h.shape)

a = np.array([1,2,3,4,5])

b=[[0,2],[2,3],[1,5]]

print(a[b])






















# class A:

#     def __init__(self,a,b):

#         self._a = a 
#         self._b = [b] 

# beam_width = 4
# batch_size = 5
# emd_dims = 10
# inti_a = np.zeros((batch_size,emd_dims))
# b = np.zeros(10)
# hyps = [A(inti_a,b) for _ in range(beam_width)]

# print(len(hyps))

# bs = np.array([a._b for a in hyps]).swapaxes(0,1)

# ass = [a._a for a in hyps]

# print(bs.shape)

# print(len(ass),ass[0].shape)

# cells = [np.expand_dims(state, axis=0) for state in ass]

# new_c = np.concatenate(cells, axis=0)

# print(new_c.shape)
