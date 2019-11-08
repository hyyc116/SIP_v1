import numpy as np

a = [[1,1],[2,2],[2,2]]
b = [[4,4],[3,3],[2,2]]
c = [2,2,2]
d = [3,3,2]


e = [a,b]

f = [c,d]

g = [e,f]

g = np.array(g)

print(g.shape)
print(g)
h = g.reshape(2,3,2)
print(h.shape)
print(h)
# print(np.mean(g,axis=0))

print([t for t in range(1,1)])

a = np.array([a,b])
print(a)
print(a.shape)

b = a[:,:,0]
print(b.shape)
print(b)

