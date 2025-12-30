import numpy as np

a  = np.random.randint(0,100 , size=(3,3))
b = np.random.randint(0,100 , size=(3,3))
print(a)
print(b)
print("ellement wise" + str(a*b))
print("matrix multiplication" + str(np.dot(a,b)))