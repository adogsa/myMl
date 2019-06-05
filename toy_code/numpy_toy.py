import numpy as np
x = np.array([1,2])
x.shape
# (2,)


y = np.expand_dims(x, axis=0)
# >>> y
# array([[1, 2]])
# >>> y.shape
# (1, 2)

y = np.expand_dims(x, axis=1)  # Equivalent to x[:,np.newaxis]
# >>> y
# array([[1],
#       [2]])
# >>> y.shape
# (2, 1)