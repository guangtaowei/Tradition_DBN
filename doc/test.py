import numpy as np
from doc import deepnet

X = np.random.rand(30, 200)
dbn = deepnet.dbn(30, [100, 100])
dbn.fit(X)
dbn.sample(10)