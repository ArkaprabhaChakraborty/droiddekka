import numpy as np
from droiddekka.simplekalman.kalmanfilter import simplekalmanfilter as skf

test = skf(4,2)
print(test)
test.state_process_setter(np.array([[3],[3],[1.5],[1.4]]),0.5)
test.predict()
print(test)
test.update(np.array([[3],[4]]))
print(test)   