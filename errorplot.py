import numpy as np
import matplotlib.pyplot as plt
cycleerrors = np.load('weights/cycleerrors.npy')
cycleerrorsval = np.load('weights/cycleerrorsval.npy')

plt.plot(cycleerrors,'black')
plt.plot(cycleerrorsval,'r')
plt.show()
