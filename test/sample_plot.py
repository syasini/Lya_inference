import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm 

gauss = norm(0,2)
x = np.linspace(-5,5,1000)
y = gauss.pdf(x)

plt.plot(x,y)
plt.show()


