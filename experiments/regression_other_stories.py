import bambi as bmb 
import pandas as pd 
import numpy as np 

x = np.arange(1, 21)
y = 0.2 + 0.3*x + 0.5*np.random.normal(size=20)
fake = pd.DataFrame({"x": x, "y": y})

model = bmb.Model("y ~ x", fake) # ≈ stan_glm(y ~ x, data=fake)
# this holds the posterior draws
idata = model.fit(draws=1000)
