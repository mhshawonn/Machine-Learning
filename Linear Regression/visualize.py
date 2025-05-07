import matplotlib.pyplot as plt
import numpy as np
import os
from data_utils import load_data

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

housing = load_data()
housing.plot(kind="scatter", x='longitude', y='latitude', grid=True,alpha=0.2)
plt.show()