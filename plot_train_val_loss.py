import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv(os.path.join('output', 'training_log.csv'))
train, = plt.plot(df.iloc[:,1], 'b-', label='Training Loss')
val, = plt.plot(df.iloc[:,2], 'r-', label='Validation Loss')
plt.legend(handles=[train, val])
plt.show()
