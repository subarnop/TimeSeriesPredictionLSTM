import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('real-daily-wages-in-pounds-england.xlsx',dtype={'Wage':float})
#print(dataset.Wage)

plt.plot(dataset.Wage)
plt.savefig('img/data_plot.png')
