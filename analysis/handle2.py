import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cumfreq
import numpy as np

numpy_vector = torch.nn.functional.silu(torch.norm(layer.weight, p=2, dim=1)).cpu().numpy()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].hist(numpy_vector, bins=30, density=True)
axs[0, 0].set_title('Histogram of Model Weights')

sns.kdeplot(numpy_vector, shade=True, ax=axs[0, 1])
axs[0, 1].set_title('Density Plot of Model Weights')

sns.boxplot(x=numpy_vector, ax=axs[1, 0])
axs[1, 0].set_title('Box Plot of Model Weights')

sns.violinplot(x=numpy_vector, ax=axs[1, 1])
axs[1, 1].set_title('Violin Plot of Model Weights')

plt.tight_layout()
plt.savefig("distribution2.png")

a = cumfreq(numpy_vector, numbins=100)
x = a.lowerlimit + np.linspace(0, a.binsize*a.cumcount.size, a.cumcount.size)
cumulative_values = a.cumcount / a.cumcount.max()

target = 0.45
target_index = np.abs(cumulative_values - target).argmin()
target_value = x[target_index]

plt.plot(x, cumulative_values)
plt.title('CDF of Model Weights')
plt.xlabel('Weight Value')
plt.ylabel('CDF')

plt.scatter([target_value], [cumulative_values[target_index]], color='red')  # 标记点
plt.axhline(y=cumulative_values[target_index], color='r', linestyle='--')  # 水平线
plt.axvline(x=target_value, color='r', linestyle='--')  # 垂直线
plt.text(target_value, cumulative_values[target_index], f'  {target_value:.2f}, {cumulative_values[target_index]:.2f}', color='red')

plt.grid(True)
plt.savefig("cdf2.png")

sum_below_target = np.sum(numpy_vector[numpy_vector < x[target_index]])
sum_above_target = np.sum(numpy_vector[numpy_vector >= x[target_index]])

sizes = [sum_below_target, sum_above_target]
labels = ['Sum Below 45% Index', 'Sum Above 45% Index']
colors = ['lightblue', 'lightgreen']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Comparison of Sum Below and Above 45% CDF Index')
plt.savefig("circle2.png")