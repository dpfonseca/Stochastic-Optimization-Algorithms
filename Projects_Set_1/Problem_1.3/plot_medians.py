import matplotlib.pyplot as plt

# The following values were obtained running run_batch.py
mu_values = [0,0.005, 0.01, 0.02, 0.05, 0.1]
median_g = [0.9902, 0.9977, 0.9993, 1.0000, 1.0000, 1.0000]


plt.figure(figsize=(3,2), dpi = 300)

plt.plot(mu_values, median_g, 'bo-', linewidth=1, markersize=3, label = r'$x_1^*$')
plt.grid()
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(r'$p_{mut}$', fontsize = 8)
plt.ylabel('Median fitness', fontsize = 8)
plt.tight_layout()
plt.show()