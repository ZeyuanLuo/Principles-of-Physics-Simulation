import matplotlib.pyplot as plt
import numpy as np


failures = [
    (53, 3), (57, 1), (58, 1), (63, 1),
    (70, 1), (70, 1),  # 70度有两次故障记录
    (75, 2)
]


success_temps_counts = {
    66: 1, 67: 3, 68: 1, 69: 1, 70: 2, 72: 1, 73: 1,
    75: 1, 76: 2, 78: 1, 79: 1, 80: 1, 82: 1
}

plt.figure(figsize=(12, 7))

def plot_stacked_points(data, color, marker, label):
    point_counts = {}
    x_vals = []
    y_vals = []

    for x, y in data:
        count = point_counts.get((x, y), 0)

        offset = count * 0.15

        x_vals.append(x)
        y_vals.append(y + offset)

        point_counts[(x, y)] = count + 1

    plt.scatter(x_vals, y_vals, color=color, s=150, marker=marker,
                edgecolors='black', label=label, alpha=0.9)


success_data = []
for t, count in success_temps_counts.items():
    for _ in range(count):
        success_data.append((t, 0))

plot_stacked_points(success_data, 'green', 's', 'Success (0 failures)')
plot_stacked_points(failures, 'red', 'X', 'Failure (>0 faults)')

plt.axvline(x=40, color='blue', linestyle='--', linewidth=2, label='Today temperature (40°F)')
plt.text(38.5, 3, 'Forecast:\nExtremely high risk', color='blue', fontweight='bold', ha='right')

plt.axvline(x=65, color='orange', linestyle=':', linewidth=2, label='critical threshold (~65°F)')

plt.title('Carter Racing: Relationship between Temperature and Gasket Failure', fontsize=16)
plt.xlabel('Ambient temperature (°F)', fontsize=14)
plt.ylabel('Number of gasket fractures', fontsize=14)
plt.xticks(np.arange(35, 90, 5))
plt.yticks(range(0, 5))
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()