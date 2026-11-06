
import numpy as np

# Rainfall measurements in mm
rainfall = np.array([10, 25, 40, 60, 80])

print("Rainfall Array:", rainfall)
print("Average Rainfall:", np.mean(rainfall))
print("Maximum Rainfall:", np.max(rainfall))
print("Minimum Rainfall:", np.min(rainfall))
print("Array Shape:", rainfall.shape)

# Reshaping array
new_array = rainfall.reshape(5, 1)

print("\nReshaped Array:")
print(new_array)
