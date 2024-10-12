import matplotlib.pyplot as plt
import numpy as np

# Data
milvus_recall = [0.52443, 0.67, 0.779, 0.8423, 0.86, 0.8769, 0.87857, 0.8786]
milvus_qps = [44907, 43955, 32835, 21410.9, 14204.46, 7851.54, 4429.41, 2318.96]

faiss_recall = [0.367, 0.521, 0.667, 0.777, 0.84, 0.869, 0.87739, 0.87884]
faiss_qps = [304564.06, 154143.98, 85984.97, 52263.84, 30097.15, 15737.11, 8977.95, 4504.08]

# Plot
plt.figure(figsize=(10, 6))

# Plot Milvus
plt.plot(milvus_recall, milvus_qps, label='Milvus', marker='o', linestyle='-', color='b')

# Plot Faiss
plt.plot(faiss_recall, faiss_qps, label='Faiss', marker='x', linestyle='--', color='r')

# Log scale for Y axis
plt.yscale('log')

# Titles and labels
plt.title('Milvus vs. Faiss on IVF_PQ (SIFT1M)')
plt.xlabel('Recall')
plt.ylabel('QPS (Log scale)')

# Add grid
plt.grid(True, which="both", ls="--")

# Add legend
plt.legend()

# Show plot
plt.savefig("1")
