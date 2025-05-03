import matplotlib.pyplot as plt
import numpy as np

# Generate random data for the message matrix
rows, cols = 10, 10  # 10 rows represent time steps, 100 columns represent bit indices
message = np.random.randn(rows, cols)  # Random data with normal distribution

# Create a heatmap for 'message'
plt.figure(figsize=(12, 12))
plt.imshow(message, cmap='coolwarm', aspect='auto')  # 'coolwarm' colormap for better visualization
plt.colorbar(label="Message Value")  # Add a color bar with a label
plt.title("Heatmap of Message")  # Change the title to reflect its role in the diagram
plt.xlabel("Bit Index")  # X-axis label
plt.ylabel("Time Step")  # Y-axis label

# Save the figure for use in the main diagram
plt.savefig("message_heatmap.png", dpi=300, bbox_inches='tight')  # Save with high resolution
plt.show()
