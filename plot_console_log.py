import re
import matplotlib.pyplot as plt

log_file_path = 'gpt2_500m_log/500m_output.log'

# Initialize lists to store the extracted data
steps = []
losses = []
lrs = []
norms = []

# Define the regex pattern to match the desired lines
pattern = re.compile(r"^step\s+(\d+)\s+\|\s+loss:\s+([\d.]+)\s+\|\s+lr\s+([\de.-]+)\s+\|\s+norm:\s+([\d.]+)")

# Read the log file and extract the data
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.match(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            lrs.append(float(match.group(3)))
            norms.append(float(match.group(4)))

# Plot the data
plt.figure(figsize=(15, 5))

# Plot loss
plt.subplot(1, 3, 1)
plt.plot(steps, losses, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Steps')
plt.legend()

# Plot learning rate
plt.subplot(1, 3, 2)
plt.plot(steps, lrs, label='Learning Rate', color='orange')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Steps')
plt.legend()

# Plot norm
plt.subplot(1, 3, 3)
plt.plot(steps, norms, label='Norm', color='green')
plt.xlabel('Step')
plt.ylabel('Norm')
plt.title('Norm over Steps')
plt.legend()

plt.tight_layout()
plt.show()
