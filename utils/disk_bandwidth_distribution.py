import re
import matplotlib.pyplot as plt
import seaborn as sns

# Read the file.
filename = 'bandwidth.txt'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse the data.
bandwidth_dict = {}
pattern = re.compile(r'([\w\s]+)bandwidth:\s*([\d.]+)GB/s')

for line in lines:
    m = pattern.search(line)
    if m:
        bw_type = m.group(1).strip()  # For example: 'CPU to Disk'
        bw_value = float(m.group(2))  # Bandwidth value.
        bandwidth_dict.setdefault(bw_type, []).append(bw_value)

# Plot the PDF (probability density function).
plt.figure(figsize=(7, 5))
for i, (bw_type, values) in enumerate(bandwidth_dict.items()):
    if len(values) > 1:
        sns.kdeplot(values, fill=True, label=bw_type)
plt.legend()
plt.title('Bandwidth PDF for All Types')
plt.xlabel('Bandwidth (GB/s)')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.savefig("bandwidth_pdf.png", dpi=150)  # Save the image.
plt.close()

print("Bandwidth distribution PDF saved as bandwidth_pdf.png.")