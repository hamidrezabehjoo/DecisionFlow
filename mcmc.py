import matplotlib.pyplot as plt
# Data
col1 = [1000, 10000, 50000, 100000]
col2 = [1.7514e-01, 9.4443e-02, 3.6476e-02, 2.1900e-02]
col3 = [4.2872e-01, 1.3005e-01, 5.5077e-02, 3.4142e-02]
# IEEE-friendly plot settings
plt.figure(figsize=(3.5, 2.8))  # IEEE column width
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': True,
    'legend.loc': 'best',
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'lightgray',
    'lines.linewidth': 1.5
})
# Plot col1 vs col2
plt.plot(col1, col2, marker='o', color='blue', linewidth=1.5, label=r'$\Delta_1 \ \text{DF}$')
# Plot col1 vs col3
plt.plot(col1, col3, marker='s', color='red', linewidth=1.5, linestyle='-', label=r'$\Delta_2 \ \text{DF}$')

# Add horizontal lines
plt.axhline(y=0.12, color='green', linestyle='--', linewidth=1, label=r'$\Delta_1 \ \text{MH}$' )
plt.axhline(y=0.18, color='purple', linestyle='-.', linewidth=1, label=r'$\Delta_2 \ \text{MH}$')

# Setting the plot appearance
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Number of Samples', fontsize=9)
plt.ylabel(r'$\Delta$', fontsize=9)
plt.xscale('log')  # Use log scale for col1
plt.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor='lightgray')
# Tighten layout and adjust margins for IEEE format
plt.tight_layout(pad=0.5)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
# Save with high DPI for print quality
plt.savefig('col1_vs_col2_col3.png', dpi=600, bbox_inches='tight')
plt.savefig('col1_vs_col2_col3.pdf', bbox_inches='tight')
# Show the plot
plt.show()
