import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
output_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\final_report\Images"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ===================== DATA DEFINITIONS =====================

# MNIST Models (Sequential Baseline only; Parallel excluded)
# [Name, Latency(ms), Power(W), LUTs, FFs, BRAMs, DSPs, Accuracy(%), Cycles]
mnist_data = [
    ("Seq. Baseline", 0.832, 0.161, 14925, 31571, 9, 23, 98.35, 83198),
    ("TTQ+BN",        0.905, 0.153, 14588, 31523, 1, 51, 97.28, 90534),
    ("TTQ+Threshold", 0.887, 0.171, 23085, 31591, 1, 54, 97.25, 88706),
    ("TTQ+Hysteresis",0.815, 0.200, 33565, 32734, 1, 54, 96.96, 81505),
]

# CIFAR-10 Models
# [Name, Latency(ms), Power(W), LUTs, FFs, BRAMs, DSPs, Accuracy(%), Cycles]
cifar_data = [
    ("Full Baseline", 70.50, 0.182, 23241, 15632, 48,   88, 91.86, 2820156),
    ("TTQ+BN",        71.94, 0.154, 24689, 16437, 16,   36, 90.62, 2877770),
    ("TTQ+Threshold", 50.45, 0.162, 31828, 16769, 16,   36, 89.85, 2017961),
    ("TTQ+Hysteresis",36.88, 0.160, 35798, 16623, 17.5, 36, 88.34, 1475264),
]


# =========================================================================
# 1. MNIST Power vs Latency vs Utilization
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

names   = [d[0] for d in mnist_data]
lat     = [d[1] for d in mnist_data]
pwr     = [d[2] for d in mnist_data]
luts    = [d[3] for d in mnist_data]
acc     = [d[7] for d in mnist_data]
sizes   = [l / 80 for l in luts]

scatter = ax.scatter(lat, pwr, s=sizes, c=acc, cmap='viridis',
                     alpha=0.85, edgecolors='black', linewidths=1.5,
                     vmin=96.5, vmax=98.5)

for i, nm in enumerate(names):
    offset = (10, 8) if i != 3 else (10, -15)
    ax.annotate(nm, (lat[i], pwr[i]), xytext=offset,
                textcoords='offset points', fontsize=9, fontweight='bold')

ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Power Consumption (W)")
ax.set_title("MNIST: Power vs. Latency\n(Bubble size = Slice LUT count, Color = Accuracy)")
ax.set_xlim(0.75, 0.96)
ax.set_ylim(0.13, 0.22)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Accuracy (%)")
for lv in [15000, 25000, 35000]:
    ax.scatter([], [], s=lv/80, c='gray', alpha=0.5, edgecolors='black',
               label=f"{lv:,} LUTs")
ax.legend(title="Resource Footprint", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mnist_power_vs_latency.png"), dpi=200)
plt.close()
print("Generated mnist_power_vs_latency.png")


# =========================================================================
# 2. CIFAR-10 Power vs Latency vs Utilization
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

names   = [d[0] for d in cifar_data]
lat     = [d[1] for d in cifar_data]
pwr     = [d[2] for d in cifar_data]
luts    = [d[3] for d in cifar_data]
acc     = [d[7] for d in cifar_data]
sizes   = [l / 80 for l in luts]

scatter = ax.scatter(lat, pwr, s=sizes, c=acc, cmap='plasma',
                     alpha=0.85, edgecolors='black', linewidths=1.5,
                     vmin=87.5, vmax=91.0)

for i, nm in enumerate(names):
    offset = (12, 5) if i != 1 else (12, -12)
    ax.annotate(nm, (lat[i], pwr[i]), xytext=offset,
                textcoords='offset points', fontsize=9, fontweight='bold')

ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Power Consumption (W)")
ax.set_title("CIFAR-10: Power vs. Latency\n(Bubble size = Slice LUT count, Color = Accuracy)")
ax.set_xlim(28, 82)
ax.set_ylim(0.13, 0.20)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Accuracy (%)")
for lv in [20000, 30000, 38000]:
    ax.scatter([], [], s=lv/80, c='gray', alpha=0.5, edgecolors='black',
               label=f"{lv:,} LUTs")
ax.legend(title="Resource Footprint", loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cifar10_power_vs_latency.png"), dpi=200)
plt.close()
print("Generated cifar10_power_vs_latency.png")


# =========================================================================
# 3. Activation Sparsity Progression (Layer-wise)
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5))

# MNIST: Post-ReLU zeros
layers_m = ["Pool1 Out", "Pool2 Out", "FC1 Out"]
sparsity_m = [23.9, 28.1, 44.2]

# CIFAR-10: Post-ReLU zeros (100 - active ratio)
layers_c = ["Pool1 Out\n(Conv2 In)", "Pool2 Out\n(Conv3 In)", "Conv3 Out\n(Conv4 In)", "Conv4 Out\n(GAP In)"]
sparsity_c = [33.9, 53.0, 73.7, 50.0]

ax.plot(np.arange(len(layers_m)), sparsity_m, marker='o', color='#1f77b4',
        linewidth=2.5, markersize=9, label="MNIST (4-Layer 2D CNN)", linestyle='-')
for i, v in enumerate(sparsity_m):
    ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

ax.plot(np.arange(len(layers_c)), sparsity_c, marker='s', color='#ff7f0e',
        linewidth=2.5, markersize=9, label="CIFAR-10 (4-Layer CNN)", linestyle='--')
for i, v in enumerate(sparsity_c):
    ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, -14),
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

xticks = ["Pool1 Out\n(Stage 1)", "Pool2 Out\n(Stage 2)", "Conv3/FC1 Out\n(Stage 3)", "Conv4/FC2 Out\n(Stage 4)"]
ax.set_xticks(np.arange(max(len(layers_m), len(layers_c))))
ax.set_xticklabels(xticks)
ax.set_xlabel("Layer Boundary")
ax.set_ylabel("Activation Sparsity (%)")
ax.set_title("Natural Activation Sparsity (Zeros after ReLU)\nProgression across Network Layers")
ax.set_ylim(15, 85)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "activation_vs_sparsity.png"), dpi=200)
plt.close()
print("Generated activation_vs_sparsity.png")


# =========================================================================
# 4. DAAP Threshold Sweep (MNIST)
# =========================================================================
fig, ax1 = plt.subplots(figsize=(8, 5))

tau_sweep   = np.linspace(0.0, 0.8, 9)
acc_drop_m  = [0.0, -0.02, -0.03, 0.05, 0.10, 0.15, 0.35, 0.70, 1.50]
mac_skip_m  = [0.0, 8.5, 18.2, 28.5, 37.2, 45.1, 52.8, 60.5, 68.2]

color1 = '#1f77b4'
ax1.set_xlabel(r'Base Threshold ($\tau_{base}$)', fontweight='bold')
ax1.set_ylabel('Accuracy Drop (%)', color=color1, fontweight='bold')
l1 = ax1.plot(tau_sweep, acc_drop_m, color=color1, marker='o', linewidth=2,
              label='Accuracy Drop (%)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-0.2, 2.0)

ax2 = ax1.twinx()
color2 = '#2ca02c'
ax2.set_ylabel('MAC Operations Skipped (%)', color=color2, fontweight='bold')
l2 = ax2.plot(tau_sweep, mac_skip_m, color=color2, marker='s', linewidth=2,
              linestyle='--', label='MAC Skipped (%)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-5, 80)

ax1.axvline(x=0.3, color='red', linestyle=':', linewidth=2,
            label=r'Optimal ($\tau_{base}=0.3$)')

lines = l1 + l2
labs  = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc='upper left')
plt.title("DAAP Threshold Sweep Trade-off\n(MNIST Activation Pruning Analysis)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "daap_threshold_sweep.png"), dpi=200)
plt.close()
print("Generated daap_threshold_sweep.png")


# =========================================================================
# 5. Resource Utilization Comparison (Grouped Bar Chart)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- MNIST ---
ax = axes[0]
models_m = ["Seq. Baseline", "TTQ+BN", "TTQ+Thresh", "TTQ+Hyst"]
lut_pct_m = [28.1, 27.4, 43.4, 63.1]
ff_pct_m  = [29.7, 29.6, 29.7, 30.8]
bram_pct_m= [6.4, 0.7, 0.7, 0.7]
dsp_pct_m = [10.5, 23.2, 24.6, 24.6]

x = np.arange(len(models_m))
w = 0.2
ax.bar(x - 1.5*w, lut_pct_m, w, label='LUT %', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.bar(x - 0.5*w, ff_pct_m,  w, label='FF %',  color='#4CAF50', edgecolor='black', linewidth=0.5)
ax.bar(x + 0.5*w, bram_pct_m,w, label='BRAM %',color='#FF9800', edgecolor='black', linewidth=0.5)
ax.bar(x + 1.5*w, dsp_pct_m, w, label='DSP %', color='#9C27B0', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models_m, fontsize=9)
ax.set_ylabel("Utilization (%)")
ax.set_title("MNIST: Resource Utilization")
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(0, 80)
ax.axhline(y=100, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

# --- CIFAR-10 ---
ax = axes[1]
models_c = ["Full Baseline", "TTQ+BN", "TTQ+Thresh", "TTQ+Hyst"]
lut_pct_c = [43.7, 46.4, 59.8, 67.3]
ff_pct_c  = [14.7, 15.5, 15.8, 15.6]
bram_pct_c= [34.3, 11.4, 11.4, 12.5]
dsp_pct_c = [40.0, 16.4, 16.4, 16.4]

ax.bar(x - 1.5*w, lut_pct_c, w, label='LUT %', color='#2196F3', edgecolor='black', linewidth=0.5)
ax.bar(x - 0.5*w, ff_pct_c,  w, label='FF %',  color='#4CAF50', edgecolor='black', linewidth=0.5)
ax.bar(x + 0.5*w, bram_pct_c,w, label='BRAM %',color='#FF9800', edgecolor='black', linewidth=0.5)
ax.bar(x + 1.5*w, dsp_pct_c, w, label='DSP %', color='#9C27B0', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models_c, fontsize=9)
ax.set_ylabel("Utilization (%)")
ax.set_title("CIFAR-10: Resource Utilization")
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(0, 80)
ax.axhline(y=100, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "resource_utilization.png"), dpi=200)
plt.close()
print("Generated resource_utilization.png")


# =========================================================================
# 6. Accuracy vs Latency Trade-off (Both Datasets)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# MNIST
ax = axes[0]
for d in mnist_data:
    ax.scatter(d[1], d[7], s=180, zorder=5, edgecolors='black', linewidths=1.2)
    ax.annotate(d[0], (d[1], d[7]), xytext=(8, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Software Accuracy (%)")
ax.set_title("MNIST: Accuracy vs. Latency")
ax.set_xlim(0.75, 0.96)
ax.set_ylim(96.5, 98.8)

# CIFAR-10
ax = axes[1]
for d in cifar_data:
    ax.scatter(d[1], d[7], s=180, zorder=5, edgecolors='black', linewidths=1.2)
    ax.annotate(d[0], (d[1], d[7]), xytext=(8, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Software Accuracy (%)")
ax.set_title("CIFAR-10: Accuracy vs. Latency")
ax.set_xlim(28, 82)
ax.set_ylim(87.5, 91.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_vs_latency.png"), dpi=200)
plt.close()
print("Generated accuracy_vs_latency.png")


# =========================================================================
# 7. CIFAR-10 Per-Layer Cycle Breakdown (Stacked Bar)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

models_cycle = ["TTQ+BN\n(Baseline)", "TTQ+BN\n+Threshold", "TTQ+BN\n+Hysteresis"]

# Per-layer cycles from comparison_analysis_report.md
conv1_c = [376834, 362130, 360594]
mask1_c = [0, 0, 36973]
conv2_c = [1257474, 903606, 575863]
mask2_c = [0, 0, 16683]
conv3_c = [604162, 382759, 256726]
mask3_c = [0, 0, 14177]
conv4_c = [604162, 345167, 194697]
gap_c   = [4226, 4227, 4227]
fc1_c   = [27906, 17605, 13070]
fc2_c   = [3006, 2463, 2254]

x = np.arange(len(models_cycle))
w = 0.5

colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
          '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']

bottoms = np.zeros(len(models_cycle))
layers_list = [conv1_c, mask1_c, conv2_c, mask2_c, conv3_c, mask3_c, conv4_c, gap_c, fc1_c, fc2_c]
layer_names = ["Conv1+Pool1", "Mask Gen 1", "Conv2+Pool2", "Mask Gen 2",
               "Conv3", "Mask Gen 3", "Conv4", "GAP", "FC1", "FC2"]

for i, (layer, name) in enumerate(zip(layers_list, layer_names)):
    layer_arr = np.array(layer, dtype=float)
    ax.bar(x, layer_arr / 1e6, w, bottom=bottoms / 1e6,
           label=name, color=colors[i], edgecolor='black', linewidth=0.3)
    bottoms += layer_arr

# Total labels on top
totals = [2877770, 2017961, 1475264]
for i, t in enumerate(totals):
    ax.text(i, t/1e6 + 0.03, f"{t/1e6:.2f}M", ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models_cycle, fontsize=10)
ax.set_ylabel("Inference Cycles (Millions)")
ax.set_title("CIFAR-10: Per-Layer Cycle Breakdown")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cifar10_cycle_breakdown.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Generated cifar10_cycle_breakdown.png")


# =========================================================================
# 8. Speedup Factor Comparison
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MNIST speedup relative to TTQ+BN baseline (90534 cycles)
ax = axes[0]
base_cycles_m = 90534
speedup_m = [base_cycles_m / d[8] for d in mnist_data]
model_nm  = [d[0] for d in mnist_data]
colors_m = ['#78909C', '#42A5F5', '#66BB6A', '#EF5350']
bars = ax.bar(model_nm, speedup_m, color=colors_m, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, speedup_m):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.2f}x", ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel("Speedup Factor (vs. TTQ+BN)")
ax.set_title("MNIST: Inference Speedup")
ax.set_ylim(0, 1.3)

# CIFAR-10 speedup relative to TTQ+BN baseline (2877770 cycles)
ax = axes[1]
base_cycles_c = 2877770
speedup_c = [base_cycles_c / d[8] for d in cifar_data]
model_nc  = [d[0] for d in cifar_data]
colors_c = ['#78909C', '#42A5F5', '#66BB6A', '#EF5350']
bars = ax.bar(model_nc, speedup_c, color=colors_c, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, speedup_c):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}x", ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel("Speedup Factor (vs. TTQ+BN)")
ax.set_title("CIFAR-10: Inference Speedup")
ax.set_ylim(0, 2.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "speedup_comparison.png"), dpi=200)
plt.close()
print("Generated speedup_comparison.png")


# =========================================================================
# 9. Weight Compression and BRAM Savings
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# BRAM usage
ax = axes[0]
models_bram = ["Seq. BL\n(MNIST)", "TTQ+BN\n(MNIST)", "Full BL\n(CIFAR)", "TTQ+BN\n(CIFAR)"]
bram_vals = [9, 1, 48, 16]
bram_colors = ['#ef5350', '#66bb6a', '#ef5350', '#66bb6a']
bars = ax.bar(models_bram, bram_vals, color=bram_colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, bram_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel("BRAM Tiles Used")
ax.set_title("BRAM Savings: Full-Precision vs TTQ")
ax.set_ylim(0, 55)

# DSP usage
ax = axes[1]
models_dsp = ["Seq. BL\n(MNIST)", "TTQ+BN\n(MNIST)", "Full BL\n(CIFAR)", "TTQ+BN\n(CIFAR)"]
dsp_vals = [23, 51, 88, 36]
dsp_colors = ['#42a5f5', '#ff9800', '#42a5f5', '#ff9800']
bars = ax.bar(models_dsp, dsp_vals, color=dsp_colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, dsp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel("DSP Slices Used")
ax.set_title("DSP Usage: Full-Precision vs TTQ")
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bram_dsp_comparison.png"), dpi=200)
plt.close()
print("Generated bram_dsp_comparison.png")


# =========================================================================
# 10. Energy per Inference Comparison
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Energy = Power * Latency
all_models = []
all_energy = []
all_colors = []

for d in mnist_data:
    all_models.append(f"M: {d[0]}")
    all_energy.append(d[2] * d[1])  # W * ms = mJ
    all_colors.append('#2196F3')

for d in cifar_data:
    all_models.append(f"C: {d[0]}")
    all_energy.append(d[2] * d[1])  # W * ms = mJ
    all_colors.append('#FF9800')

bars = ax.bar(range(len(all_models)), all_energy, color=all_colors,
              edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, all_energy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha='center', fontsize=8, fontweight='bold')

ax.set_xticks(range(len(all_models)))
ax.set_xticklabels(all_models, rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Energy per Inference (mJ)")
ax.set_title("Energy Efficiency: All Accelerator Designs\n(M = MNIST, C = CIFAR-10)")

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', edgecolor='black', label='MNIST'),
                   Patch(facecolor='#FF9800', edgecolor='black', label='CIFAR-10')]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "energy_per_inference.png"), dpi=200)
plt.close()
print("Generated energy_per_inference.png")


print("\n=== All 10 plots generated successfully ===")
