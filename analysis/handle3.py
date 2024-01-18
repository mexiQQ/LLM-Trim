import re
import pandas as pd
import matplotlib.pyplot as plt


# Initial data loading and examination
with open('/home/jli265/projects/LLM-Pruner/exper3.txt', 'r') as file:
    content = file.readlines()

# Function to extract data from a single line
def extract_data(line):
    # Extracting numbers from the line
    numbers = re.findall(r"\s[\d.]+", line)
    if numbers:
        return float(numbers[0])
    else:
        return None

# Data extraction process
modes = []
ratios = []
wikitext2_perplexity = []
ptb_perplexity = []

current_mode = None
current_ratio = None
for line in content:
    if 'Start Pruning Model' in line:
        mode = re.search(r"Local Mode: (\w+)", line)
        ratio = re.search(r"Pruning Ratio: ([\d.]+)", line)
        current_mode = mode.group(1) if mode else None
        current_ratio = float(ratio.group(1)) if ratio else None
    elif 'wikitext2' in line and 'ptb' in line:
        wikitext2_value = extract_data(line.split(',')[0])
        ptb_value = extract_data(line.split(',')[1])
        if wikitext2_value is not None and ptb_value is not None:
            modes.append(current_mode)
            ratios.append(current_ratio)
            wikitext2_perplexity.append(wikitext2_value)
            ptb_perplexity.append(ptb_value)

# Creating a DataFrame
corrected_data = pd.DataFrame({
    'Mode': modes,
    'Ratio': ratios,
    'Wikitext2': wikitext2_perplexity,
    'PTB': ptb_perplexity
})

# Graph plotting
# Adjusting the plots to have clearer distinctions between different modes and focusing on specific ranges of pruning ratios

# Colors for different modes
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

corrected_data = corrected_data[corrected_data['Mode'].isin([
    "random",
    # "k_proj",
    "v_proj",
    "o_proj",
    # "vo_proj",
    # "k_proj_p_1",
    "v_proj_p_1",
    "o_proj_p_1",
    # "vo_proj_p_1",
])]

####################################################################################
####################################################################################
####################################################################################
####################################################################################

# First set of plots for the entire range of pruning ratios (0.2 to 0.8)
# Plot for wikitext2 (Full Range)
plt.figure(figsize=(12, 6))
for i, mode in enumerate(corrected_data['Mode'].unique()):
    subset = corrected_data[corrected_data['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['Wikitext2'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on Wikitext2 Dataset (Full Range)')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure1.png")

# Plot for ptb (Full Range)
plt.figure(figsize=(12, 6))
for i, mode in enumerate(corrected_data['Mode'].unique()):
    subset = corrected_data[corrected_data['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['PTB'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on PTB Dataset (Full Range)')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure2.png")

####################################################################################
####################################################################################
####################################################################################
####################################################################################

# Second set of plots focusing on the range of pruning ratios from 0.2 to 0.5

# Filter the data for the specified range
filtered_data = corrected_data[corrected_data['Ratio'] <= 0.5]

# Plot for wikitext2 (0.2 to 0.5)
plt.figure(figsize=(12, 6))
for i, mode in enumerate(filtered_data['Mode'].unique()):
    subset = filtered_data[filtered_data['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['Wikitext2'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on Wikitext2 Dataset (0.2 to 0.5)')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure3.png")

# Plot for ptb (0.2 to 0.5)
plt.figure(figsize=(12, 6))
for i, mode in enumerate(filtered_data['Mode'].unique()):
    subset = filtered_data[filtered_data['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['PTB'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on PTB Dataset (0.2 to 0.5)')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure4.png")

####################################################################################
####################################################################################
####################################################################################
####################################################################################

# Filtering out specific modes
modes_to_exclude = ['kq_proj', 'l2', 'kqo_proj', 'q_proj', 'kqv_proj']
filtered_data_excluding_modes = filtered_data[~filtered_data['Mode'].isin(modes_to_exclude)]

# Plot for wikitext2 (0.2 to 0.5) excluding specific modes
plt.figure(figsize=(12, 6))
for i, mode in enumerate(filtered_data_excluding_modes['Mode'].unique()):
    subset = filtered_data_excluding_modes[filtered_data_excluding_modes['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['Wikitext2'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on Wikitext2 Dataset (0.2 to 0.5) - Selected Modes')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure5.png")

# Plot for ptb (0.2 to 0.5) excluding specific modes
plt.figure(figsize=(12, 6))
for i, mode in enumerate(filtered_data_excluding_modes['Mode'].unique()):
    subset = filtered_data_excluding_modes[filtered_data_excluding_modes['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['PTB'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on PTB Dataset (0.2 to 0.5) - Selected Modes')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure6.png")

####################################################################################
####################################################################################
####################################################################################
####################################################################################

# Filtering out specific modes from the full data set (not just 0.2 to 0.8 range)
full_data_excluding_modes = corrected_data[corrected_data['Mode'].isin([
    "random",
    # "k_proj",
    "v_proj",
    "o_proj",
    # "vo_proj",
    # "k_proj_p_1",
    "v_proj_p_1",
    "o_proj_p_1",
    # "vo_proj_p_1",
])]

# Plot for wikitext2 (All Ratios) excluding specific modes with log scale
plt.figure(figsize=(12, 6))
for i, mode in enumerate(full_data_excluding_modes['Mode'].unique()):
    subset = full_data_excluding_modes[full_data_excluding_modes['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['Wikitext2'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on Wikitext2 Dataset (All Ratios) - Selected Modes with Log Scale')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure7.png")

# Plot for ptb (All Ratios) excluding specific modes with log scale
plt.figure(figsize=(12, 6))
for i, mode in enumerate(full_data_excluding_modes['Mode'].unique()):
    subset = full_data_excluding_modes[full_data_excluding_modes['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['PTB'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on PTB Dataset (All Ratios) - Selected Modes with Log Scale')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity (Log Scale)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig("./mlp+attn+random/figure8.png")

plt.figure(figsize=(12, 6))
for i, mode in enumerate(full_data_excluding_modes['Mode'].unique()):
    subset = full_data_excluding_modes[full_data_excluding_modes['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['Wikitext2'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on Wikitext2 Dataset (All Ratios) - Selected Modes with Log Scale (Base 2)')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity (Log Scale, Base 2)')
plt.yscale('log', base=2)
plt.legend()
plt.grid(True, which="both", ls="-")
plt.savefig("./mlp+attn+random/figure9.png")

# Plot for ptb (All Ratios) excluding specific modes with log scale (base 2)
plt.figure(figsize=(12, 6))
for i, mode in enumerate(full_data_excluding_modes['Mode'].unique()):
    subset = full_data_excluding_modes[full_data_excluding_modes['Mode'] == mode]
    plt.plot(subset['Ratio'], subset['PTB'], marker='o', color=colors[i % len(colors)], label=mode)

plt.title('Performance on PTB Dataset (All Ratios) - Selected Modes with Log Scale (Base 2)')
plt.xlabel('Pruning Ratio')
plt.ylabel('Perplexity (Log Scale, Base 2)')
plt.yscale('log', base=2)
plt.legend()
plt.grid(True, which="both", ls="-")
plt.savefig("./mlp+attn+random/figure10.png")