import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 14})
# # Load data from CSV files
# data1 = pd.read_csv('lamda1.csv')
# data2 = pd.read_csv('ours.csv')
# data3 = pd.read_csv('lamda3.csv')
# data4 = pd.read_csv('loss.csv')

# # Assuming each file contains a single column and you know the column names
# # Here, I'm directly using the column names as provided in your description
# data = [data1['loss'], data2['loss'], data3['loss'], data4['loss']]
# labels = ['lamda=1', 'lamda=2', 'lamda=3', 'W/O penalty loss']

# # Colors specified in RGB, scaled to [0, 1]
# colors = [(212/255, 194/255, 162/255), (134/255, 181/255, 162/255), 
#           (216/255, 153/255, 123/255), (153/255, 163/255, 192/255)]

# # Create a boxplot
# plt.figure(figsize=(10, 6))
# bplot = plt.boxplot(data, patch_artist=True, labels=labels)  # Custom labels for the boxplot

# # Assigning colors to each box
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# # Adding title and labels for clarity
# # plt.title('Comparison of Different Methods')
# plt.ylabel('MSE loss')
# plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.75)

# plt.yscale('log')
# # Display the plot
# plt.show()



# Load data from CSV files
data1_mse = pd.read_csv('lamda1_mse.csv')
data1_pcc = pd.read_csv('lamda1_pcc.csv')
data1_ssim= pd.read_csv('lamda1_ssim.csv')
data2_mse = pd.read_csv('ours_mse.csv')
data2_pcc= pd.read_csv('ours_pcc.csv')
data2_ssim= pd.read_csv('ours_ssim.csv')
data3_mse = pd.read_csv('lamda3_mse.csv')
data3_pcc = pd.read_csv('lamda3_pcc.csv')
data3_ssim= pd.read_csv('lamda3_ssim.csv')
data4_mse = pd.read_csv('loss_mse.csv')
data4_pcc = pd.read_csv('loss_pcc.csv')
data4_ssim= pd.read_csv('loss_ssim.csv')
# Assuming each CSV contains columns 'mse', 'pcc', 'SSIM'
metrics = ['loss']
colors = [(212/255, 194/255, 162/255), (134/255, 181/255, 162/255), 
          (216/255, 153/255, 123/255), (153/255, 163/255, 192/255)]
labels = ['MSE', 'PCC', 'SSIM']
names = ['$\lambda$=1', '$\lambda$=2', '$\lambda$=3', 'w/o SR']
# Prepare data for plotting
data = []

data.append([
        data1_mse['loss'].dropna(),
        data2_mse['loss'].dropna(),
        data3_mse['loss'].dropna(),
        data4_mse['loss'].dropna()
    ])
data.append([
        data1_pcc['loss'].dropna(),
        data2_pcc['loss'].dropna(),
        data3_pcc['loss'].dropna(),
        data4_pcc['loss'].dropna()
    ])
data.append([
        data1_ssim['loss'].dropna(),
        data2_ssim['loss'].dropna(),
        data3_ssim['loss'].dropna(),
        data4_ssim['loss'].dropna()
    ])
# Create the boxplot
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)

for idx, ax in enumerate(axs):
    bplot = ax.boxplot(data[idx], patch_artist=True, labels=names,whis=[0.001, 99.999])
    ax.set_title(labels[idx], fontsize=14)
    
    # Assigning colors to each box
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.75)
# Create custom legend
from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor=color, label=label) for color, label in zip(colors, names)]
# fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
plt.yscale('log')
# General layout adjustments
plt.tight_layout()
plt.show()