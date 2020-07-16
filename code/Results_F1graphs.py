# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:27:24 2020

@author: Melanie Marochov


Results graphs


"""

""" Imports """
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

"""  Data   """
Helheim_RGBmodel_data = pd.read_csv("D:\S2_Images\Results_H13_09_19_3000px\Hel_RGBmodel_data.csv")
Helheim_RGBTLmodel_data = pd.read_csv("D:\S2_Images\Results_H13_09_19_3000px\Hel_RGBTLmodel_data.csv")
Helheim_RGBNIRmodel_data = pd.read_csv("D:\S2_Images\Results_H13_09_19_3000px\Hel_RGBNIRmodel_data.csv")
Scoresby_RGBmodel_data = pd.read_csv("D:\S2_Images\Results_H13_09_19_3000px\Sco_RGBmodel_data.csv")
Scoresby_RGBTLmodel_data = pd.read_csv("D:\S2_Images\Results_H13_09_19_3000px\Sco_RGBTLmodel_data.csv")
Scoresby_RGBNIRmodel_data = pd.read_csv("D:\S2_Images\Results_H13_09_19_3000px\Sco_RGBNIRmodel_data.csv")
#data.head()

Filename= "D:\S2_Images\Results_Figures\F1_Scores.png"

"""   Plot data in swarmplot   """    

order= {50:"50",75:"75",100:"100"} 
order=list(order)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10,12))   
              
sns.swarmplot(x="Patch Size",
              y="F1 Score",
              hue="Tile Size", hue_order=order,
              data=Helheim_RGBmodel_data, 
              palette="YlGnBu",
              size=6, edgecolor="grey", linewidth=0.2,
              ax=ax1)
ax1.set_title('A: Helheim RGB Model', loc="left")
ax1.set_ylim(0.7,1)
ax1.set_xlabel("Patch Size",fontsize=12)
ax1.set_ylabel("F1 Score",fontsize=12)
ax1.tick_params(axis='both', labelsize=11)
sns.set_style("ticks")
ax1.legend_.remove()

sns.swarmplot(x="Patch Size",
              y="F1 Score",
              hue="Tile Size", hue_order=order,
              data=Scoresby_RGBmodel_data, 
              palette="YlGnBu",
              size=6, edgecolor="grey", linewidth=0.2,
              ax=ax2)
ax2.set_title('B: Scoresby RGB Model', loc="left") 
ax2.set_ylim(0.7,1)
ax2.set_xlabel("Patch Size",fontsize=12)
ax2.set_ylabel("F1 Score",fontsize=12)
ax2.tick_params(axis='both', labelsize=11)
ax2.legend_.remove()

sns.swarmplot(x="Patch Size",
              y="F1 Score",
              hue="Tile Size", hue_order=order,
              data=Helheim_RGBTLmodel_data, 
              palette="YlGnBu",
              size=6, edgecolor="grey", linewidth=0.2,
              ax=ax3)
ax3.set_title('C: Helheim RGB Transfer Learning Model', loc="left")
ax3.set_ylim(0.7,1)
ax3.set_xlabel("Patch Size",fontsize=12)
ax3.set_ylabel("F1 Score",fontsize=12)
ax3.tick_params(axis='both', labelsize=11)
ax3.legend_.remove()

sns.swarmplot(x="Patch Size",
              y="F1 Score",
              hue="Tile Size", hue_order=order,
              data=Scoresby_RGBTLmodel_data, 
              palette="YlGnBu",
              size=6, edgecolor="grey", linewidth=0.2,
              ax=ax4)
ax4.set_title('D: Scoresby RGB Transfer Learning Model', loc="left")
ax4.set_ylim(0.7,1)
ax4.set_xlabel("Patch Size",fontsize=12)
ax4.set_ylabel("F1 Score",fontsize=12)
ax4.tick_params(axis='both', labelsize=11)
ax4.legend_.remove()

sns.swarmplot(x="Patch Size",
              y="F1 Score",
              hue="Tile Size", hue_order=order,
              data=Helheim_RGBNIRmodel_data, 
              palette="YlGnBu",
              size=6, edgecolor="grey", linewidth=0.2,
              ax=ax5)
ax5.set_title('E: Helheim RGB NIR Model', loc="left")
ax5.set_ylim(0.7,1)
ax5.set_xlabel("Patch Size",fontsize=12)
ax5.set_ylabel("F1 Score",fontsize=12)
ax5.tick_params(axis='both', labelsize=11)
ax5.legend_.remove()

sns.swarmplot(x="Patch Size",
              y="F1 Score",
              hue="Tile Size", hue_order=order,
              data=Scoresby_RGBNIRmodel_data, 
              palette="YlGnBu",
              size=6, edgecolor="grey", linewidth=0.2,
              ax=ax6)
ax6.set_title('F: Scoresby RGB NIR Model', loc="left")
ax6.set_ylim(0.7,1)
ax6.set_xlabel("Patch Size",fontsize=12)
ax6.set_ylabel("F1 Score",fontsize=12)
ax6.tick_params(axis='both', labelsize=11)
ax6.legend_.remove()

plt.subplots_adjust(bottom=0.85)
handles, labels = ax1.get_legend_handles_labels()

Legend_height = 0.05
fig.tight_layout(rect=(0, Legend_height, 1, 1), h_pad=0.5, w_pad=0.5)

fig.legend(handles, labels, loc='lower center', ncol=3, title="Tile Size", title_fontsize=12, prop={'size': 12})
plt.savefig(Filename, dpi=2000)

#To plot as individual plots

#sns.set_style("ticks")
#plt.ylim(0.7,1)
#a = sns.swarmplot(x="Patch Size",
#              y="F1 Score",
#              hue="Tile Size", hue_order=order,
#              data=Helheim_RGBmodel_data, 
#              palette="YlGnBu",
#              size=6, edgecolor="grey", linewidth=0.2)
#plt.legend(loc="lower right", title="Tile Size")
#plt.title('A: Helheim RGB model', loc="left")
#a.set_xlabel("Patch Size",fontsize=12)
#a.set_ylabel("F1 Score",fontsize=12)
#a.tick_params(labelsize=12)













