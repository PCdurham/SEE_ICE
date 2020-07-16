# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:30:22 2020

@author: Melanie Marochov
"""

""" Imports """
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

#Data and inputs
Epoch_tuning_data = pd.read_csv("D:\S2_Images\Models\Tiles50_outputs\EpochTuning_BigTif_RGB_50_HP7new.csv")
Filename = "D:\S2_Images\Results_Figures\Epoch_tuning_BigTif_RGB_50.png"

#Add 1 to the Epoch column to match true number of Epochs
Epoch_tuning_data["Unnamed: 0"] = Epoch_tuning_data["Unnamed: 0"]+1

#Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))   

sns.set_style("ticks")

l = sns.scatterplot(x="Unnamed: 0", y="loss",
              data=Epoch_tuning_data, 
              color="#c7e9b4",
              s=40, edgecolor="grey",linewidth=0.2, ax=ax1, label="Training Loss")

vl = sns.lineplot(x="Unnamed: 0", y="val_loss", 
             data=Epoch_tuning_data, 
             c="grey", ax=ax1, label="Validation Loss")

ax1.set_xlabel("Epochs",fontsize=12)
ax1.set_ylabel("Loss",fontsize=12)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_title("A: Training and Validation Loss", loc="left")
handles, labels = ax1.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax1.legend(handles, labels)

sns.scatterplot(x="Unnamed: 0",y="accuracy", 
              data=Epoch_tuning_data, 
              color="#41b6c4",
              s=40, edgecolor="grey",linewidth=0.2, ax=ax2, label="Training Accuracy")

sns.lineplot(x="Unnamed: 0", y="val_accuracy", 
             data=Epoch_tuning_data, 
             c="grey", ax=ax2, label="Validation Accuracy")

ax2.set_xlabel("Epochs",fontsize=12)
ax2.set_ylabel("Accuracy",fontsize=12)
ax2.tick_params(axis='both', labelsize=11)
ax2.set_title("B: Training and Validation Accuracy", loc="left")
handles, labels = ax2.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
ax2.legend(handles, labels, loc="lower right")


plt.savefig(Filename, dpi=900)