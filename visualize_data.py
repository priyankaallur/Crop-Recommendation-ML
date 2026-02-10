import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# ------------------ GRAPH 1: CROP DISTRIBUTION ------------------
plt.figure(figsize=(8, 6))
sns.countplot(y="label", data=data)
plt.title("Crop Distribution")
plt.tight_layout()
plt.savefig("crop_distribution.png")
plt.close()

# ------------------ GRAPH 2: TEMPERATURE VS CROP ------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="label", y="temperature", data=data)
plt.xticks(rotation=90)
plt.title("Temperature vs Crop")
plt.tight_layout()
plt.savefig("temperature_vs_crop.png")
plt.close()

# ------------------ GRAPH 3: RAINFALL VS CROP ------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="label", y="rainfall", data=data)
plt.xticks(rotation=90)
plt.title("Rainfall vs Crop")
plt.tight_layout()
plt.savefig("rainfall_vs_crop.png")
plt.close()

print("Graphs saved successfully as images")