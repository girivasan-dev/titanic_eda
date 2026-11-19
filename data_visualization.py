import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Four guiding questions
questions = [
    "1. How are total bills distributed across meals?",
    "2. Do tips differ across days of the week?",
    "3. Does a larger total bill come with a larger tip (and how does meal time compare)?",
    "4. Which numeric variables move together most strongly?",
]
print("=== RESEARCH QUESTIONS ===")
for q in questions:
    print(q)

# Step 2: Load dataset and apply clean theme
tips = sns.load_dataset("tips")
sns.set_theme(style="whitegrid", palette="Set2")

# Step 3-6: Build 2x2 dashboard grid with titles, units, annotations, and 150 DPI save
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Top-left: Histogram (1 numeric variable)
sns.histplot(tips["total_bill"], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set(
    title="Most Total Bills Sit Between $10 and $25",
    xlabel="Total Bill (USD)",
    ylabel="Number of Meals",
)

# Top-right: Box plot (comparing groups across days)
day_order = ["Thur", "Fri", "Sat", "Sun"]
sns.boxplot(data=tips, x="day", y="tip", order=day_order, ax=axes[0, 1])
axes[0, 1].set(
    title="Median Tips Stay Close Across Days, With Higher Outliers on Weekends",
    xlabel="Day of Week",
    ylabel="Tip Amount (USD)",
)

# Bottom-left: Scatter plot (2 numeric variables + time hue)
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="time",
    alpha=0.85,
    ax=axes[1, 0],
)
axes[1, 0].set(
    title="Larger Bills Generally Bring Larger Tips",
    xlabel="Total Bill (USD)",
    ylabel="Tip Amount (USD)",
)
axes[1, 0].legend(title="Meal Time")

# Bottom-right: Annotated correlation heatmap
corr_matrix = tips.select_dtypes("number").corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=-1,
    vmax=1,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Total Bill and Tip Show the Strongest Positive Correlation")

plt.tight_layout()
output_path = "tips_overview.png"
plt.savefig(output_path, dpi=150)
print(f"\nSaved dashboard figure to: {output_path}")

# Step 7: One-line takeaways per chart
strongest_pair = (
    corr_matrix.where(~(corr_matrix == 1.0))
    .stack()
    .idxmax()
)
strongest_val = corr_matrix.loc[strongest_pair[0], strongest_pair[1]]

takeaways = [
    f"Chart 1 (Histogram): Most meals have a total bill around $10–$25, with a right skew from higher-spending tables.",
    f"Chart 2 (Box Plot): Typical tips cluster around $2–$3 across days, with larger high-tip outliers on Saturday and Sunday.",
    f"Chart 3 (Scatter Plot): Tip amounts trend upward as total bill increases across both Lunch and Dinner.",
    f"Chart 4 (Heatmap): The strongest numeric correlation is between '{strongest_pair[0]}' and '{strongest_pair[1]}' (r = {strongest_val:.2f}).",
]

print("\n=== ONE-LINE TAKEAWAYS ===")
for t in takeaways:
    print(t)