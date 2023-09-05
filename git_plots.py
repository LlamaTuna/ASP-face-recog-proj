import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

# Read the JSON file
with open('face_finder_gitstats.json', 'r') as file:
    data = json.load(file)

# Extract commit data
commit_data = data['commits']

# Convert to DataFrame
df = pd.DataFrame(commit_data, columns=["Name", "Email", "Date"])

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort the dataframe by Date
df = df.sort_values('Date')

# Generate cumulative count of commits for plotting
df['Cumulative Commits'] = range(1, len(df) + 1)

# Plotting
plt.figure(figsize=(10, 6))

# Plot cumulative commits for each contributor
plt.plot(df['Date'], df['Cumulative Commits'], label='Total Commits', color='blue', marker='o')

# Beautify the x-labels (the dates)
plt.gcf().autofmt_xdate()

# Show the legend
plt.legend(loc='upper left')

# Set title and labels
plt.title('Cumulative Commits Over Time for Face-Finder Project')
plt.xlabel('Date')
plt.ylabel('Number of Commits')

# Display the plot
plt.tight_layout()
plt.grid(True, which="both", ls="--", c='0.7')
plt.show()
