# Data_Analysis
[mymoviedb.csv](https://github.com/user-attachments/files/21808595/mymoviedb.csv)

[projectnetflix.py](https://github.com/user-attachments/files/21808606/projectnetflix.py)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    r"C:\Users\ELECTRO LINKS\Downloads\mymoviedb.csv", encoding="latin1", lineterminator='\n')
print(df.head(10))

print("Information")
print(df.info())

print("First 10 Genre")
print(df['Genre'].head(10))

print("Duplicate Values?")
print(df.duplicated().sum())

print("Descriptive Statistics")
print(df.describe())


df.drop(columns=['Overview', 'Original_Language', 'Poster_Url'], inplace=True)

df['Release_Date'] = pd.to_datetime(df['Release_Date'])
print(df['Release_Date'].dtype)

df['Release_Date'] = df['Release_Date'].dt.year
df['Release_Date'].dtypes

print(df.head(10))


df['Vote_Average_Numeric'] = pd.to_numeric(df['Vote_Average'], errors='coerce')


def categorize_col(df, col, labels):
    # Convert column to numbers
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Find bin edges
    stats = df[col].describe()
    edges = [stats['min'], stats['25%'],
             stats['50%'], stats['75%'], stats['max']]

    # Cut into bins (no duplicate handling for now)
    df[col] = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)

    return df


labels = ['Un_popular', 'Below_avg', 'Average', 'Popular']
categorize_col(df, 'Vote_Average', labels)

df['Vote_Average'].unique()

print(df.head(10))

print("Statistics of all movies Vote_Average")
print(df['Vote_Average'].value_counts())

df.dropna(inplace=True)
print("Duplicate/missing values")
print(df.isna().sum())

df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop=True)

print(df.head(10))

# df['Genre'] = df['Genre'].astype('Category')

avg_votes_by_year = df.groupby(['Release_Date', 'Genre'], as_index=False)[
    'Vote_Average_Numeric'].mean()
sns.set_palette("pastel")
sns.set_style('whitegrid')

sns.scatterplot(data=avg_votes_by_year,
                x='Release_Date',
                y='Vote_Average_Numeric',
                hue='Genre',
                legend=False,
                alpha=0.5
                )
plt.title("Trends over the Years")
plt.xlabel('Years')
plt.ylabel('Average Votes')
plt.show()


top_5_genres = df.groupby('Genre')['Vote_Average_Numeric'].mean().sort_values(
    ascending=False).head(5)
plt.barh(top_5_genres.index, top_5_genres.values,
         color=['blue', 'pink', 'orange', 'yellow', 'purple'])
plt.xlabel('Genre')
plt.ylabel('Average Votes')
plt.title('Top 5 genre')
plt.grid(color='black', linewidth=0.5, linestyle=':')

plt.show()

movie_count = df.groupby('Release_Date')[
    'Title'].count().sort_values(ascending=True).head(10)
plt.barh(movie_count.index, movie_count.values, color=[
         'black', 'blue', 'green', 'pink', 'orange'])
plt.xlabel("Years")
plt.ylabel("Movies released")
plt.title("Movies Released per year")
plt.grid(color='black', linewidth=0.5, linestyle=':')
plt.show()


# Group the data
movie_per_year = df.groupby('Release_Date')['Title'].count()
avg_votes_per_year = df.groupby('Release_Date')['Vote_Average_Numeric'].mean()

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 7))

# Bar plot for movies released
bars = ax1.bar(movie_per_year.index, movie_per_year.values,
               color='skyblue', alpha=0.7, label='Movies Released')
ax1.set_xlabel("Year", fontsize=14, fontweight='bold')
ax1.set_ylabel("Number of Movies", fontsize=12, color='navy')
ax1.tick_params(axis='y', labelcolor='navy')
ax1.tick_params(axis='x', rotation=45)

# Second axis for average votes
ax2 = ax1.twinx()
line = ax2.plot(avg_votes_per_year.index, avg_votes_per_year.values,
                color='crimson', marker='o', linewidth=2.5, markersize=6,
                label='Average Votes')
ax2.set_ylabel('Average Vote', fontsize=12, color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

# Title & grid
plt.title("Movies Released vs Average Votes per Year",
          fontsize=16, fontweight='bold', pad=20)
ax1.grid(color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

# Combine legends from both axes
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc="upper left",
           bbox_to_anchor=(0.08, 0.93), fontsize=11)

# Make layout clean
plt.tight_layout()

# Show plot
plt.show()


genre_count = df['Genre'].value_counts().head(
    10).reset_index()  # this is to make cols work
genre_count.columns = ['Genre', 'Count']

sns.set_palette('pastel')
sns.set_style('whitegrid')

sns.barplot(data=genre_count, x='Count', y='Genre',
            palette='pastel'
            )
plt.xlabel('No Of Movies')
plt.ylabel("Genre")
plt.title("Top 10 Genres")
plt.show()

top_5 = df['Genre'].value_counts().head().index
df_top = df[df['Genre'].isin(top_5)]
sns.set_palette("pastel")
sns.set_style("whitegrid")

sns.scatterplot(data=df_top,
                x='Popularity', y='Vote_Average_Numeric',
                hue='Genre',
                legend=True,
                alpha=0.5

                )
plt.xlabel("Popularity")
plt.ylabel("Votes Average")
plt.title("Popularity vs Vote Average")
plt.show()


genre_stats = df.groupby('Genre').agg({
    'Vote_Average_Numeric': 'mean',
    'Popularity': 'mean',
    'Title': 'count'
}).reset_index()
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    genre_stats['Vote_Average_Numeric'],
    genre_stats['Popularity'],
    s=genre_stats['Title'] * 20,  # bubble size
    alpha=0.6,
    c=range(len(genre_stats)),
    cmap='tab10'
)
for i, txt in enumerate(genre_stats['Genre']):
    plt.annotate(
        txt, (genre_stats['Vote_Average_Numeric'][i], genre_stats['Popularity'][i]))
plt.xlabel("Average Vote")
plt.ylabel("Average Popularity")
plt.title("Genre Popularity vs Average Vote (Bubble Size = No. of Movies)")
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()


plt.hist(df['Popularity'], bins=50, color='pink',
         edgecolor="black")
plt.title("Popularity Scale")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.yscale('log')
plt.show()
