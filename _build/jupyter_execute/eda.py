#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis

# In this exploratory data analysis, we will examine the structure and characteristics of the dataset, including its size, distribution of variables, and any missing or abnormal values. We will also create visualizations to help identify patterns and relationships between variables. The insights gained from this analysis will be used to guide the development of hypotheses and the choice of statistical models for our subsequent analysis.

# ## Data Preparation

# In[1]:


get_ipython().run_line_magic('run', 'functions.py')


# In[2]:


# set global random seed
rand = 3
os.environ['PYTHONHASHSEED']=str(rand)
np.random.seed(rand)


# We use the `Compas-scores-two-years` dataset published by ProPublica and apply the same data filter used by them to create a new dataframe compas_df. The dataset contains information on defendants charged with a crime and assessed using the COMPAS risk assessment tool. The data includes demographic information, criminal history, and the results of the COMPAS assessment, including a predicted risk score and likelihood of recidivism over a two-year period. The filter selects only the rows from the data where the number of days between the arrest and the screening is within the range of -30 to 30, the value in the is_recid column is not -1, the value in the c_charge_degree column is not "O", and the value in the score_text column is not "N/A".

# In[3]:


dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
raw_data = pd.read_csv(dataURL)
raw_data.columns


# In[4]:


compas_df = raw_data.loc[
    (raw_data['days_b_screening_arrest'] <= 30) &
    (raw_data['days_b_screening_arrest'] >= -30) &
    (raw_data['is_recid'] != -1) &
    (raw_data['c_charge_degree'] != "O") &
    (raw_data['score_text'] != "N/A")
]
compas_df.shape


# ### Race Variables

# Initially, we assess the distribution of Race Variables 

# In[5]:


compas_df['race'].value_counts().plot(
    title = "Defendants by Race Distribution",
    kind= "barh", 
    color = "#f17775")


# It can be clearly seen from the graph that the sample is unbalanced with regards to the race feature, where the sample data of COMPAS have extremely small representation of Asian and Native American. According to the US Census data, Asians make up about 5.23% of the nation’s overall population in 2014; in the ProPublica, however, they accounts for only 0.5% of the data.

# In[6]:


value_counts = compas_df['race'].value_counts()
percentage = np.round(value_counts / len(compas_df) * 100,3)

table = pd.concat([value_counts, percentage], axis=1)
table.columns = ['Value Counts', 'Percentage']
print("Race Distribution of Defendants")
print(table)


# ### Sex Variables

# That there is a significant imbalance in the distribution of males and females in the sample population, with males accounting for approximately 81% of the sample and females accounting for approximately 19%. 

# In[7]:


compas_df['sex'].value_counts().plot(
    title = "Defendants by Sex Distribution",
    kind= "barh", 
    color = "#f17775")


# In[8]:


value_counts = compas_df['sex'].value_counts()
percentage = np.round(value_counts / len(compas_df) * 100,3)

table = pd.concat([value_counts, percentage], axis=1)
table.columns = ['Value Counts', 'Percentage']
print("Sex Distribution of Defendants")
print(table)


# In[9]:


df['two_year_recid'].value_counts()


# ### Age Variable

# Histogram shows that most of the defendants in the COMPAS data are between the ages of 25 and 35, with a concentration of ages towards the younger end of this range. This type of distribution can indicate that the population of defendants in the COMPAS data is relatively young, with fewer older defendants. 

# In[90]:


plt.hist(compas_df['age'], bins=20, color='#ffab40')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Given that African American and Caucasian groups are the predominant populations in our dataset, we will analyze age distributions within these two populations to gain a deeper understanding of their demographic structures.

# In[25]:


# to customize the histogram color
import matplotlib.cm as cm
import matplotlib.colors as colors

def hist_color(color_code_start, color_code_end, n):
    cmap = cm.colors.LinearSegmentedColormap.from_list("MyColorMap", [color_code_start, color_code_end], N = n)
    colors_array = cmap(colors.Normalize()(range(n)))
    color_codes = [colors.to_hex(c) for c in colors_array]
    return color_codes


# In[68]:


min_age = int(compas_df['age'].min())
max_age = int(compas_df['age'].max())
age_bins = np.arange(min_age, max_age+1, 5)

compas_df[compas_df['race'].isin(["African-American","Caucasian"])].groupby('race')['age'].apply(lambda x: pd.cut(x, bins = age_bins, right = False).value_counts(normalize = False)).unstack().plot(
    kind='bar', 
    color = hist_color('#fff0db', '#f38800', 20),
    figsize=(15, 7),
    title='Age Distribution by Race', 
    ylabel='Number of Defendants'
)


# ### Decile Scores

# Next, we examine the distribution of COMPAS decile scores across different racial groups in relation to recidivism.

# In[8]:


score_text_by_race = compas_df.groupby(['race','score_text'], sort = True)['id'].size()
score_text_by_race.unstack().sort_values(by="race", ascending=False).plot(
    kind = "barh", 
    title = "Score Text by Race",
    ylabel = "Score Text",
    xlabel = "Race",
    color = ["#f17775", "#016d8c", "#ffab40"],
    figsize = (8,4),
    stacked = True
)


# African American defendants have the highest number of individuals classified with high scores. However, it is important to consider that African Americans make up the largest racial group in this dataset, representing 51.442% of the total. To better understand the distribution of decile scores among different racial groups, we plotted the histogram as below. The chart eveals a noticeable pattern. As the decile score increases, the proportion of defendants from both the Caucasian and 'Other' racial groups decreases. However, this trend is not observed in the African American group, where the proportion of defendants remains relatively stable across different decile scores.

# In[72]:


compas_df.groupby('race')['decile_score'].value_counts(normalize = False).unstack().plot(
    kind='bar', 
    color = hist_color('#e3e9ed', '#016d8c',10),
    figsize=(18, 6),
    title='Decile Score Histogram by Race'
)


# ### Priors_Count Variable

# In[62]:


plt.hist(compas_df['priors_count'], bins=20, color='#016d8c')
plt.title("Priors Distribution")
plt.xlabel("Priors")
plt.ylabel("Frequency")
plt.show()


# In[70]:


min_priors = int(compas_df['priors_count'].min())
max_priors = int(compas_df['priors_count'].max())
count_bins = np.arange(min_priors, max_priors + 1, 2)

compas_df[compas_df['race'].isin(["African-American","Caucasian"])].groupby('race')['priors_count'].apply(lambda x: pd.cut(x, bins = count_bins, right = False).value_counts(normalize = False)).unstack().plot(
    kind='bar', 
    color = hist_color('#e3e9ed', '#016d8c',20),
    figsize=(15, 7),
    title='"Priors Count" Distribution by Race', 
    ylabel='Number of Defendants'
)


# In[86]:


import matplotlib.colors as mcolors

colors = ['#fdf2f2', '#f17775']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

plt.figure(figsize=(15, 10))
sns.heatmap(compas_df[['age', 'race', 'sex', 'decile_score', 'priors_count', 'two_year_recid']].corr(), annot=True, cmap=cmap, vmin=-1, vmax=1)
plt.title('Heatmap of Correlation between Features')
plt.show()


# In this exploratory data analysis, we will examine the structure and characteristics of the dataset, including its size, distribution of variables, and any missing or abnormal values. We will also create visualizations to help identify patterns and relationships between variables. The insights gained from this analysis will be used to guide the development of hypotheses and the choice of statistical models for our subsequent analysis.

# In[6]:


compas_df['race'].value_counts().plot(
    title = "Defendants by Race Distribution",
    kind= "barh", 
    color = "#ffab40")


# It can be clearly seen from the graph that the sample is unbalanced with regards to the race feature, where the sample data of COMPAS have extremely small representation of Asian and Native American. According to the US Census data, Asians make up about 5.23% of the nation’s overall population in 2014; in the ProPublica, however, they accounts for only 0.5% of the data.

# In[7]:


value_counts = compas_df['race'].value_counts()
percentage = np.round(value_counts / len(compas_df) * 100,3)

table = pd.concat([value_counts, percentage], axis=1)
table.columns = ['Value Counts', 'Percentage']
print("Race Distribution of Defendants")
print(table)


# Next, we examine the distribution of COMPAS decile scores across different racial groups in relation to recidivism.

# In[8]:


score_text_by_race = compas_df.groupby(['race','score_text'], sort = True)['id'].size()
score_text_by_race.unstack().sort_values(by="race", ascending=False).plot(
    kind = "barh", 
    title = "Score Text by Race",
    ylabel = "Score Text",
    xlabel = "Race",
    color = ["#f17775", "#016d8c", "#ffab40"],
    figsize = (8,4),
    stacked = True
)


# African American defendants have the highest number of individuals classified with high scores. However, it is important to consider that African Americans make up the largest racial group in this dataset, representing 51.442% of the total.

# In[9]:


import matplotlib.cm as cm
import matplotlib.colors as colors

cmap = cm.colors.LinearSegmentedColormap.from_list("MyColorMap", ['#F8BBBA', '#A31310'], N=10)
colors_array = cmap(colors.Normalize()(range(10)))
color_codes = [colors.to_hex(c) for c in colors_array]

compas_df.groupby('race')['decile_score'].value_counts(normalize=True).unstack().plot(
    kind='bar', 
    color = color_codes,
    figsize=(20, 7),
    title='Decile Score Histogram by Race', 
    ylabel='% with Decile Score'
)


# In[10]:


plt.hist(compas_df['age'], bins=20, color='#ffab40')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Histogram shows that most of the defendants in the COMPAS data are between the ages of 25 and 35, with a concentration of ages towards the younger end of this range. This type of distribution can indicate that the population of defendants in the COMPAS data is relatively young, with fewer older defendants. 

# Markus Waser, “Nonlinear Dependencies in and between Time Series” retrieved from: https://publik.tuwien.ac.at/files/PubDat_189752.pdf<br>
# 
# Patrick Hall, “Predictive modeling: Striking a balance between accuracy and interpretability”. Retrieved from: https://www.oreilly.com/content/predictive-modeling-striking-a-balance-between-accuracy-and-interpretability/<br>
# 
# Chunrong Ai and Edward C. Norton, “Computing interaction effects and standard errors in logit and Probit models” retrieved from: https://hhstokes.people.uic.edu/ftp/e535/probit/NortonWangAi.pdf<br>
# 
# Jerome H. Friedman, "Greedy function approximation: A gradient boosting machine.." Ann. Statist. 29 (5) 1189 - 1232, October 2001. Retrieved from: https://doi.org/10.1214/aos/1013203451<br>
# 
# Molnar, Christoph. “Interpretable Machine Learning: A Guide for Making Black Box Models Explainable” (2022): Chapter: Permutation Feature Importance
# Slundberg/shap · GitHub<br>
# 
# Rudin, Cynthia. “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and use Interpretable Models Instead” (2019): Nature Machine Intelligence. Retrieved from: https://arxiv.org/abs/1811.10154 <br>
# 
# Ai, C., & Norton, E. C. (2004). Computing interaction effects and standard errors in logit and Probit models. Retrieved January 15, 2023, from https://hhstokes.people.uic.edu/ftp/e535/probit/NortonWangAi.pdf.
# 
# Oh, S. (2019). Feature Interaction in Terms of Prediction Performance. Applied Sciences, 9(23), 5191. https://doi.org/10.3390/app9235191
# 
# Friedman, Jerome H. (2001): Greedy function approximation: A gradient boosting machine… Ann. Statist. 29 (5) 1189 - 1232. Retrieved January 15, 2023, from https://doi.org/10.1214/aos/1013203451/
# 
# Molnar, Christoph (2022): Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. Retrieved January 12, 2023, from  https://christophm.github.io/interpretable-ml-book/feature-importance.html
# 
# Lundberg, S. (2020, December 18). Welcome to the SHAP documentation Retrieved from https://github.com/slundberg/shap/blob/master/docs/index.rst
