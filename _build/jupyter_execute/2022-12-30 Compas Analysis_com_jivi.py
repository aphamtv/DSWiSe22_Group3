#!/usr/bin/env python
# coding: utf-8

# # Compas Analysis
# 
# What follows are the calculations performed for ProPublica's analaysis of the COMPAS Recidivism Risk Scores. It might be helpful to open [the methodology](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm/) in another tab to understand the following.
# 
# ## Loading the Data
# 
# We select fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years.

# In[1]:


# filter dplyr warnings
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_cell_magic('R', '', 'library(dplyr)\nlibrary(ggplot2)\nraw_data <- read.csv("./compas-scores-two-years.csv")\nnrow(raw_data) #count rows\n')


# However not all of the rows are useable for the first round of analysis.
# 
# There are a number of reasons remove rows because of missing data:
# * If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
# * We coded the recidivist flag -- `is_recid` -- to be -1 if we could not find a compas case at all.
# * In a similar vein, ordinary traffic offenses -- those with a `c_charge_degree` of 'O' -- will not result in Jail time are removed (only two of them).
# * We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

# In[3]:


get_ipython().run_cell_magic('R', '', 'df <- dplyr::select(raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, \n                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% \n        filter(days_b_screening_arrest <= 30) %>%\n        filter(days_b_screening_arrest >= -30) %>%\n        filter(is_recid != -1) %>%\n        filter(c_charge_degree != "O") %>%\n        filter(score_text != \'N/A\') #assing the data from the csv to the variable df & filter it for diffrent conditions\nnrow(df)\n')


# Higher COMPAS scores are slightly correlated with a longer length of stay. 

# In[4]:


get_ipython().run_cell_magic('R', '', 'df$length_of_stay <- as.numeric(as.Date(df$c_jail_out) - as.Date(df$c_jail_in)) #creates new column\ncor(df$length_of_stay, df$decile_score) #is there a correlation between these two columns? Yes, slightly\n')


# After filtering we have the following demographic breakdown:

# In[5]:


get_ipython().run_cell_magic('R', '', 'summary(df$age_cat)\n')


# In[6]:


get_ipython().run_cell_magic('R', '', 'summary(df$race)\n')


# In[7]:


#calculating proportions
print("Black defendants: %.2f%%" %            (3175 / 6172 * 100))
print("White defendants: %.2f%%" %            (2103 / 6172 * 100))
print("Hispanic defendants: %.2f%%" %         (509  / 6172 * 100))
print("Asian defendants: %.2f%%" %            (31   / 6172 * 100))
print("Native American defendants: %.2f%%" %  (11   / 6172 * 100))


# In[8]:


get_ipython().run_cell_magic('R', '', 'summary(df$score_text)\n')


# In[9]:


get_ipython().run_cell_magic('R', '', 'xtabs(~ sex + race, data=df) #crosstab\n')


# In[10]:


get_ipython().run_cell_magic('R', '', 'summary(df$sex)\n')


# In[11]:


print("Men: %.2f%%" %   (4997 / 6172 * 100))
print("Women: %.2f%%" % (1175 / 6172 * 100))


# In[12]:


get_ipython().run_cell_magic('R', '', 'nrow(filter(df, two_year_recid == 1)) #did they recidivate again or not. 1 = yes\n')


# In[13]:


get_ipython().run_cell_magic('R', '', 'nrow(filter(df, two_year_recid == 1)) / nrow(df) * 100 # 45% of people of our dataframe recidivated\n')


# Judges are often presented with two sets of scores from the Compas system -- one that classifies people into High, Medium and Low risk, and a corresponding decile score. There is a clear downward trend in the decile scores as those scores increase for white defendants.

# In[14]:


get_ipython().run_cell_magic('R', '-w 900 -h 363 -u px', 'library(grid)\nlibrary(gridExtra)\npblack <- ggplot(data=filter(df, race =="African-American"), aes(ordered(decile_score))) + \n          geom_bar() + xlab("Decile Score") +\n          ylim(0, 650) + ggtitle("Black Defendant\'s Decile Scores") # plot for blacks\npwhite <- ggplot(data=filter(df, race =="Caucasian"), aes(ordered(decile_score))) + \n          geom_bar() + xlab("Decile Score") +\n          ylim(0, 650) + ggtitle("White Defendant\'s Decile Scores") #plot for whites\ngrid.arrange(pblack, pwhite,  ncol = 2)\n')


# In[15]:


get_ipython().run_cell_magic('R', '', 'xtabs(~ decile_score + race, data=df)\n')


# ## Racial Bias in Compas
# 
# After filtering out bad rows, our first question is whether there is a significant difference in Compas scores between races. To do so we need to change some variables into factors, and run a logistic regression, comparing low scores to high scores.

# In[16]:


get_ipython().run_cell_magic('R', '', '#Factors are the data objects which are used to categorize the data and store it as levels. They can store both strings and integers. They are useful in the columns which have a limited number of unique values. Like "Male, "Female" and True, False etc. They are useful in data analysis for statistical modeling.\ndf <- mutate(df, crime_factor = factor(c_charge_degree)) %>%\n      mutate(age_factor = as.factor(age_cat)) %>%\n      within(age_factor <- relevel(age_factor, ref = 1)) %>% #sorting / ordering the factors\n      mutate(race_factor = factor(race)) %>%\n      within(race_factor <- relevel(race_factor, ref = 3)) %>%\n      mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%\n      within(gender_factor <- relevel(gender_factor, ref = 2)) %>%\n      mutate(score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore"))) #\nmodel <- glm(score_factor ~ gender_factor + age_factor + race_factor +\n                            priors_count + crime_factor + two_year_recid, family="binomial", data=df) \n# regression = generalized linear model, by default gausian regression but here binomial regression\nsummary(model)\n\n#z-score is tge number of standard deviations away from the mean of a given population\n#https://stats.stackexchange.com/questions/86351/interpretation-of-rs-output-for-binomial-regression\n#signif. codes are only for z-value\n')


# Black defendants are 45% more likely than white defendants to receive a higher score correcting for the seriousness of their crime, previous arrests, and future criminal behavior.

# In[17]:


get_ipython().run_cell_magic('R', '', "control <- exp(-1.52554) / (1 + exp(-1.52554)) #control group (white, male, medium age) = Intercept (thought of Joana: that's qhy they did the releveling, to get this default intercept with these attributes)\nexp(0.47721) / (1 - control + (control * exp(0.47721))) #treatment group for race\n'''Controlling for a variable means estimating the difference in average outcome between a treatment group \nand a control group within a specific category/value of the controlled variable. Regression is a convenient \nestimation strategy that helps us control for confounding variables.'''\n")


# Women are 19.4% more likely than men to get a higher score.

# In[18]:


get_ipython().run_cell_magic('R', '', 'exp(0.22127) / (1 - control + (control * exp(0.22127))) #treatment group for sex\n')


# Most surprisingly, people under 25 are 2.5 times as likely to get a higher score as middle aged defendants.

# In[19]:


get_ipython().run_cell_magic('R', '#age', 'exp(1.30839) / (1 - control + (control * exp(1.30839))) #treatment group for age\n')


# ### Risk of Violent Recidivism
# 
# Compas also offers a score that aims to measure a persons risk of violent recidivism, which has a similar overall accuracy to the Recidivism score. As before, we can use a logistic regression to test for racial bias.

# In[20]:


get_ipython().run_cell_magic('R', '', 'raw_data <- read.csv("./compas-scores-two-years-violent.csv")\nnrow(raw_data)\n')


# In[21]:


get_ipython().run_cell_magic('R', '', 'df <- dplyr::select(raw_data, age, c_charge_degree, race, age_cat, v_score_text, sex, priors_count, \n                    days_b_screening_arrest, v_decile_score, is_recid, two_year_recid) %>% \n        filter(days_b_screening_arrest <= 30) %>%\n        filter(days_b_screening_arrest >= -30) %>% \n        filter(is_recid != -1) %>%\n        filter(c_charge_degree != "O") %>%\n        filter(v_score_text != \'N/A\')\nnrow(df)\n\n#diffrence here is that they take v_decile_score and v_score_text. This is why it\'s diffrent data\n')


# In[22]:


get_ipython().run_cell_magic('R', '', 'summary(df$age_cat)\n')


# In[23]:


get_ipython().run_cell_magic('R', '', 'summary(df$race)\n')


# In[24]:


get_ipython().run_cell_magic('R', '', 'summary(df$v_score_text)\n')


# In[25]:


get_ipython().run_cell_magic('R', '', 'nrow(filter(df, two_year_recid == 1)) / nrow(df) * 100 #16% recidivated\n')


# In[26]:


get_ipython().run_cell_magic('R', '', 'nrow(filter(df, two_year_recid == 1)) #652 people of 4020 recidivated\n')


# In[27]:


get_ipython().run_cell_magic('R', '-w 900 -h 363 -u px', 'library(grid)\nlibrary(gridExtra)\npblack <- ggplot(data=filter(df, race =="African-American"), aes(ordered(v_decile_score))) + \n          geom_bar() + xlab("Violent Decile Score") +\n          ylim(0, 700) + ggtitle("Black Defendant\'s Violent Decile Scores")\npwhite <- ggplot(data=filter(df, race =="Caucasian"), aes(ordered(v_decile_score))) + \n          geom_bar() + xlab("Violent Decile Score") +\n          ylim(0, 700) + ggtitle("White Defendant\'s Violent Decile Scores")\ngrid.arrange(pblack, pwhite,  ncol = 2)\n')


# In[28]:


get_ipython().run_cell_magic('R', '', 'df <- mutate(df, crime_factor = factor(c_charge_degree)) %>%\n      mutate(age_factor = as.factor(age_cat)) %>%\n      within(age_factor <- relevel(age_factor, ref = 1)) %>%\n      mutate(race_factor = factor(race,\n                                  labels = c("African-American", \n                                             "Asian",\n                                             "Caucasian", \n                                             "Hispanic", \n                                             "Native American",\n                                             "Other"))) %>%\n      within(race_factor <- relevel(race_factor, ref = 3)) %>%\n      mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%\n      within(gender_factor <- relevel(gender_factor, ref = 2)) %>%\n      mutate(score_factor = factor(v_score_text != "Low", labels = c("LowScore","HighScore")))#caution: It\'S v_score not score\nmodel <- glm(score_factor ~ gender_factor + age_factor + race_factor +\n                            priors_count + crime_factor + two_year_recid, family="binomial", data=df)\nsummary(model)\n')


# The violent score overpredicts recidivism for black defendants by 77.3% compared to white defendants.

# In[29]:


get_ipython().run_cell_magic('R', '', 'control <- exp(-2.24274) / (1 + exp(-2.24274)) #white, medium aged, male as testing group\nexp(0.65893) / (1 - control + (control * exp(0.65893))) #treatment group\n')


# Defendands under 25 are 7.4 times as likely to get a higher score as middle aged defendants.

# In[2]:


get_ipython().run_cell_magic('R', '', "exp(3.14591) / (1 - control + (control * exp(3.14591)))\n\n#here they didn't control for gender\n")


# ## Predictive Accuracy of COMPAS
# 
# In order to test whether Compas scores do an accurate job of deciding whether an offender is Low, Medium or High risk,  we ran a Cox Proportional Hazards model. Northpointe, the company that created COMPAS and markets it to Law Enforcement, also ran a Cox model in their [validation study](http://cjb.sagepub.com/content/36/1/21.abstract).
# 
# We used the counting model and removed people when they were incarcerated. Due to errors in the underlying jail data, we need to filter out 32 rows that have an end date more than the start date. Considering that there are 13,334 total rows in the data, such a small amount of errors will not affect the results.

# In[31]:


get_ipython().run_cell_magic('R', '', 'library(survival)\nlibrary(ggfortify)\n\ndata <- filter(filter(read.csv("./cox-parsed.csv"), score_text != "N/A"), end > start) %>%\n        mutate(race_factor = factor(race,\n                                  labels = c("African-American", \n                                             "Asian",\n                                             "Caucasian", \n                                             "Hispanic", \n                                             "Native American",\n                                             "Other"))) %>%\n        within(race_factor <- relevel(race_factor, ref = 3)) %>%\n        mutate(score_factor = factor(score_text)) %>%\n        within(score_factor <- relevel(score_factor, ref=2))\n\ngrp <- data[!duplicated(data$id),] #! is the logical-NOT operator in R. --> No id should be double\nnrow(grp)\n')


# In[32]:


get_ipython().run_cell_magic('R', '', 'summary(grp$score_factor)\n')


# In[33]:


get_ipython().run_cell_magic('R', '', 'summary(grp$race_factor)\n')


# In[34]:


get_ipython().run_cell_magic('R', '', 'f <- Surv(start, end, event, type="counting") ~ score_factor # \nmodel <- coxph(f, data=data) # Cox Proportional Hazards\nsummary(model)\n\n\'\'\'only for score factor high and medium, because they consider low indirectly (exp(coef) tells us how likely \nthey are to recidivate compared with the other factors --> core factore Medium is 2.2 times as likely to recidivate \nthan lows)\'\'\'\n\n#interpreting the output: https://stats.stackexchange.com/questions/83892/understanding-coxph-output-in-r\n\n\'\'\'Concordance scores: compare two seemingly same but diffrent scores / The meaning of concordance rate is defined as \n"the proportion of pairs of individuals that share a particular attribute, given that one of the individuals has that \ncharacteristic.\'\'\'\n')


# People placed in the High category are 3.5 times as likely to recidivate, and the COMPAS system's concordance 63.6%. This is lower than the accuracy quoted in the Northpoint study of 68%.

# In[35]:


get_ipython().run_cell_magic('R', '', 'decile_f <- Surv(start, end, event, type="counting") ~ decile_score\ndmodel <- coxph(decile_f, data=data)\nsummary(dmodel)\n\n#why don\'t they need decile_score as factor to deffrintiate between scores\n')


# COMPAS's decile scores are a bit more accurate at 66%.
# 
# We can test if the algorithm is behaving differently across races by including a race interaction term in the cox model.

# In[36]:


get_ipython().run_cell_magic('R', '', 'f2 <- Surv(start, end, event, type="counting") ~ race_factor + score_factor + race_factor * score_factor\nmodel <- coxph(f2, data=data)\nprint(summary(model))\n')


# The interaction term shows a similar disparity as the logistic regression above. (so the same as logistic regression showed overprediction for blacks in recidivism)
# 
# High risk white defendants are 3.61 more likely than low risk white defendants, while High risk black defendants are 2.99 more likely than low.
# --> this could mean that as they overpredicted for blacks they also had more true positives in blacks than in whites.
# --> we critize: where is medium? Ratio is overlooked (how many lows and highs do we have in the races? no absolute numbers). These numbers only help if we now the number of recidivates for low, because the total outcome depends on it (we could calculate that from the data): 
# low white: 5 --> high white: 20 | low white: 1 --> high white: 4
# low black: 1 --> high black: 3 | low black: 5 --> high black: 15

# In[37]:


import math
print("Black High Hazard: %.2f" % (math.exp(-0.18976 + 1.28350))) #race_factorAfrican-American:score_factorHigh  + score_factorHigh in the first output from cell 36
print("White High Hazard: %.2f" % (math.exp(1.28350)))
print("Black Medium Hazard: %.2f" % (math.exp(0.84286-0.17261)))
print("White Medium Hazard: %.2f" % (math.exp(0.84286)))


# In[38]:


get_ipython().run_cell_magic('R', '-w 900 -h 563 -u px', '\nfit <- survfit(f, data=data) #took the survival formula of score_factor\n\nplotty <- function(fit, title) {\n  return(autoplot(fit, conf.int=T, censor=F) + ggtitle(title) + ylim(0,1))\n}\nplotty(fit, "Overall")\n\n\'\'\'interpretation of the plot:\nafter two years about 75% of high score factore have not recidivated, about 50% of \nMedium score have not recidivated, about 27% of low score have not recidivated\'\'\'\n')


# Black defendants do recidivate at higher rates according to race specific Kaplan Meier plots.

# In[39]:


get_ipython().run_cell_magic('R', '-w 900 -h 363 -u px', 'white <- filter(data, race == "Caucasian")\nwhite_fit <- survfit(f, data=white)\n\nblack <- filter(data, race == "African-American")\nblack_fit <- survfit(f, data=black)\n\ngrid.arrange(plotty(white_fit, "White defendants"), \n             plotty(black_fit, "Black defendants"), ncol=2)\n\n\'\'\'interpretation: blacks recidivate more per high, low and medium than whites (fewer do not recidivate per score factor)\n')


# In[40]:


get_ipython().run_cell_magic('R', '', 'summary(fit, times=c(730)) #of overall\n# survival = they did not residivate\n')


# In[41]:


get_ipython().run_cell_magic('R', '', 'summary(black_fit, times=c(730))\n#2.59 e+03 = 2590\n')


# In[42]:


get_ipython().run_cell_magic('R', '', "summary(white_fit, times=c(730))\n'''\nsurvival in medium: 0.576\n--> interpretation: white who survived are in each score factor a little more than blacks (but only by 0.0xx)''\n")


# Race specific models have similar concordance values.
# '''https://study.com/learn/lesson/concordance-rate-concept-formula.html
# meaning that if we imagine two pairs of twins (one twin pair is white, one twin pair is black) and each time we have one twin medium and one twin high, the concordance gives wether the prediction for each of the twins to recidivate is better or worse compared to his / her twin
# 
# https://pfeiffertheface.com/what-does-a-50-concordance-rate-mean/
# --> so both concordance rates are higher than 0.61 which means good
# 
# "The concordance is dened P(xi > xj |yi > yj ),
# the probability that the prediction x goes in the same direction as the actual data y. A pair of
# observations i, j is considered concordant if the prediction and the data go in the same direction,
# i.e., (yi > yj , xi > xj ) or (yi < yj , xi < xj ). The concordance is the fraction of concordant pairs." (Sofi has source)
# '''

# In[43]:


get_ipython().run_cell_magic('R', '', 'summary(coxph(f, data=white))\n')


# In[44]:


get_ipython().run_cell_magic('R', '', 'summary(coxph(f, data=black))\n')


# Compas's violent recidivism score has a slightly higher overall concordance score of 65.1%.

# In[45]:


get_ipython().run_cell_magic('R', '', 'violent_data <- filter(filter(read.csv("./cox-violent-parsed.csv"), score_text != "N/A"), end > start) %>% \n# how do cox and raw data differ?\n        mutate(race_factor = factor(race,\n                                  labels = c("African-American", \n                                             "Asian",\n                                             "Caucasian", \n                                             "Hispanic", \n                                             "Native American",\n                                             "Other"))) %>%\n        within(race_factor <- relevel(race_factor, ref = 3)) %>%\n        mutate(score_factor = factor(score_text)) %>%\n        within(score_factor <- relevel(score_factor, ref=2))\n\n\nvf <- Surv(start, end, event, type="counting") ~ score_factor\nvmodel <- coxph(vf, data=violent_data)\nvgrp <- violent_data[!duplicated(violent_data$id),]\nprint(nrow(vgrp))\nsummary(vmodel)\n')


# In this case, there isn't a significant coefficient on African American's with High Scores.

# In[46]:


get_ipython().run_cell_magic('R', '', 'vf2 <- Surv(start, end, event, type="counting") ~ race_factor + race_factor * score_factor\nvmodel <- coxph(vf2, data=violent_data)\nsummary(vmodel)\n\n#for african american with high score it\'s: 0.970513 --> we can\'t analyse the model ??\n')


# In[47]:


get_ipython().run_cell_magic('R', '', 'summary(coxph(vf, data=filter(violent_data, race == "African-American")))\n')


# In[48]:


get_ipython().run_cell_magic('R', '', 'summary(coxph(vf, data=filter(violent_data, race == "Caucasian")))\n')


# In[49]:


get_ipython().run_cell_magic('R', '-w 900 -h 363 -u px', 'white <- filter(violent_data, race == "Caucasian")\nwhite_fit <- survfit(vf, data=white)\n\nblack <- filter(violent_data, race == "African-American")\nblack_fit <- survfit(vf, data=black)\n\ngrid.arrange(plotty(white_fit, "White defendants"), \n             plotty(black_fit, "Black defendants"), ncol=2)\n\n#what does the coloured area around the line mean? variation? If so, the variation is lower for blacks than for whites\n\'\'\'does it show prediction or data (reality)? vf is formula, so it should show the data and not the prediction. But\nits an estimation. How can they make an estimation, if they dont predict????\'\'\'\n')


# ## Directions of the Racial Bias
# 
# The above analysis shows that the Compas algorithm does overpredict African-American defendant's future recidivism, but we haven't yet explored the direction of the bias. We can discover fine differences in overprediction and underprediction by comparing Compas scores across racial lines.

# In[50]:


from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable #import classes and definitions from his own py
from csv import DictReader

people = [] #create a new dataframe with the lenght of cox-parsed
with open("./cox-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader) 
            if p.valid:
                people.append(p)
    except StopIteration:
        pass

pop = list(filter(lambda i: ((i.recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.score_valid, people)))) # total prisoner populatio within two years
#either they recidivated before date X or it's date X they are included ??
#https://www.w3schools.com/python/python_lambda.asp
# what does score_valid mean?
recid = list(filter(lambda i: i.recidivist == True and i.lifetime <= 730, pop)) #pop filtered by recidivists 
rset = set(recid) # makes it a diffrent data type: collection of data which is unordered, unchangeable*, and unindexed.
surv = [i for i in pop if i not in rset] #all the people from pop who aren't recidivists

#http://localhost:8888/edit/1-%20Data%20Science%20project/data/truth_tables.py
'''they duplicate the data without premutating or bootsstrapping'''
# its the sames as: def create_two_year_files():


# In[51]:


print("All defendants")
table(list(recid), list(surv)) #counting definition defined in truth_tables.py
#we assume: medium is in high


# In[52]:


print("Total pop: %i" % (2681 + 1282 + 1216 + 2035))


# In[53]:


import statistics
print("Average followup time %.2f (sd %.2f)" % (statistics.mean(map(lambda i: i.lifetime, pop)),
                                                statistics.stdev(map(lambda i: i.lifetime, pop)))) 
'''In article it sattes: Starting with the database of COMPAS scores, we built a profile of each person’s criminal history, 
both before and after they were scored. We collected public criminal records from the Broward County Clerk’s Office website
through April 1, 2016. On average, defendants in our dataset were not incarcerated for 622.87 days (sd: 329.19).'''
#= time when they were not incarenated. Is it the time between their first prison time and their recidivism?
print("Median followup time %i" % (statistics.median(map(lambda i: i.lifetime, pop))))


# Overall, the false positive rate is 32.35%.

# In[54]:


print("Black defendants")
is_afam = is_race("African-American")
table(list(filter(is_afam, recid)), list(filter(is_afam, surv)))


# That number is higher for African Americans at 44.85%.

# In[55]:


print("White defendants")
is_white = is_race("Caucasian")
table(list(filter(is_white, recid)), list(filter(is_white, surv)))


# And lower for whites at 23.45%.

# In[56]:


44.85 / 23.45 #devide fp rates of blacks by whites


# Which means under COMPAS black defendants are 91% more likely to get a higher score and not go on to commit more crimes than white defendants after two year.

# COMPAS scores misclassify white reoffenders as low risk at 70.4% more often than black reoffenders.

# In[57]:


47.72 / 27.99 #devide fn rates of blacks by whites


# In[58]:


hightable(list(filter(is_white, recid)), list(filter(is_white, surv))) #here they have medium in low


# In[59]:


hightable(list(filter(is_afam, recid)), list(filter(is_afam, surv)))


# In[4]:


'''interpretation:
fp: high and they didnt recidivate --> 
fn: low / medium and they did recidivate
'''

print('fn relation ' + str(61/80)) # blacks are less likely to be labeld low / medium and do recidivate
print('fp relation ' + str(16/5)) # blacks are 3 times more likely to be labeld as high even thought they dont recidivate

#blacks are put in high scores more often making it fn rate go down for them and fp go up


# ## Risk of Violent Recidivism
# 
# Compas also offers a score that aims to measure a persons risk of violent recidivism, which has a similar overall accuracy to the Recidivism score.

# In[60]:


vpeople = []
with open("./cox-violent-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                vpeople.append(p)
    except StopIteration:
        pass

vpop = list(filter(lambda i: ((i.violent_recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.vscore_valid, vpeople))))
vrecid = list(filter(lambda i: i.violent_recidivist == True and i.lifetime <= 730, vpeople))
vrset = set(vrecid)
vsurv = [i for i in vpop if i not in vrset]


# In[61]:


print("All defendants")
vtable(list(vrecid), list(vsurv))


# Even moreso for Black defendants.

# In[62]:


print("Black defendants")
is_afam = is_race("African-American")
vtable(list(filter(is_afam, vrecid)), list(filter(is_afam, vsurv)))


# In[63]:


print("White defendants")
is_white = is_race("Caucasian")
vtable(list(filter(is_white, vrecid)), list(filter(is_white, vsurv)))


# Black defendants are twice as likely to be false positives for a Higher violent score than white defendants.

# In[64]:


38.14 / 18.46


# White defendants are 63% more likely to get a lower score and commit another crime than Black defendants.

# In[65]:


62.62 / 38.37


# ## Gender differences in Compas scores
# 
# In terms of underlying recidivism rates, we can look at gender specific Kaplan Meier estimates. There is a striking difference between women and men.

# In[66]:


get_ipython().run_cell_magic('R', '', '\nfemale <- filter(data, sex == "Female")\nmale   <- filter(data, sex == "Male")\nmale_fit <- survfit(f, data=male)\nfemale_fit <- survfit(f, data=female)\n')


# In[67]:


get_ipython().run_cell_magic('R', '', 'summary(male_fit, times=c(730))\n')


# In[68]:


get_ipython().run_cell_magic('R', '', 'summary(female_fit, times=c(730))\n')


# In[69]:


get_ipython().run_cell_magic('R', '-w 900 -h 363 -u px', 'grid.arrange(plotty(female_fit, "Female"), plotty(male_fit, "Male"),ncol=2)\n')


# As these plots show, the Compas score treats a High risk women the same as a Medium risk man.

# In[ ]:




