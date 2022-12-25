#!/usr/bin/env python
# coding: utf-8

# In[1]:


# filter dplyr warnings
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
import warnings
warnings.filterwarnings('ignore')


# <h2 id="cox">Cox with interaction term</h2>

# In[2]:


get_ipython().run_cell_magic('R', '', 'library(survival)\nlibrary(ggfortify)\nlibrary(dplyr)\n\ndata <- filter(filter(read.csv("./cox-parsed.csv"), score_text != "N/A"), end > start) %>%\n        mutate(race_factor = factor(race,\n                                  labels = c("African-American", \n                                             "Asian",\n                                             "Caucasian", \n                                             "Hispanic", \n                                             "Native American",\n                                             "Other"))) %>%\n        within(race_factor <- relevel(race_factor, ref = 3))\n\ngrp <- data[!duplicated(data$id),]\nnrow(grp)\nf <- Surv(start, end, event, type="counting") ~ race_factor * decile_score + age_factor + gender_factor +\n                                                 priors_count + crime_factor\nfelonies <- c(\'(F1)\',\'(F2)\', \'(F3)\',\'(F5)\',\'(F6)\',\'(F7)\')\ndata_filtered <-  mutate(data, crime_factor = factor(ifelse(c_charge_degree %in% felonies, \'F\', \'M\'))) %>%\n                  mutate(age_factor = as.factor(age_cat)) %>%\n                  within(age_factor <- relevel(age_factor, ref = 1)) %>%\n                  mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%\n                  within(gender_factor <- relevel(gender_factor, ref = 2)) %>%\n                  filter(days_b_screening_arrest <= 30) %>%\n                  filter(days_b_screening_arrest >= -30) %>%\n                  filter(is_recid != -1)\ngrp <- data_filtered[!duplicated(data_filtered$id),]\nprint(nrow(grp))\nmodel <- coxph(f, data=data_filtered)\nprint(summary(model))\n')


# In[3]:


get_ipython().run_cell_magic('R', '', 'd <- c(1:10)\ndata.frame(decile_score=d, black_vs_white_hazard_ratio=exp(0.244810 - 0.042704 * d))\n')


# <h2 id="logistic">Logistic regression without two year recidivism term</h2>

# In[4]:


get_ipython().run_cell_magic('R', '', 'raw_data <- read.csv("./compas-scores-two-years.csv")\n\ndf <- dplyr::select(raw_data, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, \n                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% \n        filter(days_b_screening_arrest <= 30) %>%\n        filter(days_b_screening_arrest >= -30) %>%\n        filter(is_recid != -1) %>%\n        filter(c_charge_degree != "O") %>%\n        filter(score_text != \'N/A\')\n\ndf <- mutate(df, crime_factor = factor(c_charge_degree)) %>%\n      mutate(age_factor = as.factor(age_cat)) %>%\n      within(age_factor <- relevel(age_factor, ref = 1)) %>%\n      mutate(race_factor = factor(race)) %>%\n      within(race_factor <- relevel(race_factor, ref = 3)) %>%\n      mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%\n      within(gender_factor <- relevel(gender_factor, ref = 2)) %>%\n      mutate(score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))\nmodel <- glm(score_factor ~ gender_factor + age_factor + race_factor +\n                            priors_count + crime_factor, family="binomial", data=df)\nsummary(model)\n')


# In[5]:


get_ipython().run_cell_magic('R', '', 'control <- exp(-1.26216) / (1 + exp(-1.26216))\nexp(0.47752) / (1 - control + (control * exp(0.47752)))\n')


# In[6]:


get_ipython().run_cell_magic('R', '', 'library(car)\nvif(model <- glm(score_factor ~ gender_factor + age_factor + race_factor +\n                            priors_count + crime_factor + two_year_recid, family="binomial", data=df))\n')


# In[7]:


get_ipython().run_cell_magic('R', '-w 900 -h 900 -u px', 'library(pROC)\nraw_data <- read.csv("./compas-scores-two-years.csv")\nraw_data <- mutate(raw_data, score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))\nw <- filter(raw_data, race=="Caucasian")\nwroc <- roc(two_year_recid ~ decile_score, data=w)\nb <- filter(raw_data, race=="African-American")\nbroc <- roc(two_year_recid ~ decile_score, data=b)\n')


# <h2 id="roc-curves">Unsmoothed ROC Curves</h2>

# In[8]:


get_ipython().run_cell_magic('R', ' -w 900 -h 900 -u px', 'library(gridExtra)\nw <- filter(raw_data, race=="Caucasian")\nwroc <- roc(two_year_recid ~ decile_score, data=w)\nb <- filter(raw_data, race=="African-American")\nbroc <- roc(two_year_recid ~ decile_score, data=b)\nplot(wroc)\nplot(broc, add=TRUE, col="red")\n')


# In[9]:


get_ipython().run_cell_magic('R', '-w 900 -h 900 -u px', 'wroc <- roc(two_year_recid ~ decile_score, data=w, smooth=TRUE)\nbroc <- roc(two_year_recid ~ decile_score, data=b, smooth=TRUE)\nplot(wroc)\nplot(broc, col="red", add=TRUE)\n')


# In[ ]:




