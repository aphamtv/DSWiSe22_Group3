#!/usr/bin/env python
# coding: utf-8

# # Recidivism Classification Models

# In this part, we aim to answer the question of whether recidivism can be predicted based on simple factors such as age, gender, race, and prior history of recidivism. We also compare the performance of a more complex tree-based statistical model with a simpler logistic regression model.

# In[1]:


get_ipython().run_line_magic('run', 'functions.py')


# In[2]:


# set global random seed
rand = 3
os.environ['PYTHONHASHSEED']=str(rand)
np.random.seed(rand)


# In[3]:


# load compas data
compas_df = load_compas_df()


# ### Processing Data

# Our hypothesis focuses on the predictability of recidivism based on simple factors, therefore, we only use a subset of the features available in the dataset. The selected features include 'sex', 'age', 'race', 'priors_count', and 'c_charge_degree', which will be used to predict the two-year recidivism. These features were chosen based on their simplicity and potential impact on recidivism, and represent a mix of demographic and criminal history information."

# In[4]:


df = compas_df.copy()


# In[5]:


df = df[['sex', 'age', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid']]
df.shape


# #### Data bias 
# To solve the data imbalance this issue in Race group, we merge these two groups into a single "Other" group, making the sample data more balanced and allowing us to make more accurate predictions."

# In[6]:


df.loc[df['race'].isin(['Native American', 'Asian']), 'race'] = "Other"
df['race'].value_counts()


# #### Data Imbalance

# In this step, we are checking the balance of the target variable, which is the rate of recidivism. According to the data, the rate of not recidivism is 45.51% and the rate of recidivism is 54.49%. This means that the data is imbalanced, with a higher rate of recidivism than not recidivism. Despite the imbalance, the difference in the rate is small, so we have decided to leave it as it is, without making any adjustments.

# In[7]:


not_recid = (df['two_year_recid'].sum()/df['two_year_recid'].shape[0])*100

print("Not Recidivism Rate: %.3f%%" % not_recid)
print("Recidivism Rate: %.3f%%" % (100-not_recid))


# #### Convert Categorical Data

# The categorical data is converted into numerical data with OneHotCoder. The purpose of converting these variables into numerical form is to make it easier for machine learning algorithms to work with the data. In addition, to ensure consistency and accurate scaling, the numerical data is then standardized using StandardScaler. This helps to eliminate any biases or disparities that may be present in the data, leading to more accurate results from machine learning models.

# In[8]:


categorical_features = ['race', 'sex', 'c_charge_degree']
categorical_transformer = Pipeline(steps = [
    ('onehot', OneHotEncoder(drop='if_binary'))
])

numerical_features = ['age', 'priors_count']
numerical_transformer = Pipeline(steps=[
    ('scale', StandardScaler())
])

# Pipeline for Data Preprocess
preprocessor = ColumnTransformer(transformers=[
        ('cate', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
])


# ### Model Implementation

# #### Training and Testing Data

# The dataset was divided into two parts: 70% for training and 30% for testing. In order to ensure that the proportion of positive and negative samples in both the training and testing datasets was the same as in the original dataset, the stratified sampling approach was used with the target variable 'two_year_recid' as the stratification criterion. This helps to avoid any bias in the model evaluation that could result from an imbalanced distribution of the target variable in the training and testing datasets.

# In[9]:


# define features, target, and stratify for splitting
features = df[['sex','age','race','priors_count','c_charge_degree']]
target = df['two_year_recid']
stratify = df['two_year_recid']
race = df['race'] # for fairness check later


# In[10]:


X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
        features, 
        target, 
        race,
        test_size = 0.3,
        random_state = rand
)


# #### Model Selections

# We use Logistic Regression as the baseline model due to its simplicity and ease of interpretability. It is widely used for binary classification problems and can provide a quick and efficient solution for our problem.
# 
# For the tree-based models, we use Decision Tree and Random Forest. Random Forest is an ensemble model that builds multiple decision trees and aggregates their predictions to reduce overfitting and improve accuracy. Decision Tree, on the other hand, builds a single tree from the training data, making it easier to interpret the model. These two models are popular for their performance, scalability, and ability to handle large datasets. 
# 
# Initially, the model is implemented on the training data without any tuning or adjustments to the model's parameters. This allows us to obtain a baseline performance of the model and serves as a starting point for further optimization.

# In[11]:


classifiers = {
    'Logistic Regression': {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state = rand))])},
    'Decision Tree' : {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state = rand))])},
    'Random Forest' : {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state = rand))])}
}


# #### Evaluation Metrics
# The F1 score and AUC ROC are used as the evaluation metrics for the models due to the slight imbalance in the data class. These metrics provide a comprehensive evaluation of the model's performance in terms of precision, recall, and accuracy, allowing us to address the imbalance in the data class and accurately assess the model's ability to predict recidivism. In addition, cross validation using Stratified Kfolds with 5 splits is also employed to improve the robustness of the model evaluation results.

# In[12]:


# create a df to store the cross-validation results of each model
cv_result = pd.DataFrame(columns= ['Model', 
                                   'AVG F1', 
                                   'AVG AUC', 
                                   'AVG Precision', 
                                   'AVG Recall', 
                                   'AVG Accuracy'])


# In[13]:


for model_name in classifiers.keys():
    model = classifiers[model_name]['model']
    
    # define scoring metrics
    scoring = ['f1','roc_auc','precision','recall','accuracy']
        
    # generate cross validation for with defined random state
    skf = StratifiedKFold(n_splits = 5, random_state = rand, shuffle = True)
        
    # cross validation
    scores = cross_validate(
        model, 
        X_train, 
        y_train, 
        scoring = scoring, 
        cv = skf)
    cv_result = cv_result.append({'Model': model_name, 
                                  'AVG F1': scores['test_f1'].mean(), 
                                  'AVG AUC': scores['test_roc_auc'].mean(),
                                  'AVG Precision': scores['test_precision'].mean(), 
                                   'AVG Recall': scores['test_recall'].mean(), 
                                   'AVG Accuracy': scores['test_accuracy'].mean()
                                 }, ignore_index = True
                                )


# In[14]:


# print the result, sort by the AVG F1
print("Cross Validation Result")
cv_result.sort_values(by='AVG F1', ascending=False)


# The Logistic Regression model has demonstrated the best performance in terms of F1 score and AUC ROC. As a next step, we will fine-tune the tree-based models to see if we can improve their performance and find a model that outperforms Logistic Regression.

# #### Model Tuning

# We utilized Random Search to find the best hyperparameters for each of the tree-based models. The hyperparameters are the parameters that cannot be learned from the training data and are used to control the learning process of the model. By fine-tuning these hyperparameters, we aimed to improve the performance of the tree-based models and find the best one for our problem.

# In[15]:


# define params grid for decision tree
dt_param_grid = {
    'classifier__criterion': ['gini'],
    'classifier__splitter': ['best','random'],
    'classifier__max_depth': [2,5,10,20,40,None],
    'classifier__min_samples_split': [1,3,5,7],
    'classifier__max_features': ['auto','sqrt','log2',None]}


# In[16]:


# define params grid for random forest
rf_param_grid = {
    'classifier__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 200, num = 20)],
    'classifier__max_features': ['auto', 'sqrt'],
    'classifier__max_depth': [int(x) for x in np.linspace(80, 100, num = 10)],
    'classifier__min_samples_split': [int(x) for x in np.linspace(start = 2, stop = 20, num = 5)],
    'classifier__min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 20, num = 2)],
    'classifier__bootstrap': [True, False]
}


# In[17]:


tuning_classifiers = {
    'Decision Tree' : {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state = rand))]), 'param_grid': dt_param_grid},
    'Random Forest' : {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state = rand))]), 'param_grid': rf_param_grid}
}


# In[18]:


# comment out b/c running time was a bit long, the results were added to the classifiers below

# for model_name in tuning_classifiers.keys():
#     rs = RandomizedSearchCV(
#         estimator = tuning_classifiers[model_name]['model'], 
#         param_distributions = tuning_classifiers[model_name]['param_grid'], 
#         n_iter = 100, 
#         cv = 5, 
#         scoring ='f1_micro',
#         random_state = rand)
#     rs.fit(X_train, y_train)
#     print(model_name)
#     print('best score = ' + str(rs.best_score_))
#     print('best params = ' + str(rs.best_params_))
#     print()


# In[19]:


tuned_classifiers = {
    'Logistic Regression': {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state = rand))])},
     'Decision Tree' : {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            splitter = 'best', 
            min_samples_split = 3, 
            max_features = 'log2', 
            max_depth = 5, 
            criterion = 'gini',
            random_state = rand))])},
    'Random Forest' : {'model': Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators = 178,
            min_samples_split = 20,
            min_samples_leaf = 20,
            max_features = 'auto' ,
            max_depth = 91,
            bootstrap = True,
            random_state = rand))])}
}


# ### Results

# We use the Dalex package which provides a suite of tools for interpretation and explanation of complex predictive models. The use of Dalex allows us to obtain a better understanding of the relationship between our predictors and the outcome we aim to predict. 

# In[20]:


# use Dalexdalex library
import dalex as dx


# In[21]:


exp_list = []
for model_name in tuned_classifiers.keys():
    model = tuned_classifiers[model_name]['model']
    fitted_model = model.fit(X_train, y_train)
    
    # get the predict probability and prediction from each mode
    y_test_prob = fitted_model.predict_proba(X_test)[:, 1]
    # y_test_pred = np.where(y_test_prob  > 0.5, 1, 0)
    y_test_pred = fitted_model.predict(X_test)
    tuned_classifiers[model_name]['pred_test'] = y_test_pred
    tuned_classifiers[model_name]['prob_test'] = y_test_prob
    
    # create explainer for each model
    exp = dx.Explainer(
        fitted_model,
        X_test, 
        y_test,
        label = model_name, 
        verbose = False)

    exp_list += [exp] 


# #### Model Performance

# We know that F1 score is defined as the harmonic mean of precision and recall. Also, F1 work well with imbalanced data. Here we can see that F1 for Random Forest is better than Logistic Regression. A low F1 score means that both, Precision and Recall, are low; and a high F1 score means they are both high, which is the desired ideal scenario. A medium F1 score happens when Recall is high and Precision low, or viceversa. When we look at the AUC, the Logistic Regression performs slightly better, but as our data, even after the processing of it, is still a bit imbalanced, we can rely more on the R1 score.
# 
# ‍Precision is defined as the proportion of the positive class predictions that were actually correct. Within everything that has been predicted as a positive, precision counts the percentage that is correct. In this case, it would be the proportion of people classified as someone who would recidivate that did actually commit another crime in two years.
# 
# Recall refers to the proportion of actual positive class samples that were identified by the model. Within everything that actually is positive, how many did the model succeed to find. In this case, of those who did recidivate, how many did the model find as people who would commit another crime.
# 
# As mentioned before, there is a trade-off between recall and precision. A very precise model is very “pure”: maybe it does not find all the positives, but the ones that the model does class as positive are very likely to be correct. On the contrary, a model with high recall succeeds well in finding all the positive cases in the data, even though they may also wrongly identify some negative cases as positive cases.

# In[22]:


m_performance_list = [e.model_performance() for e in exp_list]
pd.concat([mp.result[['f1','auc','accuracy','recall','precision']] for mp in m_performance_list], axis=0).sort_values(by='f1', ascending=False)


# The ROC graph shows that the decision tree has a lower performance compared to the other two methods, logistic regression and random forest, which have similar results. In terms of AUC, logistic regression has a better performance than random forest.

# In[23]:


m_performance_list[0].plot(m_performance_list[1:], geom="roc")


# #### Error Breakdown

# When we compare the scores of each model by race, we can see that in terms of Precision and Recall, random forest and logistic regression have similar scores for African-Americans and Caucasians.
# 
# However, our random forest model has a higher recall for white people (43.4% vs.35.3% from the logistic regression). This means that our model succeeds better in finding the positives (will recidivate), even though it might wronly label some actual negatives (didn't recidivate). Although this is not optimal, it mitigates the effect ProPublica complaint about: that among defendants who ultimately did not reoffend, blacks were more than twice as likely as whites to be classified as medium or high risk.
# 
# In the three models, both, precision and recall, are higher for African-Americans. This is expectable since there are more black individuals in the data set, so the model can be better trained.

# In[24]:


# to calculate the error down by group
def metrics_by_group(s):
    accuracy = metrics.accuracy_score(s.y_true, s.y_pred) * 100
    precision = metrics.precision_score(s.y_true, s.y_pred) * 100
    recall = metrics.recall_score(s.y_true, s.y_pred) * 100
    f1 = metrics.f1_score(s.y_true, s.y_pred) * 100
    roc_auc = metrics.roc_auc_score(s.y_true, s.y_prob) * 100
    tn, fp, fn, tp = metrics.confusion_matrix(s.y_true, s.y_pred).ravel()
    fnr = (fn/(tp+fn)) * 100
    fpr = (fp/(tn+fp)) * 100
    support = len(s.y_true)

    return pd.Series((support, f1, roc_auc, accuracy, precision, recall, tn, fp, fn, tp, fnr, fpr),\
                     index=['support', 'f1', 'roc-auc', 'accuracy', 'precision', 'recall', 'tn','fp','fn','tp', 'fnr', 'fpr'])

def error_breakdown_by_race(model_name):
    model = tuned_classifiers[model_name]
    
    print('Error breakdown for model', model_name, 'and group by Race')
        
    predict_df = pd.DataFrame({'race': X_test['race'].tolist(),\
                              'y_true': y_test.tolist(),
                              'y_pred': model['pred_test'],
                              'y_prob': model['prob_test']}, index=X_test.index)

    group_metrics_df = predict_df.groupby(['race']).apply(metrics_by_group)

    html = group_metrics_df.sort_values(by='f1', ascending=False).style.\
            format({'support':'{:,.0f}', 'f1':'{:.1f}%', 'roc-auc':'{:.1f}%', 'accuracy':'{:.1f}%',\
                    'precision':'{:.1f}%', 'recall':'{:.1f}%',\
                    'tn':'{:.0f}', 'fp':'{:.0f}','fn':'{:.0f}', 'tp':'{:.0f}',\
                    'fnr':'{:.1f}%', 'fpr':'{:.1f}%'})
            # highlight_max(subset=['f1','roc-auc'])
            # highlight_min(subset=['fnr','fpr'])
    return html


# In[25]:


error_breakdown_by_race('Logistic Regression')


# In[26]:


error_breakdown_by_race('Decision Tree')


# In[27]:


error_breakdown_by_race('Random Forest')


# #### Variable Importance

# Dalex uses drop-out loss to represent how much the overall accuracy of the model would decrease if a specific feature were removed. According to the chart, if the feature "priors_count" were removed, the model's information would decrease significantly. In constrast, it would have been better if the "race" feature was dropped because our models do not use the 'race' feature. 
# Nevertheless, we need to examinate other metrics before making the conclusion

# In[28]:


m_vi_list = [e.model_parts() for e in exp_list]


# In[29]:


m_vi_list[0].plot(m_vi_list[1:])


# **Partial Dependences**

# After identifying the influential variables in all three models, the next step is to compare the relationship between these variables and the predicted response across the models. To do this, we use [Partial-dependence profiles (PDPs)](https://ema.drwhy.ai/partialDependenceProfiles.html), which are graphical representations of how a model's prediction changes as the value of a single input feature changes, while all other features remain constant. These profiles provide insight into the behavior of the models and help explain why they are making certain predictions.

# The partial-dependence profiles for age and prior crimes show a general trend of increased predicted probability of recidivism with younger age and higher number of prior crimes. The relationship between the features and the target in logistic regression is modeled using a linear equation, resulting in a smooth partial-dependence profile curve. In contrast, decision trees and random forests use non-linear decision rules, leading to a more complex and non-linear relationship between the features and the target. As a result, the partial-dependence profiles for these models are not necessarily smooth and tend to be flatter. The difference also can be seen at the right edge of the age scale, which is likely due to random forest models' tendency to shrink predictions towards the average and poor extrapolation performance outside the range of values seen in the training data.

# In[30]:


m_pdp_list = [e.model_profile(type = 'partial') for e in exp_list]


# In[31]:


m_pdp_list[0].plot(m_pdp_list[1:])


# In the case of categorical variables, it is interesting to focus on race. The results show that being African-American increases the probability of recidivism for all three models, while the other races decrease it. This is related to the fact that, as a Washington Post article explains, if the recidivism rate for white and black defendants is the same within each risk category, and if black defendants have a higher overall recidivism rate, then a greater share of black defendants will be classified as high risk (will recidivate). As well, being female reduces the probability of recidivism, as does committing a misdemeanor crime.
# 
# When analyzing all categorical variables, the green bar for logistic regression is always bigger than the blue one for random forest. This is expected because, in general, random forest models shrink predictions towards the average.

# In[32]:


m_pdp_cate_list = [e.model_profile(variable_type = 'categorical') for e in exp_list]
m_pdp_cate_list[0].plot(m_pdp_cate_list[1:])


# #### Instance-level Exploration

# The previous global explorations give us a comprehensive perspective on our model by highlighting errors, determining the variables with the greatest overall impact, and exploring predictor-response relationships across all observations. Now we will look at the model's behavior at the instance-level to understand why a specific prediction was made for a particular instance. DALEX uses a technique called [Break Down]((https://ema.drwhy.ai/breakDown.html#BDMethod) to calculate localized variable importance scores. The basic idea is to calculate the contribution of variable in prediction of f(x) as changes in the expected model response given other variables. However, the values of Break-down plots may vary based on the order of the variables as it examines the contribution of each variable to the prediction while holding the previous variables constant. Thus, we also use [SHapley Additive exPlanations (SHAP)](https://ema.drwhy.ai/shapley.html#shapley) to which is based on the concept of Shapley values for local explaination.

# **First Sample**

# In[33]:


sample_1 = X_test.iloc[[3]]
sample_1.head()


# When considering a 25-year-old African-American woman with 5 prior crimes charged with a felony, the global behavior of the variables applies to the local explanation of the prediction when fixing the values of the other variables: **her age, race, number of prior crimes, and charge increase the probability of recidivism, while her sex decreases it**.
# 
# In both models, random forest and logistic regression, the variable race = African-American increases the probability of recidivism, but in the random forest, it has a greater weight (0.029 compared to 0.018 in logistic regression). The same occurs with the other variables, where they have a greater absolute contribution in the random forest than in logistic regression. This is different from the analysis conducted in the global explanation, where it was mentioned that the effect of the variable race is greater for logistic regression compared to random forest. Overall, the prediction is higher than the mean.

# In[34]:


sample_1_bd_list = [e.predict_parts(sample_1, type='break_down') for e in exp_list]
sample_1_bd_list[0].plot(sample_1_bd_list[1:])


# *The row marked “intercept” represents the overall mean value of predictions for the entire dataset. Subsequent rows display changes in the mean prediction caused by fixing the value of a particular explanatory variable. Positive changes are shown with green bars, while negative differences are shown with red bars. The final row, marked “prediction,” is the sum of the overall mean value and the changes, i.e., the predicted value of recidivism probability for an individual, indicated by the blue bar.*

# **The Shapley value** represents the average marginal contribution of a feature value across all possible combinations of the other features (i.e., all possible coalitions of features that are not being analyzed). For the sample_1 woman, in both logistic regression and random forest, all variables except gender increase the probability of recidivism, and the Shapley values have a similar behavior as seen in the Break-down plots.

# In[35]:


sample_1_shap_list = [e.predict_parts(sample_1, type='shap') for e in exp_list]
sample_1_shap_list[0].plot(sample_1_shap_list[1:])


# **Sample 2**

# In[36]:


sample_2 = X_test.iloc[[20]]
sample_2


# For a 37-years-old white man, with only 1 prior crime and charged with felony, the probability of recidivism is less than the mean. The variables that raise it are the c_charge_degree and sex. In the random forest, the race variable has the biggest decrease in the probability if we compare it to the same variable in the other two models. We connect this to the variable importance graphic presented before, where the feature "race" has some importance (0.001) in the random forest.
# 
# In this case, we can see that the contribution of each variable (except for the race) in the logistic regression is bigger in absolute terms than for the random forest, as we would expect since the random forest tends to shrink towards the mean.

# In[37]:


sample_2_bd_list = [e.predict_parts(sample_2, type='break_down') for e in exp_list]
sample_2_bd_list[0].plot(sample_2_bd_list[1:])


# When analyzing the Shapley values for the man in sample_2, the contribution of the variables in the logistic regression behaves similarly to what was shown in the break-down plot. However, in the random forest, when averaged across all possible combinations of other variables, the feature "race" increases the probability of recidivism, in contrast to the behavior shown in the break-down plot, where it reduces it.
# 
# Overall, for logistic regression and random forest, we can see that the most important variables for both local and global explanations are the number of prior crimes, age, and, to a lesser extent, gender. Regarding race, it seems to be important for the random forest, as its performance would decrease if it was removed. Additionally, when analyzing partial dependence, the effect is larger - in the global explanation - and it has a greater impact on the prediction when viewed in the context of local explanation, compared to the logistic regression.

# In[38]:


sample_2_shap_list = [e.predict_parts(sample_2, type='shap') for e in exp_list]
sample_2_shap_list[0].plot(sample_2_shap_list[1:])


# #### Model Fairness

# The term "bias" is commonly used to describe disparities in algorithmic systems that are perceived as unjust or unethical from a societal standpoint. It's important to differentiate this meaning from the statistical definition of bias. In statistics and machine learning, the term bias has a well-established meaning that has been in use for a long time.
# 
# A statistical estimator is considered biased if its average or expected value diverges from the true value it aims to estimate. Statistical bias is a crucial concept in statistics, and there is a wide range of established techniques for analyzing and correcting it. Machine learning is based on historical data, which means it reproduces the patterns and relationships it finds in that data. If different groups have different base rates, the model will inevitably have different error rates. This requires providing the computer with a sufficient number of diverse examples to uncover subtle patterns and accurately generalize from historical cases to future ones. However, the fact that machine learning is evidence-based does not guarantee accurate, reliable, or fair decisions.

# **"Fairness" Decision** <br>
# That fairness decision is based on [epsilon](https://dalex.drwhy.ai/python/api/fairness/index.html), which is a parameter that defines acceptable fairness scores. The closer epsilon is to 1 the more strict the verdict is. If the ratio of certain unprivileged and privileged subgroup is within the (epsilon, 1/epsilon) range, then there is no discrimination in this metric and for these subgroups. As default, epsilon is set to 0.8 (it matches the four-fifths 80% rule).
# 
# We get the information about bias according to the metrics TPR (True Positive Rate), ACC (Accuracy), PPV (Positive Predictive Value), FPR (False Positive Rate), and STP(Statistical parity). The metrics are derived from a confusion matrix for each unprivileged subgroup and then divided by metric values based on the privileged subgroup. There are [3 types of possible conclusions](https://medium.com/responsibleml/how-to-easily-check-if-your-ml-model-is-fair-2c173419ae4c):
# 
# * `Not fair`: your model is not fair because 2 or more metric scores exceeded acceptable limits set by epsilon.
# 
# * `Neither fair or not`: your model cannot be called fair because 1 metric score exceeded acceptable limits set by epsilon.It does not mean that your model is unfair but it cannot be automatically approved based on these metrics.
# 
# * `Fair`: your model is fair in terms of checked fairness metrics.

# To determine if the model has any bias, we will use the `fairness` module from Dalex.<br>
# * `protected`: An array-like object that contains subgroup values that represent a sensitive attribute, such as sex or nationality. The fairness metrics will be calculated for each of these subgroups and compared.
# 
# * `privileged`: A string that specifies one of the subgroups, which is suspected of having the most privilege. This string will be used as a reference point for comparison with the other subgroups.

# According to ProPublica, Caucasian individuals was determined to be underpredicted. Hence, this group was selected as the privileged group to see if our model is also privileged them.

# In[39]:


mf_list = [e.model_fairness(protected = race_test, privileged = "Caucasian") for e in exp_list]


# In[40]:


mf_list[0].plot(mf_list[1:])


# * Equal opportunity ratio = A classifier satisfies this definition if the subjects in the protected and unprotected groups have equal FNR (or equal TPR, since TPR= 1-FNR).
# 
# * Predictive parity ratio = A classifier satisfies this definition if the subjects in the protected and unprotected groups have equal Positive Predictive Value (PPV)
# 
# * Predictive equality ratio = A classifier satisfies this definition if the subjects in the protected and unprotected groups have aqual FPR.
# 
# * Accuracy equality ratio = A classifier satisfies this definition if the subject in the protected and unprotected groups have equal prediction accuracy, that is, the probability of a subject from one class to be assigned to it.
# 
# * Statistical parity ratio= A classifier satisfies this definition if the subjects in the protected and unprotected groups have equal probability of being assigned to the positive predicted class.

# According to the fairness check, the logistic regression model, decision three and random forest are not fair. This is because the ratios for TPR, FPR and STP exceed the threshold (should be between 0.8 and 1.25). We can also observe the same graphically in the previous chart Fairnes Check, there the ratios for the mentioned matris are in the red zone (over the threshold).
# 
# We analyse these metrics for African-Americans:
# 
# * TPR: all models have a higher sensitivity or recall for black people, which means that the model succeeds well in finding all the positive cases in the data, even though they may also wrongly identify some negative cases as positive cases. Again, this was one of the complaints from ProPoblica.
# 
# * FPR: this metric is also higher for black defendants, since it's related to the previous explanation: the model finds more positive cases, with the downsize that it also labels someone who didn't reoffend as "will recidivate".
# 
# * STP: this ratio shows that black defendants are assigned a positive outcome (will reoffend) more often than white defendants. This is also related to the results in TPR and FPR.

# In[41]:


for mf in mf_list:
    print(mf.label)
    mf.fairness_check()
    print("\n")


# In[42]:


mf_list[0].plot(mf_list[1:],type = 'stacked')


# In[43]:


mf_list[0].plot(mf_list[1:],type = 'radar')


# In[44]:


mf_list[0].plot(mf_list[1:],type = 'heatmap')


# #### Conclusion
# Overall, "priors_count" and "age" are important predictors for the target variable in all three models. Our analysis shows that the performance of our random forest model is only slightly better than the simple logistic model. While there are tools available to help us understand the "black box" nature of the random forest model, the logistic model is still easier to interpret and understand due to its straightforward linear equation. However, it is important to note that all three models are unfair, and this limitation highlights the need for continued work to address biases in algorithmic systems. As machine learning methods play an increasingly important role in our society, it is crucial that we remain aware of their limitations and work to improve their fairness and accuracy.
