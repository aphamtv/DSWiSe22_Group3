# Further Discussion
## Question 2
### 2.1 Are interpretable models just as accurate as black box machine learning models for predicting recidivism?
Interpretable models are machine learning models that help provide understandable explanations of their predictions and decision-making processes. The term “Black box machine learning models” refers to models that are not interpretable by humans because they lack transparency. Examples for black box machine learning models include neural networks. The opposite of black box machine learning models are interpretable models, sometimes also referred to as “White box models” or “Glass box models”. They are understandable and verifiable by humans, thus they are transparent. An example for an interpretable model is a logistic regression (compare Molnar 2022). 

This opacity terminology not only describes missing transparency in the algorithms themselves, but voluntary missing transparency, which is caused by the companies who develop them. Even interpretable models can be black box models, when how they work isn’t revealed by the companies - as is the case with the COMPAS algorithm that will be discussed in the next paragraph.  These algorithms are proprietary. The company’s financial benefits from selling the model would be lost, if how they work would be public as Cynthia Rudin, a scientist who advocates for interpretable models in high stake decisions, points out (2019, p.9).


Rudin argues that interpretable models can be just as accurate as black box machine learning models for predicting recidivism. In the case of the COMPAS algorithm, the interpretable alternative is COREL. COREL is a model that only uses age, priors and optionally gender. Both COMPAS and COREL “have similar true and false positive and true and false negative rates on data from Broward County, Florida” (Rudin 2019, p. 6). 

For machine learning models in general Rudin even goes as far as to state that interpretable models could even have in-practice a higher accuracy, because typographical errors can occur in the datasets and can then have an impact on the predicted outcome (2019, p.5).  Nevertheless there is a point in the fact that interpretable models are just as accurate as “the machine learning researcher’s ability to create accurate-yet-interpretable models” (2019, p. 8) allows the model to be. So definitely interpretable models aren’t automatically as accurate as black box machine learning models.

Before moving on to the discussion of transparency, it should be briefly mentioned that even if there is not necessarily a tradeoff between accuracy and interpretability, there is one for interpretability and flexibility. As Gareth et al. put it “In general, as the flexibility of a method increases, its interpretability decreases.” (2013, p. 25)

### 2.2 Can complex models be as transparent as simple ones?
That being said, be cautious that “interpretable machine learning” and “explainable machine learning” are often used interchangeably, but as Rudin argues, should be differentiated and used consciously on a common ground (2019, p.14). She defines interpretable models as models that are inherently interpretable. Whereas explainable models, in her perception, are black box models that are explained using a second (posthoc) model. Which then in turn is also the reason why troubleshooting the model is also very difficult and time-consuming. 

One of her main arguments for interpretable models, in comparison with explainable models, is “explanations are often not reliable, and can be misleading [...]. If we instead use models that are inherently interpretable, they provide their own explanations, which are faithful to what the model actually computes.” (2019, p.1) This means, the explanation model is always just an approximation of how the black box model functions. 

Zachary Lipton doesn’t make this differentiation, between interpretability and explainability, but agrees that “the task of interpretation appears underspecified” and also that “transparency to humans and post-hoc explanations [are] competing notions” (2017, p.1).
In trying to specify interpretability he considers different levels of transparency, this being the opposite of opacity:
“transparency at the level of the entire model (simulatability), 
at the level of individual components (e.g. parameters) (decomposability), 
and at the level of the training algorithm (algorithmic transparency).” (2017, p.4)
The derived levels of interpretability are therefore: 
Algorithm Transparency: “How does the algorithm create the model?” (Molnar 2022),
Global, Holistic Model Interpretability: “How does the trained model make predictions?” (Molnar 2022),
Global Model Interpretability on a Modular Level: “How do parts of the model affect predictions?” (Molnar 2022),
Local Interpretability for a Single Prediction: “Why did the model make a certain prediction for an instance?” (Molnar 2022),
Local Interpretability for a Group of Predictions: “Why did the model make specific predictions for a group of instances?” (Molnar 2022).

Lipton concludes that “. Linear models are not strictly more interpretable than deep neural networks”, but equal, less or more interpretable depending on the level of interpretability that one is concerned about (2017, p. 7).

Rudin agrees indirectly when stating that even with her differentiation or explainable and interpretable models “there is a spectrum between fully transparent models (where we understand how all the variables are jointly related to each other) and models that are lightly constrained in model form (such as models that are forced to increase as one of the variables increases, or models that, all else being equal, prefer variables that domain experts have identified as important)” (2019, p. 1). 

### 2.3 How do I communicate nonlinear dependencies?

Non-linear dependencies, also referred to as non-linear relationships, play an important role in the interpretability of models. Non-linear dependencies describe relationships between the predicting features and the predicted response of the model and are inherent to the features and predicted response, so they do occur for both black box models as well as interpretable models (Gareth et al. 2013, p.90ff). 
The ability of a model to adjust to these non-linear dependencies depends on the model's flexibility. It was already briefly mentioned that there is a tradeoff between flexibility and interpretability. Meaning that complex models, including black box machine learning models which are not interpretable by humans, can adapt better to these dependencies.

Consequently it is important to choose the right model. The model in turn can help communicate the non-linear dependencies:
For example, a decision tree can show the specific conditions that lead to a certain prediction, and how those conditions are interconnected. A linear regression model with non-linear dependencies could be better expressed by a polynomial regression function as follows: 
yi = β0 + β1xi1 + β2xi2 + ··· + βpxip + i
Another way to communicate non-linear dependencies in linear regression - but also for classification problems - are generalized additive models (GAM). Generalized additive models model each feature with a non-linear function. Then these functions are weighed and summed. Apart from these three there are even more models, including step functions and regression splines, etc. (compare Gareth et al. 2013, p.265ff; Molnar 2022).

Apart from choosing the right model the non-linear dependencies can also be analyzed directly by performing a feature importance analysis. It can show the contribution of the features involved. This in turn can help for feature tuning and minimizing the number of features for interpretable models. A feature importance analysis can be performed using permutation. Molnar explains “A feature is “important” if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is “unimportant” if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction.” (2022)

## Question 5. How can I find relevant interactions?
There are several ways to statistically find relevant interactions between features in an interpretable machine learning model. Some common methods include:

**Using interaction terms**: One way to find interactions between features is by creating interaction terms. Because the “magnitude of the interaction effect in nonlinear models does not equal the marginal effect of the interaction term, the magnitude of the interaction of the variable effect then depends on all the covariates in the model” (Ai and Norton, p. 154)., which are the product of two or more predictor variables. These interaction terms can be added to the model as new predictors, and their coefficients can be used to evaluate the strength and significance of the interactions.

**Using partial dependence plots**: Another way to find interactions between features is by using partial dependence plots. Jerome H. Friedman (p. 1219) explained that “Functions of a categorical variable and another variable (real or categorical) are best summarized by a sequence of(''trellis”) plots, each one showing the dependence of first input on the second variable, conditioned on the respective values of the first variable”. Essentially, the plots show the relationship between a predictor and the outcome variable, holding all other predictors constant.

**Using Permutation Importance**: Feature importance provides a highly compressed, global insight into the model’s behavior (Christoph Molnar). By using Permutation Importance as a way to understand the feature importance, shuffling the values of a feature and observing the change in the performance of the model. The features that have a larger effect on the model performance when shuffled are considered more important. 

**Using SHapley Additive exPlanations (SHAP) values**: Based on the Shapley documentation, SHAP values are a unified measure of feature importance that assigns each feature an importance value for a particular prediction. Originally a game theoretic approach to explain the output of any machine learning model, It connects optimal credit allocation with local explanations using classic Shapley values. By comparing the SHAP values for different features, you can identify interactions between features.

It's important to note that none of these methods are foolproof in finding interactions. The best way to find interactions is through experimentation and testing different combinations of features and also visualizing the data, will give a better understanding of the relationship between features. 
We know from several sources that in the data from Broward County, Florida, age and priors interact with race. But in order to know to which extent they interact, we would have to conduct one or several of the above mentioned methods.

