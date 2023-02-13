# Are the COMPAS risk scales racially biased against blacks?
The topic of fairness in machine learning has garnered much attention in recent years, particularly in the context of criminal justice. One such case is the use of the COMPAS system by Northpointe, which uses machine learning algorithms to predict recidivism in defendants. The results of the COMPAS system have been the subject of numerous claims, both in support and against its fairness.

In this notebook, we aim to examine the claims surrounding the COMPAS system's fairness in predicting recidivism. We use recidivism data from ProPublica to reproduce the claims made by ProPublica, Northpointe, and the Washington Post. In the second part of the paper, we address some important questions about fairness in machine learning models. Finally, we aim to answer the question of whether recidivism can be predicted based on simple factors such as age, gender, race, and prior history of recidivism. We also compare the performance of a more complex tree-based statistical model with a simpler logistic regression model.

```{tableofcontents}
```
