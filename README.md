# Readmission-Prediction
In this notebook, I used simple machine learning techniques to analyze healthcare data in interesting and meaningful ways and build predictive models with 94% accuracy.

### Why hospital readmissions matter?
A hospital readmission is when a patient who is discharged from the hospital, gets re-admitted again within a certain period of time. Hospital readmission rates for certain conditions are now considered an indicator of hospital quality, and also affect the cost of care adversely. 

For this reason, Centers for Medicare & Medicaid Services established the [Hospital Readmissions Reduction Program](https://www.cms.gov/medicare/medicare-fee-for-service-payment/acuteinpatientpps/readmissions-reduction-program.html) which aims to improve quality of care for patients and reduce healthcare spending by applying payment penalties to hospitals that have more than expected readmission rates for certain conditions. Although diabetes is not yet included in the penalty measures, in 2011, American hospitals spent over [$41 billion on diabetic](https://www.hcup-us.ahrq.gov/reports/statbriefs/sb172-Conditions-Readmissions-Payer.jsp) patients who got readmitted within 30 days of discharge. Being able to determine factors that lead to higher readmission in such patients, and correspondingly being able to predict which patients will get readmitted can help hospitals save millions of dollars while improving quality of care. So, with that background in mind, we used a medical claims dataset (description below), to answer these questions:

* What factors are the strongest predictors of hospital readmission in diabetic patients?
* How well can we predict hospital readmission in this dataset with limited features?

### Choosing a dataset
Finding a good dataset is one of the first challenges (besides defining a meaningful question), when trying out machine learning methods. The current state of the healthcare world is such that we can easily find datasets that rich (full of useful information) but messy (unstructured content or messy schemas) or datasets that are very clean but otherwise sterile in terms of information contained.

With this limitation, I picked a publicly available dataset from [UCI repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008#) containing de-identified diabetes patient encounter data for 130 US hospitals (1999–2008) containing 101,766 observations over 10 years. The dataset has over 50 features including patient characteristics, conditions, tests and 23 medications. Only diabetic encounters are included (i.e. at least one of three primary diagnosis was diabetes).

### Dealing with missing values
First we have to see how many missing values are (which were coded as “?” for most variables in the data):

`for col in df.columns:`<br>
&emsp;`if df[col].dtype == object:`<br>
&emsp;&emsp;`print(col,df[col][df[col] == '?'].count())`<br>
`print('gender', df['gender'][df['gender'] == 'Unknown/Invalid'].count())`

This gives us a long list but the following variables had missing values:

race 2273
>weight 98569
>payer_code 40256
>medical_specialty 49949
>diag_1 21
>diag_2 358
>diag_3 1423
gender 3
