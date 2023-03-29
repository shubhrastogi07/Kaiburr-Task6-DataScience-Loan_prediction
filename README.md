# Kaiburr Task-6 DataScience - Loan Prediction Machine Learing Model
## Introduction
- In this Loan Status Prediction dataset, we have the data of applicants those who previously applied for the loan based on the property which is Property Loan. 
- The bank will decide whether to give a loan for the applicant based on some factors such as Applicant Income, Loan Amount, previous Credit History, Co-applicant Income, etc.., 
- Our goal is to build a Machine Learning Model to predict the loan to be approved or to be rejected for an applicant.
- In this project, we are going to classify an individual whether he/she can get the loan amount based on his/her Income, Education, Working Experience, Loan which is taken previously, and many more factors. 
- Let’s get more into it by looking at the data.

Project Explanation
### Data Collection
- The dataset which we get from kaggle consists of two CSV(Comma Separated Values) files.
  - One is Train Data (`train.csv`)
  - Another is Test Data (`test.csv`)

**Loading the collected data**

- The CSV data is loaded with the help of [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method in pandas library.
- The Training data consists of 2001 applicant samples and 12 features.
- The 12 features are Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicanIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History and Property Area.

### Feature Engineering
There are 12 features in the training data. Let's explore the features.

***Loan_ID***

- The Loan_ID is generally is used to identify an applicant uniquely but in any way, it doesn’t decide the loan status. So we can ignore the Loan_ID column for the prediction.

***Gender***

- Gender is a **nominal** kind of **qualitative** data, because there is no numerical relation between different genders.
- For 13 applicants, Gender is not mentioned in the data.
- The Unique values are Male and Female

![image](https://user-images.githubusercontent.com/83091167/228514726-a1a3cf1e-c816-428c-a69f-06c1c36db8c9.png)

By extracting the samples having null values on Gender columns separately, I am able to get the folowing details :
  - Most of these people are married. So we can't fill Gender based on Married column.
  - Most of these people are graduated. So we can't fill Gender based on Education column.
  - Most of these people are self employed. So we can't fill Gender based on Self_Employed column.
  - Most of these people are having Credit_History and Loan_Amount_Term as 360 days(1 year).
  - Since we are going to predict applicant's Gender, we can omit CoapplicantIncome
  - So we are going predict the Gender based on Dependents, ApplicantIncome, LoanAmount, and Property_Area
  
***Married***

- Since there are only 2 kind of values are possible to be present in this feature which is married or not married. This is a **binary** kind of **qualitative** data.
- For 3 applicants, Married is not mentioned in the data.
- The two unique values present in the feature is `Yes` and `No`.

![image](https://user-images.githubusercontent.com/83091167/228515660-e517b984-1905-43d5-abfa-a7bf9f711219.png)

- By extracting the samples having null values on Married columns separately, I am able to get the folowing details :
  - Those 3 applicants are Graduated, Not Self_Employed, and having Credit History.
  - Applied for different Loan_Amount_Term - 360, 240 and 480
  - The property area is Semiurban for 2 applicants and Urban for 1 applicant.
  - Loan is approved for all 3 applicants.
  
  ***Dependents***

- The Dependents feature is a **discrete** kind of **quantitative** data.
- From my thought, dependents feature refer to the number of children of applicant.
- For 15 applicants, Dependents is not mentioned in the data.
- There are 4 unique values present in this feature. They are `0`, `1`, `2`, and `3+`.

![image](https://user-images.githubusercontent.com/83091167/228516628-47fe0eca-a12d-4d9e-8e3d-5ee93de9c027.png)

- By extracting the samples having null values on Dependents columns separately, I am able to get the folowing details :
  - Most of them are Married, Male applicants, Graduated and Not Self_Employed.
  - Since the data is in the form of string, we should convert it into integer values.
  - In this generation, 3+ children is very less. So we can convert `3+` into `3`.

![image](https://user-images.githubusercontent.com/83091167/228517852-e277885f-8258-40db-8f14-fe32fefbcab5.png)

  ***Education***

- The Education column is a **binary** kind of **qualitative** data. Because there are only two values possible in this feature. They are Graduated and Not Graduated.
- All the applicants given their Education Details
- The two binary values are `Graduate` and `Not Graduate`.

![image](https://user-images.githubusercontent.com/83091167/228516865-53339d22-2f4f-415c-967f-b9bc097354ad.png)

- Most of the applicants are graduated.
- It is a binary data, we can encode the null value with 0 for Not Graduated and 1 for Graduated

***Self_Employed***

- The Self_Employed column is a **binary** kind of **qualitative** data. Because there are only two values possible in this feature. They are Self_Employed and Not Self_Employed.
- For 32 applicants, Self_Employed status is not mentioned in the data
- The two binary values are `Yes` and `No`.

![image](https://user-images.githubusercontent.com/83091167/228517119-a6256927-5485-4729-ba3b-abf5e26d3722.png)

- Nearly 86% percentage of the applicant are not self employed.
- Since it is a binary data, we can encode the column with binary values. 1 for Self_Employed and 0 for Not Self_Employed.

***Applicant_Income***

- The Applicant Income column is a **continuous** kind of quantitative data.
- All the applicants provided their Applicant Income.

Let's see the distribution of Applicant Income

![image](https://user-images.githubusercontent.com/83091167/228518312-a80deb9f-5cdb-4029-a37a-164f4d4151d6.png)

- From the above distplot, most of the Applicants income less than Rs.10,000 and some considerable amount of applicants having income between Rs.10,000 and Rs.20,000.

***Co-applicant_Income***

- The Co-applicant Income column is a **continuous** kind of **quantitative** data.
- All the applicants provided their Co-applicant Income.

Let's see the distribution of Co-applicant Income

![image](https://user-images.githubusercontent.com/83091167/228518164-cf905c1a-2d3d-476e-8cec-5818a6508b0a.png)

From the above dist plot most of the co-applicant income is zero or nearer to zero

***Loan_Amount***

- The Co-applicant Income column is a **continuous** kind of **quantitative** data.
- For 22 applicants, the LoanAmount are not mentioned in the data.

Let's see the distribution of Loan Amount

![image](https://user-images.githubusercontent.com/83091167/228518396-5de31fe1-c720-4f4a-a662-a3d5675c17bc.png)


***Loan_Amount_Term***

- The Loan_Amount_Term column is a **discrete** kind of **quantitative** data.
- For 14 applicants, the Loan_Amount_Term is not included in the data.
- The different Loan_Amount_Terms are 12, 3, 60, 84, 120, 180, 240, 300, 360 and 480.

![image](https://user-images.githubusercontent.com/83091167/228518661-3c26db2c-f159-4649-9282-7a857ce60683.png)

- From the above plot, we can see that the Loan_Amount_Term of 360 is most frequently chosen. 
- Nearly 83 % of applicants choose to 360 Term.

***Credit_History***

- It is a **binary** kind of **qualitative** data.
- For 50 applicants, the Credit_History are not mentioned in the data.
- It consists of binary values.
  - For applicants having Credit_History - 1
  - For applicants aving Credit_History - 0


From the above plot, the point we got is
- If the applicant is having Credit_History, then there is a difficulty on classifying.
- But if the applicant is not having Credit_History, then there is a high probability chance of rejection.


***Property_Area***

- The Property_Area column is a **ordinal** kind of **qualitative** data.
- All the applicants given their Property_Area.
- The ordinal datas present in this column are Urban, Semiurban and Rural.

![image](https://user-images.githubusercontent.com/83091167/228519171-6ea0a9cb-dd62-4afd-9a25-965c091e6558.png)


<h5 align="center">Hence we finished our Feature Engineering part</h5>

<h3>Data Tranformation<h3>
 - Handling Numerical Data by taking log values of applicant income, co-applicant income, Total Income, Loan Amount, Loan Term
 
![image](https://user-images.githubusercontent.com/83091167/228521161-dbbd2d19-b513-4004-b2d6-cb2be29455d2.png)

 - Handeling Catagorical data by converting them in to dummy variables and changing it in numerical value.

![image](https://user-images.githubusercontent.com/83091167/228522195-6bec1e33-339a-4c8b-9056-5ac30ed05fc5.png)

Now as we have all the parameters in numerical we are ready for Model Training.

- Done the same Data Processing and Data Transformation for Test Dataset.
- Distributing attributes in X and Y

## Model Tranning

![image](https://user-images.githubusercontent.com/83091167/228522909-142804fe-ceea-4552-9359-be8feae243dc.png)

