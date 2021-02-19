# NLP-project

# Brainstorming
5 reasons people buy a product :
- Make or save money (Profit)
- Saves Time  (Easy to Use)
- Looks Good (Prestige)
- Feels Good (Pleasure)
- Protects what we have (Security)


## Notes from HumAIn
Our NLP challenge: extract AI usecases out of organisational reports

End-to-end
Good story telling
Creativity

Inspire --> 	Technology	Evangelisation
engage-->	Digital 	Discovery Sprints
industry -->	Data Science Consulting

# Our process
To be taken care of: [HumAIn]
Time in your presentation, business case, learnings, evaluation, documentation


## Initial proposal
Initial proposal usecases extraction.

"To build a model that is going to identify the use cases and related industry type inside a 
provided document based on an existing (internal) dataset of use cases

- time
- Varied topic labels for each text
- challenge of extracting part of text and weight similarities

## Updated proposal 
TO BE DEFINED: 

"To build a model that is going to identify related industry/industries type inside a 
provided document based on an existing (internal) dataset of use cases


### Context
Look for relevant docs like:

Using Information Extraction to Classify Newspapers Advertisements
https://www.researchgate.net/publication/37441752_Using_Information_Extraction_to_Classify_Newspapers_Advertisements

Internet Articles Classification by Industry Types Based on TF-IDF
https://link.springer.com/chapter/10.1007/978-981-10-7605-3_179
In order to understand a specific industry field, people usually look at the financial statements of the companies relevant to the industry field. 
Financial statements have diverse and numerical information but have past financial states of companies because those are usually quarterly reported. So, needs to timely obtain the current states of an industry field is increasing. Proposed method is focusing on internet articles because they are easy to obtain and updated with new information every day. As a preliminary study of extracting information on industries from internet articles, this paper proposes a method to classify internet articles by industry types. The proposed method in this paper computes importance values of nouns in internet articles based on TF-IDF. Using calculated importance values, proposed method classifies articles by industry types. Through experiments, it is proven that proposed method can achieve high accuracy in industry article classification.

## Establishing context
We used Various sources for building the Industry related tags
- 'Use search engine results for your contex' [HUMAAIN]
- News dictionary
- Topic modelling, few of them
- Cambridge definition 
- Humaain tags for each article industry

## Preprocessing 
'Having good data is 80% of the effort' preprocessing' [HUMAAIN]

## Topic Modelling
We make use of unsupervised technique using NMF to cluster news articles. 
We labelled each news text with a specific Industry tag/label, considering the top 2 probilistic values.
Now we converted our problem to Supervised learning problem.
And our Dataset was ready to be trained to build  a model.

## Multi-Label Classification
We usually encounter multiclass classification in Machine learning. Not all the classification algorithms work well multilabel classification.

## The difference between a multi-class classification and a multi-label classification?
A multi class classification is where there are multiple categories associated with the target variable but each row of data falls under single category.
Where as in multi-label classification multiple categories are associated with the same data. Simply each row may have multiple categorical values.

## Models Used

## 1.Multi-label binarizer 
We Used Multi-label binarizer to transform our target(Industry type1 and type2) into multi-label format.

## 2 .Label Powerset??
We used Sklearn Label Powerset to build the model.
 What is Label Powerset??
In this, we transform the problem into a multi-class problem with one multi-class classifier is trained on all unique label combinations found in the training data.

## 3. Saving the Model and Tokenizer
We saved the model in the pickle file which were further used for single prediction.


### Deployment 
We deployed our app through streamlit on heroku. 
You can accesss it here : https://humain-app.herokuapp.com/






