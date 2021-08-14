## **Movie rating by using machine learning algorithms**
![social](https://img.shields.io/github/followers/VijayBarige?style=social) ![twitter](https://img.shields.io/twitter/follow/VijayBarige?style=social) 


## Description
Movies have been a primary source of entertainment for everyone, most of the people go to the theatres regularly or binge watch them from their personal computers. People spend considerble amount of time searching for a good movie to watch. It is difficult to choose one movie from hundreds of options. But there is huge amount of data in the form of reviews, comments, blogs etc., which help people to choose from. This project is based on recommender systems using machine learning algorithms mainly Multinomial Logistic regression, K Nearest Neighbours and Decision Tree Classifier. With the help of this project  users can search their desired genre movies and pick one with highest rating available. We have a dataset cantaining ratings, genres and titles of different movies, applying the said aalgorithms on this data gives the prediction of best movies in the given genre.

**For further detailed information about this project's repository, refer to the following table.**
**Feel free to reach out to me if you have any suggestions or additions you want to add to the project.** ![twittwer](https://img.shields.io/twitter/follow/VijayBarige?style=social) 

## Table of contents
  1. [About this Document](#About-this-document)
  2. [Repository Structure](#Repository-Structure)
  3. [Installation and Implementation](#Installation-and-Implementation)
  4. [Libraries imported](#Libraries-imported)
  5. [Database](#Database)
  6. [Data cleaning and Preparing](#Data-cleaning-and-Preparing)
  7. 

## About this doocument
This document contains information about my project on recommender systems for movie suggestions. Understanding and implementation of the project. Visual Representation of results and comparision between algorithms.

## Repository Structure
```cpp
-README.md -->> The file you are seeing is called Readme file, it contains all the basic information about the project code, implwmwntation, results and explanation.
-movie rating commented.ipynb -->> It contains the commented code of the discussed project.
-mrating.csv -->> It is a csv dataset file containing all the data that is being worked on with in the project.
 ```
## Installation and Implementation
We can implement this project on Jupyter online notebook which is available in any browser on the internet.
Open this [link](https://jupyter.org/try) and follow the below specified path to get implement the project.

Jupyter online notebook --> Try jupyter lab --> Upload files --> Run files

## Libraries imported

These libraries are used to downlaod neccessary tools and packages that are needed for this project.
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
```
## Database

Datab is taken from kaggle.com and merged together to form a meaningful dataset as the algorithms used are supervised algorithms, they need the data to be structured and complete to perform well. The data is in the form of a .csv file (comma separated values). They are loaded into the model and displayed by implementing the following piece of code.
```python
df=pd.read_csv('mrating.csv')
df
```
The dataset contains following fields of data.

<img src="Images/Database.png" width= 600>


## Data cleaning and Preparing
The data we get from internet is incomplete, it is neccessary to clean the data by removing noise like empty cells and filling with meaningful values.
I used label encoder to convert charatcers\strings into numerical form by assigning a unique label to each unique string. Label encoding is neccessary as machine learining algorithms cannot understand characters or strings but numbers/numericals.
![label encoder](Images/Label%20encoder.png)

**Assigned lables for title and genre column of the dataset.**

![Genre labels](Images/genres_genres_labels.png)
![Title Labels](Images/title_title_Labels.png)

Genome score column of the dataset represent the rating of each movie. Based on the genome score, the movies are assigned to one of the categories of __poor, average__ and __good__ and it is labelled as outcome column.

![outcomes](Images/outcome.png)




