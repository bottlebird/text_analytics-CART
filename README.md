# Predicting Airbnb review scores using text analytics
Users of Airbnb who reserved lodging can leave a review comment on the Airbnb website and rate their experience of stay. In this page, I will analyze the texts of review comments to build a predictive model that assess whether any given texts are positive (four or five stars) or negative (one through three stars).

Structure:

1. Data Exploration and Preliminary Insights
2. Texts Preprocessing
3. Sparsifying the Corpus
4. Training and Testing
5. Making Prediction
<br/><br/>


## 1. Data Exploration and Preliminary Insights
The dataset airbnb-small.csv contains review comments and ratings written for listed lodges in New York City between March 2011 and March 2018.

There are seven variables in the dataset:
- **listing id**: an integer key associated with the listing
- **id**: an integer key associated with the review
- **date**: the date of the review in the format YYYY-MM-DD
- **reviewer id**: an integer key associated with the reviewer
- **reviewer**: the first name of the reviewer
- **comments**: the text of the review
- **review scores rating**: the score that the reviewer gave the listing, with 20 corresponding to one star, and 100 corresponding to five stars

### Review rating table

```bash
###Preliminary insights
table(reviews$review_scores_rating)
reviews$nchar = nchar(reviews$comments)
aggregate(reviews$nchar, list(reviews$review_scores_rating), mean)
```
|    Rating    |   Counts   |  Avg. length (chr) |
|:--------:|:------:|:------:|
| 20  |  62  | 464.6  |
| 40  |  56  | 597.73  |
| 60  |  156  | 388.8  |
| 80  |  708  | 276.48  |
| 100  |  3191  | 289.42  |


## 2. Texts preprocessing
Texts Preprocessing according to the following steps.
1. Convert the texts to lowercase
2. Remove punctuation
3. Remove all stop words
4. Remove the word 'airbnb'
5. Stem the document

```bash
# 1. Change all the comments to lower case. 
corpus = Corpus(VectorSource(reviews$comments)) # An array/collection of documents containing texts
corpus[[1]]   # individual doc
strwrap(corpus[[1]]) #strwrap: split the sentence
strwrap(corpus[[2]])

# tm_map applies an operation to every document in the corpus. 
# In this case, the operation is to lowercase (tolower). 
corpus = tm_map(corpus, tolower)
strwrap(corpus[[1]])

# 2. Remove punctuation from the document
corpus <- tm_map(corpus, removePunctuation)
strwrap(corpus[[1]])

# 3. Remove all stop words:  
corpus = tm_map(corpus, removeWords, stopwords("english"))  # removeWords(corpus,stopwords("english"))
# stopwords("english") is a dataframe that constains a list of 
# stop words. Let us look at the first ten stop words. 
stopwords("english")[1:10]

# Checking again:  
strwrap(corpus[[1]])

# 4. Remove the particular word: 'airbnb'
# This list can be customized depending on the application context
#kwic(reviews$comments, pattern="airbnb")
#reviews[grepl("word",reviews$comments),"comments"] #look for comments containing particular word
strwrap(corpus[[162]])
corpus = tm_map(corpus, removeWords, c("airbnb"))
strwrap(corpus[[162]])

# 5. Now stem documents. This corresponds to removing the parts of words
# that are in some sense not necessary (e.g. 'ing' and 'ed'). 
corpus = tm_map(corpus, stemDocument)
```



