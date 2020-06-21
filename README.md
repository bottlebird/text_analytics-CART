# Predicting Airbnb review scores using text analytics
The users of Airbnb who reserve lodging can leave a review comment on the Airbnb website and rate their stay. This page analyzes the texts of reviews to build a model that assesses whether any given texts are positive (four or five stars) or negative (one through three stars).

### Structure:

1. Data Exploration
2. Texts Preprocessing
3. Sparsifying the Corpus
4. Training and Testing
5. Making Prediction
<br/><br/>


## 1. Data Exploration
The dataset airbnb-small.csv contains review comments and ratings written for listed lodges in New York City between March 2011 and March 2018.

There are seven variables in the dataset:
- **listing id**: an integer key associated with the listing
- **id**: an integer key associated with the review
- **date**: the date of the review in the format YYYY-MM-DD
- **reviewer id**: an integer key associated with the reviewer
- **reviewer**: the first name of the reviewer
- **comments**: the text of the review
- **review scores rating**: the score that the reviewer gave the listing, with 20 corresponding to one star, and 100 corresponding to five stars

#### Review rating table

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
The texts are preprocessed based on the following steps:
1. Convert to lowercase
2. Remove punctuation
3. Remove all stop words
4. Remove the word 'airbnb'
5. Stem the document


#### I. Convert to lowercase
This process converts all the comments to lower case. tm_map applies an operation to every document in the corpus. In this case, the operation is to lowercase (tolower).

```bash
corpus = Corpus(VectorSource(reviews$comments)) # An array/collection of documents containing texts
corpus[[1]]   # individual doc
strwrap(corpus[[1]]) #strwrap: split the sentence

corpus = tm_map(corpus, tolower)
strwrap(corpus[[1]])
```

||Original|
|:--------|:--------|
|[1]| "Good stay, a few issues with the direction instructions as they were different from where"  |
|[2]|  "the actual property was."  |

||Converted|
|:--------|:--------|
|[1]| "good stay, a few issues with the direction instructions as they were different from where" |
|[2]|  "the actual property was."  |

#### II. Remove punctuation
This process removes all the puctuations from the document.

```bash
corpus <- tm_map(corpus, removePunctuation)
strwrap(corpus[[1]])
```

||Original|
|:--------|:--------|
|[1]| "good stay, a few issues with the direction instructions as they were different from where" |
|[2]|  "the actual property was."  |

||Converted|
|:--------|:--------|
|[1]| "good stay a few issues with the direction instructions as they were different from where"  |
|[2]|  "the actual property was"   |

#### III. Remove all stop words
This process removes all the stop words, such as 'my', 'me', 'myself', etc. 'stopwords("english")' is a dataframe that contains a list of stop words. 
```bash
corpus = tm_map(corpus, removeWords, stopwords("english"))
# stop words. Let us look at the first ten stop words. 
stopwords("english")[1:10]
# Checking again:  
strwrap(corpus[[1]])
```
|First ten stop words from 'stopwords("english")'|
|:--------|
|"i"         "me"        "my"        "myself"    "we"        "our"       "ours"      "ourselves"     "you"       "your" |


|Original|
|:--------|
|[1] "good stay a few issues with the direction instructions as they were different from where"  |
|[2]  "the actual property was"   |

|Converted|
|:--------|
|[1] "good stay issues direction instructions different actual property"|



```bash
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



