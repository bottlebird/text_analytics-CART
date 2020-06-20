# install.packages("tm")
# install.packages("SnowballC")
library(tidyverse)
library(tm) #to create corpus and modify text therein.
library(SnowballC) #for "stemming." 
library(rpart) #to construct a CART model.
library(rpart.plot) #to plot CART tree.
library(caret)

reviews = read.csv("data/airbnb-small.csv", stringsAsFactors = FALSE)

###a) Preliminary insights
table(reviews$review_scores_rating)
reviews$nchar = nchar(reviews$comments)
aggregate(reviews$nchar, list(reviews$review_scores_rating), mean)

### b) Corpus
# Processing the text in the corpus.
# Text processing steps  
# 1. Convert the txt to lowercase
# 2. Remove punctuation
# 3. Remove all stop words
# 4. Remove the word 'airbnb'
# 5. Stem the document

# 1. Change all the comments to lower case. 
corpus = Corpus(VectorSource(reviews$comments)) # An array/collection of documents containing texts
corpus[[1]]   # individual doc
strwrap(corpus[[1]]) #strwrap: split a sentence
strwrap(corpus[[2]])

# The function tm_map applies an operation to every document in the corpus. 
# In this case, the operation is 'tolower" (i.e. to lowercase). 
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

# We have: 
strwrap(corpus[[1]])

### c) Sparsify
# "Sparsify" the corpus and remove infrequent words. 
# First, calculate the frequency of each words over all tweets. 
frequencies = DocumentTermMatrix(corpus)
frequencies   # documents as the rows, terms as the columns

# Words that appear at least 900 times: 
findFreqTerms(frequencies, lowfreq=900)

# Only keep terms that appear in at least 1% of the tweets and
# create a list of these words as follows.
sparse = removeSparseTerms(frequencies, 0.99)  # 0.99: maximal allowed sparsity 
sparse # We now have 404 terms instead of 7,040


### d) Training and Test Split
# Create a new data frame. 
# Each column corresponds to each word (term), 
# Each row corresponds to each document (review)
document_terms = as.data.frame(as.matrix(sparse))
str(document_terms)
# Create a new column for the dependent variable: 
document_terms$positive_review = reviews$review_scores_rating >= 80
head(reviews)

# Split training and test set to prepare for the modeling
split1 = (reviews$date < "2018-01-01")
split2 = (reviews$date >= "2018-01-01")
train = document_terms[split1,]
test = document_terms[split2,]


### e) Prediction
# Construct and plot the CART model.
cart = rpart(positive_review ~ ., data=train, method="class")
prp(cart)

# Assess the out-of-sample performance of the CART model
predictions.cart <- predict(cart, newdata=test, type="class")
matrix.cart = table(test$positive_review, predictions.cart) # confusion matrix
accuracy.cart = (matrix.cart[1,1]+matrix.cart[2,2])/nrow(test)
# True Positive Rate (TPR) and False Positive Rate (FPR)
TPR.cart = (matrix.cart[2,2])/sum(matrix.cart[2,])
FPR.cart = (matrix.cart[1,2])/sum(matrix.cart[1,])


##### Baseline: Baseline model where all reviews are classified as positive
accuracy.baseline = sum(test$positive_review)/nrow(test)
TPR.baseline = 1
FPR.baseline = 1

#Summary of performance
summary.performance <- data.frame (
  accuracy=round(c(accuracy.baseline,accuracy.cart),3),
  TPR=round(c(TPR.baseline,TPR.cart),3),
  FPR=round(c(FPR.baseline,FPR.cart),3))
summary.performance
