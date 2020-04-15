# Select the row indices of a random training set of "prop" proportion of
# the ratings that guarantees each rater appears at least once and each item
# appears at least once.

# install.packages("tm")
# install.packages("SnowballC")
library(tidyverse)
library(tm) # Will use to create corpus and modify text therein.
library(SnowballC) # Will use for "stemming." 
library(rpart) # Will use to construct a CART model.
library(rpart.plot) # Will use to plot CART tree.
library(caret)

reviews = read.csv("data/airbnb-small.csv", stringsAsFactors = FALSE)
#a) Preliminary insights
table(reviews$review_scores_rating)
reviews$nchar = nchar(reviews$comments)
aggregate(reviews$nchar, list(reviews$review_scores_rating), mean)

# b) Corpus
# Processing the text in the corpus.
# Text processing steps  
# 1. Convert the txt to lowercase
# 2. Remove punctuation
# 3. Remove all stop words
# 4. Remove the word 'airbnb'
# 5. Stem the document

# 1. Let's change all the comments to lower case. 
corpus = Corpus(VectorSource(reviews$comments)) # An array of document
corpus[[1]]   # individual doc
strwrap(corpus[[1]])
strwrap(corpus[[3]])

# The function tm_map applies an operation to every document in the corpus. 
# In this case, the operation is 'tolower" (i.e. to lowercase). 
corpus = tm_map(corpus, tolower)
strwrap(corpus[[1]])

# 2. remove punctuation from the document
corpus <- tm_map(corpus, removePunctuation)
strwrap(corpus[[1]])

# 3. remove all stop words:  
corpus = tm_map(corpus, removeWords, stopwords("english"))  # removeWords(corpus,stopwords("english"))
# stopwords("english") is a dataframe that constains a list of 
# stop words. Let us look at the first ten stop words. 
stopwords("english")[1:10]

# Checking again:  
strwrap(corpus[[1]])

# Next, we remove the particular word: 'airbnb'
# This list can be customized depending on the application context
strwrap(corpus[[7]])
corpus = tm_map(corpus, removeWords, c("airbnb"))
strwrap(corpus[[1]])

# 3. Now we stem our documents. Recall that this corresponds toremoving the parts of words
# that are in some sense not necessary (e.g. 'ing' and 'ed'). 
corpus = tm_map(corpus, stemDocument)

# We have: 
strwrap(corpus[[1]])

# 4. Let us "sparsify" the corpus and remove infrequent words. 
# First, we calculate the frequency of each words over all tweets. 
frequencies = DocumentTermMatrix(corpus)
frequencies                              # documents as the rows, terms as the columns
# Let us get a feel for what words occur the most. Words that appear at least 200 times: 
findFreqTerms(frequencies, lowfreq=200)
# Words that appear at least 100 times: 
findFreqTerms(frequencies, lowfreq=100)
# Let us only keep terms that appear in at least 1% of the tweets. We create a list of these words as follows. 
sparse = removeSparseTerms(frequencies, 0.99)  # 0.99: maximal allowed sparsity 
sparse # We now have 172 terms instead of 12,093

# 5. We first create a new data frame. Each variable corresponds to one of the 172 words, and each row corresponds to one of the tweets.
document_terms = as.data.frame(as.matrix(sparse))
str(document_terms)
# Lastly, we create a new column for the dependent variable: 
document_terms$TrumpWrote = tweets$TrumpWrote

# We have processed our data! Let us briefly construct a CART model. 
head(tweets)
# Training and test set.
split1 = (tweets$created_at < "2016-06-01")
split2 = (tweets$created_at >= "2016-06-01")
train = document_terms[split1,]
test = document_terms[split2,]

# Constructing the logistic regression model
logreg = glm(TrumpWrote ~., data=train, family="binomial")
summary(logreg)

# Assessing the out-of-sample performance of the logistic regression model
predictions.logreg <- predict(logreg, newdata=test, type="response")
matrix.logreg = table(test$TrumpWrote, predictions.logreg > 0.5)   # threshold = 0.5
matrix.logreg    # confusion matrix
accuracy.logreg = (matrix.logreg[1,1]+matrix.logreg[2,2])/nrow(test)
TPR.logreg = (matrix.logreg[2,2])/sum(matrix.logreg[2,])
FPR.logreg = (matrix.logreg[1,2])/sum(matrix.logreg[1,])

# Constructing and plotting the CART model.
cart = rpart(TrumpWrote ~ ., data=train, method="class", cp = .003)  # classification
prp(cart)

# Assessing the out-of-sample performance of the CART model
predictions.cart <- predict(cart, newdata=test, type="class")
matrix.cart = table(test$TrumpWrote, predictions.cart) # confusion matrix
accuracy.cart = (matrix.cart[1,1]+matrix.cart[2,2])/nrow(test)
TPR.cart = (matrix.cart[2,2])/sum(matrix.cart[2,])
FPR.cart = (matrix.cart[1,2])/sum(matrix.cart[1,])










cf.training.set <- function(rater, item, prop) {
  # Draw a  sample from a passed vector, including a length-1 vector
  # (see, eg, http://stackoverflow.com/a/13990144/3093387)
  set.seed(4)
  resample <- function(x, ...) x[sample.int(length(x), ...)]
  
  # Select one rating from each rater
  rater.samples <- sapply(split(seq_along(rater), rater), resample, size=1)
  
  # Select one rating for each item
  item.samples <- sapply(split(seq_along(item), item), resample, size=1)
  
  # Determine the samples currently drawn and not drawn for the training set
  selected <- unique(c(rater.samples, item.samples))
  unselected <- seq_along(rater)[-selected]
  
  # Draw the remaining elements for the training set randomly, and return
  num.needed <- max(0, round(prop*length(rater)) - length(selected))
  return(c(selected, resample(unselected, size=num.needed)))
}

# Select the desired rank for the model
# dat: data frame of ratings, with row indices in the first
#      column, column indices in the second, and ratings in
#      the third column
# ranks: all the ranks to be tested
# prop.validate: the proportion to be set aside in a validation set
cf.evaluate.ranks <- function(dat, ranks, prop.validate=0.05) {
  # Draw a training and validation set from the original training set
  train.rows <- cf.training.set(dat[,1], dat[,2], 1-prop.validate)
  dat.train <- dat[train.rows,]
  dat.validate <- dat[-train.rows,]
  
  # Compute a scaled version of the training set
  mat.train <- Incomplete(dat.train[,1], dat.train[,2], dat.train[,3])
  minimum <- min(dat.train[,3])
  maximum <- max(dat.train[,3])
  mat.scaled <- biScale(mat.train, maxit=1000, row.scale = FALSE, col.scale = FALSE)
  
  # For each rank, compute validation-set predictions
  pred <- lapply(ranks, function(r) {
    if (r == 0) {
      # Rank-0 fit is just sum of row and column indices
      imputed <- attr(mat.scaled, "biScale:row")$center[dat.validate[,1]] +
        attr(mat.scaled, "biScale:column")$center[dat.validate[,2]]
    } else {
      fit <- softImpute(mat.scaled, rank.max=r, lambda=0, maxit=1000)
      imputed <- impute(fit, dat.validate[,1], dat.validate[,2])
    }
    pmin(pmax(imputed, minimum), maximum)
  })
  data.frame(rank = ranks,
             r2 = sapply(pred, function(x) 1-sum((x-dat.validate[,3])^2) / sum((mean(dat.train[,3])-dat.validate[,3])^2)),
             MAE = sapply(pred, function(x) mean(abs(x-dat.validate[,3]))),
             RMSE = sapply(pred, function(x) sqrt(mean((x-dat.validate[,3])^2))))
}