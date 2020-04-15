install.packages("ggcorrplot")
library(ggcorrplot)
install.packages(("caret"))
library(caret)

hitters_raw <- read.csv("Hitters.csv")
hitters_num <-hitters_raw[,2:18]
ggcorrplot(round(cor(hitters_num),1), method="circle")
round(cor(hitters_num),1)

lm.mod <- lm(Salary ~., data = hitters_num)
summary(lm.mod)

#Plot Salary vs. CAtBat
plot(hitters_num$CAtBat,hitters_num$Salary, 
     xlab='Number of times at bat in the career', 
     ylab='Annual salary ($ in thousands)',cex.lab=1.1)
abline(lm(hitters_num$Salary ~ hitters_num$CAtBat), col="blue")

#Plot Salary vs. AtBat
plot(hitters_num$AtBat,hitters_num$Salary, 
     xlab='Number of times at bat in the season', 
     ylab='Annual salary ($ in thousands)',cex.lab=1.1)
abline(lm(hitters_num$Salary ~ hitters_num$AtBat), col="blue")

#Plot Salary vs. Hits
plot(hitters_num$Hits,hitters_num$Salary, 
     xlab='Number of hits in the season', 
     ylab='Annual salary ($ in thousands)',cex.lab=1.1)
abline(lm(hitters_num$Salary ~ hitters_num$Hits), col="blue")

#Normalize and split data
pp <- preProcess(hitters_raw, method=c("center", "scale"))
Hitters <- predict(pp, hitters_raw)
set.seed(15071)
train.obs <- sort(sample(seq_len(nrow(Hitters)), 0.7*nrow(Hitters)))
train <- Hitters[train.obs,2:21]
test <- Hitters[-train.obs,2:21]

#Fit a Linear Regression and predict the test set
lin.mod <- lm(train$Salary ~ ., data = train)
pred.train = predict(lin.mod, newdata = train)
summary(lin.mod)

#Out-of-Sample R-squared
pred.test = predict(lin.mod, newdata = test)
SSE.test = sum((pred.test - test$Salary)^2)
SST.test = sum((test$Salary - mean(train$Salary))^2)
OSR2 = 1 - SSE.test/SST.test
OSR2

#Fit a restricted Linear Regression with variables with significance
head(train)
lin.mod2 <- lm(Salary ~ AtBat+Hits+Walks+CWalks+PutOuts,
               data = train)
pred.train = predict(lin.mod2, newdata = train)
summary(lin.mod2)

#Out-of-Sample R-squared
pred.test = predict(lin.mod2, newdata = test)
SSE.test = sum((pred.test - test$Salary)^2)
SST.test = sum((test$Salary - mean(train$Salary))^2)
OSR2 = 1 - SSE.test/SST.test
OSR2

#c) Regularization
##### Ridge Regression
### Run Ridge Regression in the train Set
x.train=model.matrix(Salary~.-1,data=train) 
y.train=train$Salary
x.test=model.matrix(Salary~.-1,data=test) 
y.test=test$Salary

all.lambdas <- c(exp(seq(15, -10, -.1)))
cv.ridge=cv.glmnet(x.train,y.train,alpha=0,lambda=all.lambdas, nfold=10)
cv.lasso=cv.glmnet(x.train,y.train,alpha=1,lambda=all.lambdas, nfold=10)

plot(cv.ridge)
plot(cv.lasso)
cv.ridge$lambda.min
cv.lasso$lambda.min
### Prediction on the train and test sets
# Re-train ridge regression and LASSO models on full training set.
ridge.final <- glmnet(x.train, y.train, alpha=0, lambda=cv.ridge$lambda.min)
lasso.final <- glmnet(x.train, y.train, alpha=1, lambda=cv.lasso$lambda.min)

ridge.final$beta
lasso.final$beta

pred.train.ridge <- predict(ridge.final, x.train)
pred.train.lasso <- predict(lasso.final, x.train)

R2.ridge <- 1-sum((pred.train.ridge-train$Salary)^2)/sum((mean(train$Salary)-train$Salary)^2)
R2.ridge
R2.lasso <- 1-sum((pred.train.lasso-train$Salary)^2)/sum((mean(train$Salary)-train$Salary)^2)
R2.lasso

pred.test.ridge <- predict(ridge.final, x.test)
pred.test.lasso <- predict(lasso.final, x.test)

OSR2.ridge <- 1-sum((pred.test.ridge-test$Salary)^2)/sum((mean(train$Salary)-test$Salary)^2)
OSR2.ridge
OSR2.lasso <- 1-sum((pred.test.lasso-test$Salary)^2)/sum((mean(train$Salary)-test$Salary)^2)
OSR2.lasso

#D)
nvars=6
all_constrained_validation_runs <- cv.lasso$nzero <= nvars
best_constrained_run <- which.min(cv.lasso$cvm[all_constrained_validation_runs])
best_constrained_lambda <- cv.lasso$lambda[best_constrained_run]
best_constrained_lambda
#retrain lasso
lasso.final2 <- glmnet(x.train, y.train, alpha=1, lambda=best_constrained_lambda)
pred.train.lasso2 <- predict(lasso.final2, x.train)
R2.lasso2 <- 1-sum((pred.train.lasso2-train$Salary)^2)/sum((mean(train$Salary)-train$Salary)^2)
R2.lasso2

pred.test.lasso2 <- predict(lasso.final2, x.test)
OSR2.lasso2 <- 1-sum((pred.test.lasso2-test$Salary)^2)/sum((mean(train$Salary)-test$Salary)^2)
OSR2.lasso2

lasso.final2$beta


#Forward stepwise selection with 10-fold cross-validation
fs <- train(Salary~.,train,
            methods = "leapForward",
            trControl = trainControl(method = "cv", number = 10), 
            tuneGrid = expand.grid(.nvmax=seq(1,15)))
summary(fs)            
            

#XGBOOST
library(xgboost)
set.seed(1)
cv.xgb <- train(y = train$Salary,
                         x = data.matrix(subset(train, select=-c(Salary))),
                         method = "xgbTree",
                         trControl = trainControl(method="cv", number=5))

# The out-of-sample R2 is 0.6305704:
model.xgb <- cv.xgb$finalModel
preds.xgb.test <- predict(model.xgb, newdata = data.matrix(subset(test, select=-c(Salary))))
preds.xgb.train <- predict(model.xgb, newdata = data.matrix(subset(train, select=-c(Salary))))
r.xgb <- 1-sum((preds.xgb.train - train$Salary)^2)/sum((mean(train$Salary)-train$Salary)^2)
osr.xgb <- 1 - sum((preds.xgb.test - test$Salary)^2)/sum((mean(train$Salary) - test$Salary)^2)
r.xgb
osr.xgb
summary(model.xgb)
# By looking at the results and the best parameters, 
# we can see what parameters are available to tune and if there is further tuning necessary.
cv.xgb$results
cv.xgb$bestTune

# Notice that the value chosen for max_depth is the minimum value of 1.
# This suggests that the xgboost algorithm wants less variable interaction for each individual tree.
# We can't decrease max_depth any further, but we can remove the interaction terms in the diabetes dataset before building the model.
simple <- diabetes[,c("age", "sex", "bmi", "map", "tc", "ldl", "hdl", "tch", "ltg", "glu", "y")]
simple.train <- simple[split,]
diabetes.simple.test <- diabetes.simple[-split,]
set.seed(1)
diabetes.cv.xgb <- train(y = diabetes.simple.train$y,
                         x = data.matrix(subset(diabetes.simple.train, select=-c(y))),
                         method = "xgbTree",
                         trControl = trainControl(method="cv", number=5))
diabetes.cv.xgb$results
diabetes.cv.xgb$bestTune

# The out-of-sample R2 is 0.521, a decent improvement:
diabetes.model.xgb <- diabetes.cv.xgb$finalModel
diabetes.preds.xgb <- predict(diabetes.model.xgb, newdata = data.matrix(subset(diabetes.simple.test, select=-c(y))))
diabetes.osr.xgb <- 1 - sum((diabetes.preds.xgb - diabetes.test$y)^2)/sum((mean(diabetes.train$y) - diabetes.test$y)^2)

# Note that the R2's for lasso, ridge regression, and boosting are too close to make the assertion that XGBoost > lasso > ridge regression on this dataset.
# In fact, if we just change the random splits by replacing set.seed(1) with set.seed(101) throught the file, we will obtain the ordering lasso > ridge > XGBoost.

# For reference, to train an XGBoost model without using the train function or cross-validation, the basic command is
diabetes.model.xgb <- xgboost(data = data.matrix(subset(diabetes.train, select=-c(y))), label = diabetes.train$y, 
                              nrounds = 50, max_depth = 1, eta = 0.3, gamma = 0, colsample_bytree = 0.6, min_child_weight = 1, subsample = 1)
# This would replace diabetes.cv.xgb$finalModel.
