rm(list=ls())
library(data.table)
library(rpart)          # one of the main ways to train classification and regression trees in R
library(caret)      
library(randomForest)   # for bagging and random forests
library(ROCR) 
library(gbm)            # generalized boosting regression modeling


# read the data
data <- fread("CAC40.csv")

# format dates
data$Date <- as.Date(data$Date, format = "%m/%d/%Y")

# create a target variable with direction of CAC40
data$Direction <- factor(ifelse(data$CAC40>0,"Up","Down"))

# create a formula to avoid repetition
fm.class <- as.formula(Direction~CAC40_LAG1 + CAC40_LAG2 + DAX_LAG1 + DAX_LAG2 
                       + SP500_LAG1 + SP500_LAG2 + MSCIEM_LAG1 + MSCIEM_LAG2 
                       + EURUSD_LAG1 + EURUSD_LAG2)

# split sample into training data (2010-2017) and test data (2017-2020)
train_inds=(year(data$Date)<=2017)
train <- data[train_inds,]
test <- data[!train_inds,]

# create an object to store the test errors of various models
test.errors <- data.frame(tree = 0, bagging = 0, randomforest = 0, boost1 = 0,
                          boost2 = 0, boost3 = 0, boost4 = 0, boost5 = 0)

# fit a single classification tree using rpart
set.seed(10891)
fit.class <- rpart(fm.class, method = "class", data = train)


# plot trees
par(mfrow = c(1,1), xpd = NA) # otherwise on some devices the text is clipped
plot(fit.class, uniform = TRUE)
text(fit.class, all = TRUE, cex = 1)


par(mfrow = c(1,1))
printcp(fit.class) # display the results
plotcp(fit.class) # visualize cross-validation results
# in this example, the lowest error is obtained using the whole tree. The error bars on the
# plot show one standard deviation of the cross-validated error. A good choice of cp 
# for pruning is often the leftmost value for which the mean lies below the horizontal line.
# In this case, this would be the tree of size 4

# The function below shows a detailed summary of splits
summary(fit.class) 

# Let's prune the tree using the one-standard deviation rule:
pfit.class <- prune(fit.class, cp = 0.026)

# plot the pruned tree
par(mfrow = c(1,1), xpd = NA)
plot(pfit.class, uniform = TRUE, main = "Pruned Classification Tree - CAC40")
text(pfit.class, all = TRUE, cex = .8)


# error rates of full tree and pruned tree on training and test sets
fit.class.pred.train <- predict(fit.class, train, type = "class")
fit.class.pred.test <- predict(fit.class, test, type = "class")
pfit.class.pred.train <- predict(pfit.class, train, type = "class")
pfit.class.pred.test <- predict(pfit.class, test, type = "class")

# confusion matrices
confusionMatrix(data = fit.class.pred.train, reference = train$Direction, positive = 'Up')
confusionMatrix(data = fit.class.pred.test, reference = test$Direction, positive = 'Up')
confusionMatrix(data = pfit.class.pred.train, reference = train$Direction, positive = 'Up')
confusionMatrix(data = pfit.class.pred.test, reference = test$Direction, positive = 'Up')

# error rate for pruned tree on test set
cm <- confusionMatrix(data = pfit.class.pred.test,
                      reference = test$Direction, positive = 'Up')
test.errors$tree <- 1 - cm$overall[1]

# bagging/ RF
# to use the randomForest function, the data need to be in the format below
xtrain <- model.matrix(fm.class, data = train)
ytrain <- train$Direction
xtest <- model.matrix(fm.class, data = test)
ytest <- test$Direction

# for bagging, we use the randomForest funcion, but we need to tell the function
# to use all variables (11 in this case)
bagmodel <- randomForest(xtrain, ytrain, xtest, ytest, # training and test data
                         ntree = 1000, # number of trees in the model
                         mtry = 11, # number of variables to try
                         maxnodes = 20, # maximum number of leaves
                         importance = TRUE, # calculate variable importance
                         keep.forest = TRUE) # keep all trees 

varImpPlot(bagmodel) # variance importance plot
bagmodel$test$err.rate[1000] # test misclassification
bagmodel$test$confusion # confusion matrix

# store bagging test error rate
test.errors$bagging <- bagmodel$test$err.rate[1000]

# random forest - we use the same function, without the option mtry
RFmodel <- randomForest(xtrain, ytrain, xtest, ytest,
                        ntree = 1000, maxnodes = 20,importance = TRUE,
                        keep.forest = TRUE)
varImpPlot(RFmodel)
RFmodel$test$err.rate[1000,] # test misclassification
RFmodel$test$confusion # confusion matrix

# store random forest test error rate
test.errors$randomforest <- RFmodel$test$err.rate[1000]

# plot OOB and test error estimates
op <- par(cex = 0.5)
plot(bagmodel$err.rate[,1], type = "l", xlab = "Number of trees", 
     ylab = "Misclassification rate", col = "blue", ylim = c(0.3,0.6))
lines(RFmodel$err.rate[,1], type = "l", col = "green")
lines(bagmodel$test$err.rate[,1], type = "l", col = "red")
lines(RFmodel$test$err.rate[,1], type = "l",  col = "orange")
abline(h = test.errors$tree, lty = 3)
legend(x="bottomleft",c("OOB - Bagging","OOB - Random Forest", "Test - Bagging", "Test - Random Forest", "Single classification tree"), lty=c(1,1,1,1,3), lwd=c(1,1,1,1,1),col=c("blue","green","red","orange","black"))
# in this example, bagging is consistently better than RF.

# boosting using the gbm package
# The gbm package requires a specific format for the formula, as below: 
fmclassgbm <- as.formula(as.integer(CAC40>0)~CAC40_LAG1 + CAC40_LAG2 + DAX_LAG1
                         + DAX_LAG2 + SP500_LAG1 + SP500_LAG2 + MSCIEM_LAG1 
                         + MSCIEM_LAG2 + EURUSD_LAG1 + EURUSD_LAG2)

# we are going to fit 5 types of boosting models. The difference is the type of
# tree that is used as the base learner. The interaction.depth parameters 
# controls the maximum depth of each tree. 
set.seed(123)
fitboost1 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "bernoulli", interaction.depth = 1,
                  shrinkage = 0.001, train.fraction = .6)
fitboost2 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "bernoulli", interaction.depth = 2, 
                  shrinkage = 0.001, train.fraction = .6)
fitboost3 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "bernoulli", interaction.depth = 3,
                  shrinkage = 0.001, train.fraction = .6)
fitboost4 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "bernoulli", interaction.depth = 4,
                  shrinkage = 0.001, train.fraction = .6)
fitboost5 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "bernoulli", interaction.depth = 5,
                  shrinkage = 0.001, train.fraction = .6)
summary(fitboost1)

# the gbm.perf function estimates the optimal number of boosting iterations 
# for a gbm object
fitboost1.optTrees <- gbm.perf(fitboost1)
title('Plot for interaction.depth = 1')
fitboost2.optTrees <- gbm.perf(fitboost2)
title('Plot for interaction.depth = 2')
fitboost3.optTrees <- gbm.perf(fitboost3)
title('Plot for interaction.depth = 3')
fitboost4.optTrees <- gbm.perf(fitboost4)
title('Plot for interaction.depth = 4')
fitboost5.optTrees <- gbm.perf(fitboost5)
title('Plot for interaction.depth = 5')

# checking accuracy in the training set - commands for the first boosting model
# shown with breaks below for better readability 
fitboost1.predtrain <- as.factor(ifelse(predict(fitboost1, 
                                                newdata = train, 
                                                n.trees = fitboost1.optTrees, 
                                                type = "response") > 0.5,
                                        "Up", "Down"))
confusionMatrix(data = fitboost1.predtrain, reference = train$Direction, 
                positive = 'Up')

# the commands below do the same for the other boosting models
fitboost2.predtrain <- as.factor(ifelse(predict(fitboost2,newdata = train, n.trees = fitboost2.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost2.predtrain,reference=train$Direction, positive='Up')
fitboost3.predtrain <- as.factor(ifelse(predict(fitboost3,newdata = train, n.trees = fitboost3.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost3.predtrain,reference=train$Direction, positive='Up')
fitboost4.predtrain <- as.factor(ifelse(predict(fitboost4,newdata = train, n.trees = fitboost4.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost3.predtrain,reference=train$Direction, positive='Up')
fitboost5.predtrain <- as.factor(ifelse(predict(fitboost5,newdata = train, n.trees = fitboost5.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost3.predtrain,reference=train$Direction, positive='Up')

# checking accuracy in the test set
fitboost1.predtest <- as.factor(ifelse(predict(fitboost1,newdata = test, n.trees = fitboost1.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost1.predtest,reference=test$Direction, positive='Up')
fitboost2.predtest <- as.factor(ifelse(predict(fitboost2,newdata = test, n.trees = fitboost2.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost2.predtest,reference=test$Direction, positive='Up')
fitboost3.predtest <- as.factor(ifelse(predict(fitboost3,newdata = test, n.trees = fitboost3.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost3.predtest,reference=test$Direction, positive='Up')
fitboost4.predtest <- as.factor(ifelse(predict(fitboost4,newdata = test, n.trees = fitboost4.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost4.predtest,reference=test$Direction, positive='Up')
fitboost5.predtest <- as.factor(ifelse(predict(fitboost5,newdata = test, n.trees = fitboost5.optTrees, type = "response")>0.5,"Up","Down"))
confusionMatrix(data=fitboost5.predtest,reference=test$Direction, positive='Up')

# it seems the best performing model here was using interaction.depth=4

# Now we refit the models using the optimal number of trees from the gbm.perf function
set.seed(123)
fitboost1.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost1.optTrees, 
                        distribution = "bernoulli", interaction.depth = 1, 
                        shrinkage = 0.001, train.fraction = 1)
fitboost2.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost2.optTrees, 
                        distribution = "bernoulli", interaction.depth = 2, 
                        shrinkage = 0.001, train.fraction = 1)
fitboost3.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost3.optTrees, 
                        distribution = "bernoulli", interaction.depth = 3, 
                        shrinkage = 0.001, train.fraction = 1)
fitboost4.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost4.optTrees, 
                        distribution = "bernoulli", interaction.depth = 4, 
                        shrinkage = 0.001, train.fraction = 1)
fitboost5.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost5.optTrees, 
                        distribution = "bernoulli", interaction.depth = 5, 
                        shrinkage = 0.001, train.fraction = 1)

# check performance of the refited models
fitboost1.refit.predtest <- as.factor(ifelse(predict(fitboost1.refit, newdata = test, n.trees = fitboost1.optTrees, type = "response")>0.5,"Up","Down"))
cm <- confusionMatrix(data=fitboost1.refit.predtest,reference=test$Direction, positive='Up')
test.errors$boost1 <- 1 - cm$overall[1]

fitboost2.refit.predtest <- as.factor(ifelse(predict(fitboost2.refit,newdata = test, n.trees = fitboost2.optTrees, type = "response")>0.5,"Up","Down"))
cm <- confusionMatrix(data=fitboost2.refit.predtest,reference=test$Direction, positive='Up')
test.errors$boost2 <- 1 - cm$overall[1]

fitboost3.refit.predtest = as.factor(ifelse(predict(fitboost3.refit,newdata = test, n.trees = fitboost3.optTrees, type = "response")>0.5,"Up","Down"))
cm <- confusionMatrix(data=fitboost3.refit.predtest,reference=test$Direction, positive='Up')
test.errors$boost3 <- 1 - cm$overall[1]

fitboost4.refit.predtest <- as.factor(ifelse(predict(fitboost4.refit,newdata = test, n.trees = fitboost4.optTrees, type = "response")>0.5,"Up","Down"))
cm <- confusionMatrix(data=fitboost4.refit.predtest,reference=test$Direction, positive='Up')
test.errors$boost4 <- 1 - cm$overall[1]

fitboost5.refit.predtest <- as.factor(ifelse(predict(fitboost5.refit,newdata = test, n.trees = fitboost5.optTrees, type = "response")>0.5,"Up","Down"))
cm <- confusionMatrix(data=fitboost5.refit.predtest,reference=test$Direction, positive='Up')
test.errors$boost5 <- 1 - cm$overall[1]

# comparison of a single pruned tree, bagging, random forest and the boosting models
# Boosting model with with a level of interaction of two is the best one
op <- par(cex = 1)
barplot(as.matrix(test.errors))


