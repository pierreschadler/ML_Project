rm(list=ls())
library(data.table)
library(rpart)          # one of the main ways to train classification and regression trees in R
library(caret)      
library(randomForest)   # for bagging and random forests
library(ROCR) 
library(gbm)            # generalized boosting regression modeling
library(xts)             # time series objects
library(dplyr)           # illustrate dplyr and piping
library(PerformanceAnalytics)


rv_data <- fread("oxfordmanrealizedvolatilityindices.csv")

rv_subset <- rv_data %>% 
  filter(Symbol == ".FTSE") %>%
  select(Date = V1, 
         RV = rv5, 
         close_price,
         open_to_close,
         bv,
         bv_ss,
         medrv,
         rk_parzen,
         rk_th2,
         rk_twoscale,
         rsv,
         rsv_ss,
         rv10,
         rv10_ss,
         rv5,
         rv5_ss)


# format Date 
rv_subset$Date <- as.Date(rv_subset$Date)

rv_subset <- as.xts(rv_subset)
rv_subset$RV <- rv_subset$RV^.5 * sqrt(252)
rv_subset$rets <- CalculateReturns(rv_subset$close_price)

par(mfrow = c(1, 1) )
plot(rv_subset$rets)
plot(rv_subset$open_to_close)
plot(rv_subset$RV)

# Computing realized volatility moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_RV1 <- frollmean(rv_subset$RV, 1)
rv_subset$MA_RV5 <- frollmean(rv_subset$RV, 5)
rv_subset$MA_RV10 <- frollmean(rv_subset$RV, 10)
rv_subset$MA_RV22 <- frollmean(rv_subset$RV, 22)
rv_subset$MA_RV50 <- frollmean(rv_subset$RV, 50)

# Computing Bipower Variation moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_bv1 <- frollmean(rv_subset$bv, 1)
rv_subset$MA_bv5 <- frollmean(rv_subset$bv, 5)
rv_subset$MA_bv10 <- frollmean(rv_subset$bv, 10)
rv_subset$MA_bv22 <- frollmean(rv_subset$bv, 22)
rv_subset$MA_bv50 <- frollmean(rv_subset$bv, 50)

# Computing Median Realized Variance moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_medrv1 <- frollmean(rv_subset$medrv, 1)
rv_subset$MA_medrv5 <- frollmean(rv_subset$medrv, 5)
rv_subset$MA_medrv10 <- frollmean(rv_subset$medrv, 10)
rv_subset$MA_medrv22 <- frollmean(rv_subset$medrv, 22)
rv_subset$MA_medrv50 <- frollmean(rv_subset$medrv, 50)

# Computing Realized Kernel Variance (Non-Flat Parzen) moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_rk_parzen1 <- frollmean(rv_subset$rk_parzen, 1)
rv_subset$MA_rk_parzen5 <- frollmean(rv_subset$rk_parzen, 5)
rv_subset$MA_rk_parzen10 <- frollmean(rv_subset$rk_parzen, 10)
rv_subset$MA_rk_parzen22 <- frollmean(rv_subset$rk_parzen, 22)
rv_subset$MA_rk_parzen50 <- frollmean(rv_subset$rk_parzen, 50)

# Computing Realized Kernel Variance (Tukey-Hanning(2)) moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_rk_th21 <- frollmean(rv_subset$rk_th2, 1)
rv_subset$MA_rk_th25 <- frollmean(rv_subset$rk_th2, 5)
rv_subset$MA_rk_th210 <- frollmean(rv_subset$rk_th2, 10)
rv_subset$MA_rk_th222 <- frollmean(rv_subset$rk_th2, 22)
rv_subset$MA_rk_th250 <- frollmean(rv_subset$rk_th2, 50)

# Computing Realized Kernel Variance (Two-Scale/Bartlett) moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_rk_twoscale1 <- frollmean(rv_subset$rk_twoscale, 1)
rv_subset$MA_rk_twoscale5 <- frollmean(rv_subset$rk_twoscale, 5)
rv_subset$MA_rk_twoscale10 <- frollmean(rv_subset$rk_twoscale, 10)
rv_subset$MA_rk_twoscale22 <- frollmean(rv_subset$rk_twoscale, 22)
rv_subset$MA_rk_twoscale50 <- frollmean(rv_subset$rk_twoscale, 50)

# Computing Realized Semi-variance moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_rsv1 <- frollmean(rv_subset$rsv, 1)
rv_subset$MA_rsv5 <- frollmean(rv_subset$rsv, 5)
rv_subset$MA_rsv10 <- frollmean(rv_subset$rsv, 10)
rv_subset$MA_rsv22 <- frollmean(rv_subset$rsv, 22)
rv_subset$MA_rsv50 <- frollmean(rv_subset$rsv, 50)

# Computing Realized Variance (10-min) moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_rv101 <- frollmean(rv_subset$rv10, 1)
rv_subset$MA_rv105 <- frollmean(rv_subset$rv10, 5)
rv_subset$MA_rv1010 <- frollmean(rv_subset$rv10, 10)
rv_subset$MA_rv1022 <- frollmean(rv_subset$rv10, 22)
rv_subset$MA_rv1050 <- frollmean(rv_subset$rv10, 50)

# Computing Realized Variance (5-min) moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_rv51 <- frollmean(rv_subset$rv5, 1)
rv_subset$MA_rv55 <- frollmean(rv_subset$rv5, 5)
rv_subset$MA_rv510 <- frollmean(rv_subset$rv5, 10)
rv_subset$MA_rv522 <- frollmean(rv_subset$rv5, 22)
rv_subset$MA_rv550 <- frollmean(rv_subset$rv5, 50)

# Computing daily returns moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_RET1 <- frollmean(rv_subset$rets, 1)
rv_subset$MA_RET5 <- frollmean(rv_subset$rets, 5)
rv_subset$MA_RET10 <- frollmean(rv_subset$rets, 10)
rv_subset$MA_RET22 <- frollmean(rv_subset$rets, 22)
rv_subset$MA_RET50 <- frollmean(rv_subset$rets, 50)

# Computing Open to Close Return moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_INTDAYRET1 <- frollmean(rv_subset$open_to_close, 1)
rv_subset$MA_INTDAYRET5 <- frollmean(rv_subset$open_to_close, 5)
rv_subset$MA_INTDAYRET10 <- frollmean(rv_subset$open_to_close, 10)
rv_subset$MA_INTDAYRET22 <- frollmean(rv_subset$open_to_close, 22)
rv_subset$MA_INTDAYRET50 <- frollmean(rv_subset$open_to_close, 50)

# lag the moving averages:
rv_subset$MA_RV1 <- stats::lag(rv_subset$MA_RV1, 1)
rv_subset$MA_RV5 <- stats::lag(rv_subset$MA_RV5, 1)
rv_subset$MA_RV10 <- stats::lag(rv_subset$MA_RV10, 1)
rv_subset$MA_RV22 <- stats::lag(rv_subset$MA_RV22, 1)
rv_subset$MA_RV50 <- stats::lag(rv_subset$MA_RV50, 1)

rv_subset$MA_RET1 <- stats::lag(rv_subset$MA_RET1, 1)
rv_subset$MA_RET5 <- stats::lag(rv_subset$MA_RET5, 1)
rv_subset$MA_RET10 <- stats::lag(rv_subset$MA_RET10, 1)
rv_subset$MA_RET22 <- stats::lag(rv_subset$MA_RET22, 1)
rv_subset$MA_RET50 <- stats::lag(rv_subset$MA_RET50, 1)

rv_subset$MA_INTDAYRET1 <- stats::lag(rv_subset$MA_INTDAYRET1, 1)
rv_subset$MA_INTDAYRET5 <- stats::lag(rv_subset$MA_INTDAYRET5, 1)
rv_subset$MA_INTDAYRET10 <- stats::lag(rv_subset$MA_INTDAYRET10, 1)
rv_subset$MA_INTDAYRET22 <- stats::lag(rv_subset$MA_INTDAYRET22, 1)
rv_subset$MA_INTDAYRET50 <- stats::lag(rv_subset$MA_INTDAYRET50, 1)

rv_subset$MA_bv1 <- stats::lag(rv_subset$MA_bv1, 1)
rv_subset$MA_bv5 <- stats::lag(rv_subset$MA_bv5, 1)
rv_subset$MA_bv10 <- stats::lag(rv_subset$MA_bv10, 1)
rv_subset$MA_bv22 <- stats::lag(rv_subset$MA_bv22, 1)
rv_subset$MA_bv50 <- stats::lag(rv_subset$MA_bv50, 1)

rv_subset$MA_medrv1 <- stats::lag(rv_subset$MA_medrv1, 1)
rv_subset$MA_medrv5 <- stats::lag(rv_subset$MA_medrv5, 1)
rv_subset$MA_medrv10 <- stats::lag(rv_subset$MA_medrv10, 1)
rv_subset$MA_medrv22 <- stats::lag(rv_subset$MA_medrv22, 1)
rv_subset$MA_medrv50 <- stats::lag(rv_subset$MA_medrv50, 1)

rv_subset$MA_rk_parzen1 <- stats::lag(rv_subset$MA_rk_parzen1, 1)
rv_subset$MA_rk_parzen5 <- stats::lag(rv_subset$MA_rk_parzen5, 1)
rv_subset$MA_rk_parzen10 <- stats::lag(rv_subset$MA_rk_parzen10, 1)
rv_subset$MA_rk_parzen22 <- stats::lag(rv_subset$MA_rk_parzen22, 1)
rv_subset$MA_rk_parzen50 <- stats::lag(rv_subset$MA_rk_parzen50, 1)

rv_subset$MA_rk_th21 <- stats::lag(rv_subset$MA_rk_th21, 1)
rv_subset$MA_rk_th25 <- stats::lag(rv_subset$MA_rk_th25, 1)
rv_subset$MA_rk_th210 <- stats::lag(rv_subset$MA_rk_th210, 1)
rv_subset$MA_rk_th222 <- stats::lag(rv_subset$MA_rk_th222, 1)
rv_subset$MA_rk_th250 <- stats::lag(rv_subset$MA_rk_th250, 1)

rv_subset$MA_rk_twoscale1 <- stats::lag(rv_subset$MA_rk_twoscale1, 1)
rv_subset$MA_rk_twoscale5 <- stats::lag(rv_subset$MA_rk_twoscale5, 1)
rv_subset$MA_rk_twoscale10 <- stats::lag(rv_subset$MA_rk_twoscale10, 1)
rv_subset$MA_rk_twoscale22 <- stats::lag(rv_subset$MA_rk_twoscale22, 1)
rv_subset$MA_rk_twoscale50 <- stats::lag(rv_subset$MA_rk_twoscale50, 1)

rv_subset$MA_rsv1 <- stats::lag(rv_subset$MA_rsv1, 1)
rv_subset$MA_rsv5 <- stats::lag(rv_subset$MA_rsv5, 1)
rv_subset$MA_rsv10 <- stats::lag(rv_subset$MA_rsv10, 1)
rv_subset$MA_rsv22 <- stats::lag(rv_subset$MA_rsv22, 1)
rv_subset$MA_rsv50 <- stats::lag(rv_subset$MA_rsv50, 1)

rv_subset$MA_rv101 <- stats::lag(rv_subset$MA_rv101, 1)
rv_subset$MA_rv105 <- stats::lag(rv_subset$MA_rv105, 1)
rv_subset$MA_rv1010 <- stats::lag(rv_subset$MA_rv1010, 1)
rv_subset$MA_rv1022 <- stats::lag(rv_subset$MA_rv1022, 1)
rv_subset$MA_rv1050 <- stats::lag(rv_subset$MA_rv1050, 1)

rv_subset$MA_rv51 <- stats::lag(rv_subset$MA_rv51, 1)
rv_subset$MA_rv55 <- stats::lag(rv_subset$MA_rv55, 1)
rv_subset$MA_rv510 <- stats::lag(rv_subset$MA_rv510, 1)
rv_subset$MA_rv522 <- stats::lag(rv_subset$MA_rv522, 1)
rv_subset$MA_rv550 <- stats::lag(rv_subset$MA_rv550, 1)




train <- rv_subset["/2019", ]
test<-  rv_subset["2020/", ]



train <- na.omit(train)



HAR_RV <- lm(RV ~ MA_RV1 + MA_RV5 + MA_RV10+ MA_RV22+ MA_RV50 +
               MA_RET1 + MA_RET5 + MA_RET10 + MA_RET22 + MA_RET50 +
               bv + medrv + rk_parzen + rk_th2 + rk_twoscale + rsv + rv10 + rv5_ss,
             data = train)

HAR_RV<-lm(RV~ + close_price + open_to_close + bv + medrv + rk_parzen + rk_th2 + 
             rk_twoscale + rsv + rv10 + rv5_ss + rets + MA_RV1 + MA_RV5 + MA_RV10 + 
             MA_RV22 + MA_RV50 + MA_bv1 + MA_bv5 + MA_bv10 + MA_bv22 + MA_bv50 + 
             MA_medrv1 + MA_medrv5 + MA_medrv10 + MA_medrv22 + MA_medrv50 + 
             MA_rk_parzen1 + MA_rk_parzen5 + MA_rk_parzen10 + MA_rk_parzen22 + 
             MA_rk_parzen50 + MA_rk_th21 + MA_rk_th25 + MA_rk_th210 + MA_rk_th222 + 
             MA_rk_th250 + MA_rk_twoscale1 + MA_rk_twoscale5 + MA_rk_twoscale10 + 
             MA_rk_twoscale22 + MA_rk_twoscale50 + MA_rsv1 + MA_rsv5 + MA_rsv10 + 
             MA_rsv22 + MA_rsv50 + MA_rv101 + MA_rv105 + MA_rv1010 + MA_rv1022 + 
             MA_rv1050 + MA_rv51 + MA_rv55 + MA_rv510 + MA_rv522 + MA_rv550 + MA_RET1 + 
             MA_RET5 + MA_RET10 + MA_RET22 + MA_RET50 + MA_INTDAYRET1 + MA_INTDAYRET5 + 
             MA_INTDAYRET10 + MA_INTDAYRET22 + MA_INTDAYRET50,
           data=train)

summary(HAR_RV)

train$pred_HAR_RV <- as.numeric(predict(HAR_RV))




plot(train["2019", 
                c("RV", "pred_HAR_RV")], 
     col=c("black", "red"),
     lwd = c(1,1), 
     main = "Actual vs predicted RV", 
     legend.loc = "topleft")

# create functions to calculate R2 and RMSE
MSE = function(y_actual, y_predict){
  sqrt(mean((y_actual-y_predict)^2))
}

# recall that R2 is given by 1 - SSR/SST
 RSQUARE = 
  function(y_actual,y_predict){
    1 - sum( (y_actual-y_predict)^2)/sum( (y_actual-mean(y_actual))^2)
  }

MSE(train$RV, train$pred_HAR_RV)

RSQUARE(train$RV, train$pred_HAR_RV)



# create an object to store the test errors of various models
test.errors <- data.frame(tree = 0, bagging = 0, randomforest = 0, boost1 = 0,
                          boost2 = 0, boost3 = 0, boost4 = 0, boost5 = 0)


# boosting using the gbm package
# The gbm package requires a specific format for the formula, as below: 

# fmclassgbm <- as.formula(as.integer(CAC40>0)~CAC40_LAG1 + CAC40_LAG2 + 
#                            DAX_LAG1+ DAX_LAG2 + SP500_LAG1 + SP500_LAG2 + 
#                            MSCIEM_LAG1 + MSCIEM_LAG2 + EURUSD_LAG1 + EURUSD_LAG2)

fmclassgbm <- as.formula(RV~ + close_price + open_to_close + bv + medrv + rk_parzen + rk_th2 + 
                           rk_twoscale + rsv + rv10 + rv5_ss + rets + MA_RV1 + MA_RV5 + MA_RV10 + 
                           MA_RV22 + MA_RV50 + MA_bv1 + MA_bv5 + MA_bv10 + MA_bv22 + MA_bv50 + 
                           MA_medrv1 + MA_medrv5 + MA_medrv10 + MA_medrv22 + MA_medrv50 + 
                           MA_rk_parzen1 + MA_rk_parzen5 + MA_rk_parzen10 + MA_rk_parzen22 + 
                           MA_rk_parzen50 + MA_rk_th21 + MA_rk_th25 + MA_rk_th210 + MA_rk_th222 + 
                           MA_rk_th250 + MA_rk_twoscale1 + MA_rk_twoscale5 + MA_rk_twoscale10 + 
                           MA_rk_twoscale22 + MA_rk_twoscale50 + MA_rsv1 + MA_rsv5 + MA_rsv10 + 
                           MA_rsv22 + MA_rsv50 + MA_rv101 + MA_rv105 + MA_rv1010 + MA_rv1022 + 
                           MA_rv1050 + MA_rv51 + MA_rv55 + MA_rv510 + MA_rv522 + MA_rv550 + MA_RET1 + 
                           MA_RET5 + MA_RET10 + MA_RET22 + MA_RET50 + MA_INTDAYRET1 + MA_INTDAYRET5 + 
                           MA_INTDAYRET10 + MA_INTDAYRET22 + MA_INTDAYRET50)

# we are going to fit 5 types of boosting models. The difference is the type of
# tree that is used as the base learner. The interaction.depth parameters 
# controls the maximum depth of each tree. 

set.seed(123)
fitboost1 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 1,
                  shrinkage = 0.001, train.fraction = .6)
fitboost2 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 2,
                  shrinkage = 0.001, train.fraction = .6)
fitboost3 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 3,
                  shrinkage = 0.001, train.fraction = .6)
fitboost4 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 4,
                  shrinkage = 0.001, train.fraction = .6)
fitboost5 <- gbm( fmclassgbm, data = train, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 5,
                  shrinkage = 0.001, train.fraction = .6)


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
fitboost1.predtrain <- as.factor(predict(fitboost1,newdata = train, n.trees = fitboost1.optTrees))
postResample(pred = as.numeric(fitboost1.predtrain), obs = as.numeric(train$RV))
fitboost2.predtrain <- as.factor(predict(fitboost2,newdata = train, n.trees = fitboost2.optTrees))
postResample(pred = as.numeric(fitboost2.predtrain), obs = as.numeric(train$RV))
fitboost3.predtrain <- as.factor(predict(fitboost3,newdata = train, n.trees = fitboost3.optTrees))
postResample(pred = as.numeric(fitboost3.predtrain), obs = as.numeric(train$RV))
fitboost4.predtrain <- as.factor(predict(fitboost4,newdata = train, n.trees = fitboost4.optTrees))
postResample(pred = as.numeric(fitboost4.predtrain), obs = as.numeric(train$RV))
fitboost5.predtrain <- as.factor(predict(fitboost5,newdata = train, n.trees = fitboost5.optTrees))
postResample(pred = as.numeric(fitboost5.predtrain), obs = as.numeric(train$RV))


# checking accuracy in the test set
fitboost1.predtest <- as.factor(predict(fitboost1,newdata = test, n.trees = fitboost1.optTrees))
postResample(pred = as.numeric(fitboost1.predtest), obs = as.numeric(test$RV))
fitboost2.predtest <- as.factor(predict(fitboost2,newdata = test, n.trees = fitboost2.optTrees))
postResample(pred = as.numeric(fitboost2.predtest), obs = as.numeric(test$RV))
fitboost3.predtest <- as.factor(predict(fitboost3,newdata = test, n.trees = fitboost3.optTrees))
postResample(pred = as.numeric(fitboost3.predtest), obs = as.numeric(test$RV))
fitboost4.predtest <- as.factor(predict(fitboost4,newdata = test, n.trees = fitboost4.optTrees))
postResample(pred = as.numeric(fitboost4.predtest), obs = as.numeric(test$RV))
fitboost5.predtest <- as.factor(predict(fitboost5,newdata = test, n.trees = fitboost5.optTrees))
postResample(pred = as.numeric(fitboost5.predtest), obs = as.numeric(test$RV))


set.seed(123)
fitboost1.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost1.optTrees, 
                  distribution = "gaussian", interaction.depth = 1,
                  shrinkage = 0.001, train.fraction = .6)
fitboost2.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost2.optTrees, 
                  distribution = "gaussian", interaction.depth = 2,
                  shrinkage = 0.001, train.fraction = .6)
fitboost3.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost3.optTrees, 
                  distribution = "gaussian", interaction.depth = 3,
                  shrinkage = 0.001, train.fraction = .6)
fitboost4.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost4.optTrees, 
                  distribution = "gaussian", interaction.depth = 4,
                  shrinkage = 0.001, train.fraction = .6)
fitboost5.refit <- gbm( fmclassgbm, data = train, n.trees = fitboost5.optTrees, 
                  distribution = "gaussian", interaction.depth = 5,
                  shrinkage = 0.001, train.fraction = .6)

# checking accuracy in the test set
fitboost1.refit.predtest <- as.factor(predict(fitboost1.refit,newdata = test, n.trees = fitboost1.optTrees))
cm <-postResample(pred = as.numeric(fitboost1.refit.predtest), obs = as.numeric(test$RV))
test.errors$boost1 <- 1 - cm[2]
fitboost2.refit.predtest <- as.factor(predict(fitboost2.refit,newdata = test, n.trees = fitboost2.optTrees))
cm <-postResample(pred = as.numeric(fitboost2.refit.predtest), obs = as.numeric(test$RV))
test.errors$boost2 <- 1 - cm[2]
fitboost3.refit.predtest <- as.factor(predict(fitboost3.refit,newdata = test, n.trees = fitboost3.optTrees))
cm <-postResample(pred = as.numeric(fitboost3.refit.predtest), obs = as.numeric(test$RV))
test.errors$boost3 <- 1 - cm[2]
fitboost4.refit.predtest <- as.factor(predict(fitboost4.refit,newdata = test, n.trees = fitboost4.optTrees))
cm <-postResample(pred = as.numeric(fitboost4.refit.predtest), obs = as.numeric(test$RV))
test.errors$boost4 <- 1 - cm[2]
fitboost5.refit.predtest <- as.factor(predict(fitboost5.refit,newdata = test, n.trees = fitboost5.optTrees))
cm <-postResample(pred = as.numeric(fitboost5.refit.predtest), obs = as.numeric(test$RV))
test.errors$boost5 <- 1 - cm[2]

# comparison of a single pruned tree, bagging, random forest and the boosting models
# Boosting model with with a level of interaction of two is the best one
op <- par(cex = 1)
barplot(as.matrix(test.errors))

