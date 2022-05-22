if (!require("randomForest")) {
  install.packages("randomForest")
  library(randomForest)
}
if (!require("data.table")) {
  install.packages("data.table")
  library(data.table)
}
if (!require("xts")) {
  install.packages("xts")
  library(xts)
}
if (!require("dplyr")) {
  install.packages("dplyr")
  library(dplyr)
}
if (!require("PerformanceAnalytics")) {
  install.packages("PerformanceAnalytics")
  library(PerformanceAnalytics)
}
if (!require("caret")) {
  install.packages("caret")
  library(caret)
}
if (!require("glmnet")) {
  install.packages("glmnet")
  library(glmnet)
}
if (!require("gbm")) {
  install.packages("gbm")
  library(gbm)
}
if (!require("rpart")) {
  install.packages("rpart")
  library(rpart)
}
if (!require("ROCR")) {
  install.packages("ROCR")
  library(ROCR)
}
if (!require("car")) {
  install.packages("car")
  library(car)
}
if (!require("plotmo")) {
  install.packages("plotmo")
  library(plotmo)
}
if (!require("mlbench")) {
  install.packages("mlbench")
  library(mlbench)
}
if (!require("e1071")) {
  install.packages("e1071")
  library(e1071)
}
rm(list=ls())

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

# create an object to store the test_data errors of various models
test_data.errors <- data.frame(har_rv = 0, har_rv_add = 0, linear_ols = 0, ridge = 0, lasso = 0,
                               bagging = 0, randomforest = 0, boost1 = 0,boost2 = 0, boost3 = 0,
                               boost4 = 0, boost5 = 0)

# format Date 
rv_subset$Date <- as.Date(rv_subset$Date)

rv_subset <- as.xts(rv_subset)
rv_subset$RV <- rv_subset$RV^.5 * sqrt(252)

par(mfrow = c(3, 1) )
plot(rv_subset$open_to_close)
plot(rv_subset$RV)

#lagging the data points

rv_subset$Date <- stats::lag(rv_subset$Date, -1)
rv_subset$RV_target <- rv_subset$RV
rv_subset$RV_target <- stats::lag(rv_subset$RV_target, -1)
rv_subset <- na.omit(rv_subset)

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

# Computing daily returns from close prices
rv_subset$RET <- Return.calculate(rv_subset$close_price)

# Computing daily returns moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_RET1 <- frollmean(rv_subset$RET, 1)
rv_subset$MA_RET5 <- frollmean(rv_subset$RET, 5)
rv_subset$MA_RET10 <- frollmean(rv_subset$RET, 10)
rv_subset$MA_RET22 <- frollmean(rv_subset$RET, 22)
rv_subset$MA_RET50 <- frollmean(rv_subset$RET, 50)

# Computing Open to Close Return moving averages 1, 5, 10, 22 and 50 days
rv_subset$MA_INTDAYRET1 <- frollmean(rv_subset$open_to_close, 1)
rv_subset$MA_INTDAYRET5 <- frollmean(rv_subset$open_to_close, 5)
rv_subset$MA_INTDAYRET10 <- frollmean(rv_subset$open_to_close, 10)
rv_subset$MA_INTDAYRET22 <- frollmean(rv_subset$open_to_close, 22)
rv_subset$MA_INTDAYRET50 <- frollmean(rv_subset$open_to_close, 50)


rv_subset <- na.omit(rv_subset)
train_data <- rv_subset["/2019", ]
test_data <-  rv_subset["2020/", ]

# create functions to calculate R2 and RMSE
MSE = function(y_actual, y_predict){
  sqrt(mean((y_actual-y_predict)^2))
}

# recall that R2 is given by 1 - SSR/SST
RSQUARE = function(y_actual,y_predict){
  1 - sum( (y_actual-y_predict)^2)/sum( (y_actual-mean(y_actual))^2)
}

#train_data <- na.omit(train_data)

#----------------- HAR-RV (basic) begins ------------------------

HAR_RV <- lm(RV_target ~ MA_RV1 + MA_RV5 + MA_RV22,
             data = train_data)
summary(HAR_RV)

test_data$pred_HAR_RV <- as.numeric(predict(HAR_RV, test_data))

plot(test_data["2021", 
                c("RV_target", "pred_HAR_RV")], 
     col=c("black", "red"),
     lwd = c(1,1), 
     main = "Actual vs predicted RV", 
     legend.loc = "topleft")

MSE(test_data$RV_target, test_data$pred_HAR_RV)

RSQUARE(test_data$RV_target, test_data$pred_HAR_RV)

test_data.errors$har_rv <- RSQUARE(test_data$RV_target, test_data$pred_HAR_RV)

#----------------- HAR-RV (basic) ends ------------------------



#----------------- HAR-RV (additional) begins ------------------------

HAR_RV_add <- lm(RV_target ~ close_price + open_to_close + bv + medrv + rk_parzen + rk_th2 + 
               rk_twoscale + rsv + rv10 + rv5_ss + MA_RV1 + MA_RV5 + MA_RV10 + 
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
             data=train_data)

summary(HAR_RV_add)

test_data$pred_HAR_RV_add <- as.numeric(predict(HAR_RV_add, test_data))

MSE(test_data$RV_target, test_data$pred_HAR_RV_add)

RSQUARE(test_data$RV_target, test_data$pred_HAR_RV_add)

test_data.errors$har_rv_add <- RSQUARE(test_data$RV_target, test_data$pred_HAR_RV_add)

#----------------- HAR-RV (additional) ends ------------------------


# ---------------- Basic Linear model OLS starts------------------
temp_fm <- as.formula(RV_target ~ + MA_RV1 + MA_RV5 + MA_RV10 + 
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

temp_ols.fit<-lm(temp_fm, data=train_data)

summary(temp_ols.fit)

fm<-as.formula(RV_target~ + MA_RV1  + MA_RV10 
               + MA_bv1 + MA_bv5 + MA_bv10   + 
                 MA_medrv1 + MA_medrv5  + MA_medrv22 + MA_medrv50 + 
                 MA_rk_parzen10 + MA_rk_parzen22 + 
                 MA_rk_th21 + MA_rk_th210  + 
                 MA_rk_th250 + MA_rk_twoscale1  + MA_rk_twoscale10 + 
                 MA_rsv5 + MA_rsv10 + 
                 MA_rsv22 +  MA_rv101 + MA_rv105  + MA_rv51 + MA_rv55 + 
                 MA_rv510  + MA_RET1 + 
                 MA_RET50 + MA_INTDAYRET1 + MA_INTDAYRET50)

ols.fit<-lm(fm, data=train_data)
summary(ols.fit)

test_data$predicted_RV1 <- as.numeric(predict(temp_ols.fit, test_data))
test_data$predicted_RV2 <- as.numeric(predict(ols.fit, test_data))

RSQUARE(test_data$RV_target, test_data$predicted_RV1)

MSE(test_data$RV_target, test_data$predicted_RV1)

RSQUARE(test_data$RV_target, test_data$predicted_RV2)

MSE(test_data$RV_target, test_data$predicted_RV2)

test_data.errors$linear_ols <- RSQUARE(test_data$RV_target, test_data$predicted_RV2)

# ---------------- Basic Linear model OLS ends------------------



#---------------------- Ridge Lasso LM starts---------------

# We create the xtrain_data et ytrain_data matrix to train_data the following models
xtrain_data <- model.matrix(fm, data = train_data)
ytrain_data <- train_data$RV_target

# We are starting  by running simple lasso and ridge models without cross-validation

lasso.fit <- glmnet(xtrain_data, ytrain_data, family = "gaussian", alpha = 1, nlambda = 10000)
ridge.fit <- glmnet(xtrain_data, ytrain_data, family = "gaussian", alpha = 0, nlambda = 10000)

# plot the fit (need to maximize the graphs to see properly)
# obs: add the option label = TRUE to see all variable names, or label = x to see x variable names
par(mfrow = c(1,2))
plot_glmnet(lasso.fit, main = "LASSO", xvar = "lambda")
plot_glmnet(ridge.fit, main = "Ridge",  xvar= "lambda")

# Cross validation of the LASSO model
set.seed(123)
lasso.cvfit <- cv.glmnet(xtrain_data, ytrain_data, alpha = 1, nfolds = 100)
plot(lasso.cvfit,main = "LASSO")

# lambda which produces minimum error 
lambdaLASSO <- lasso.cvfit$lambda.min
log(lambdaLASSO)

# coefficients (note some have been shrunk to zero)
coef(lasso.cvfit, s = "lambda.min")

# Cross validation of the RIDGE model
ridge.cvfit <- cv.glmnet(xtrain_data, ytrain_data, alpha = 0, nfolds = 100)
plot(ridge.cvfit, main = "Ridge")

# lambda which produces minimum error 
lambdaRidge <- ridge.cvfit$lambda.min

# coefficients 
coef(ridge.cvfit, s = "lambda.min")

# compare OLS, lasso and ridge model coefficients
#We can see that the coeficients across the models differ a lot
coefs <- cbind(ols.fit$coefficients, 
               coef(lasso.cvfit, s = "lambda.min"),
               coef(ridge.cvfit, s = "lambda.min"))
colnames(coefs) <- c("OLS", "LASSO","Ridge")
coefs

# predict on test_data set with optimal cross-validation parameters
#Creating test_dataing data matrix
xtest_data <- model.matrix(fm, data = test_data)
ytest_data <- test_data$RV_target

# predict realized volatility in the test_data sample using each method
ols.fit.predtest_data <- predict(ols.fit,test_data)
lasso.fit.predtest_data <- predict(lasso.fit, s = lambdaLASSO,  newx = xtest_data)
ridge.fit.predtest_data <- predict(ridge.fit, s = lambdaRidge,  newx = xtest_data)

postResample(pred = as.numeric(ols.fit.predtest_data), obs = as.numeric(ytest_data))
postResample(pred = as.numeric(lasso.fit.predtest_data), obs = as.numeric(ytest_data))
postResample(pred = as.numeric(ridge.fit.predtest_data), obs = as.numeric(ytest_data))

# LASSO and Ridge seem to reduce root-mean-square error slightly
# at the cost of a slightly reduced R-squared

par(mfrow = c(1,1))
plt <- test_data$RV
plt$ols.fit <- as.numeric(ols.fit.predtest_data)
plt$lasso.fit <- as.numeric(lasso.fit.predtest_data)
plt$ridge.fit <- as.numeric(ridge.fit.predtest_data)

plot(plt["2022",], 
     col=c("black", "red", "blue","green"),
     lwd = c(1,1), 
     main = "Actual vs predicted RV", 
     legend.loc = "topleft")

test_data.errors$lasso <- RSQUARE(test_data$RV_target, lasso.fit.predtest_data)
test_data.errors$ridge <- RSQUARE(test_data$RV_target, ridge.fit.predtest_data)

#---------------------- Ridge Lasso LM ends--------------------



# -------------- Random Forest Begins ----------------------------


rv_rf <- randomForest(RV_target ~ .,
                      data = train_data,
                      importance=TRUE)

test_data$predicted_RV <- as.numeric(predict(rv_rf, test_data))

RSQUARE(test_data$RV_target, test_data$predicted_RV)

MSE(test_data$RV_target, test_data$predicted_RV)

summary(rv_rf)

print(rv_rf)

print(varImp(rv_rf))

plot(test_data[, c("RV_target", "predicted_RV")], 
     col=c("black", "red"),
     lwd = c(1,1), 
     main = "Actual vs predicted RV", 
     legend.loc = "topleft")

test_data.errors$randomforest = RSQUARE(test_data$RV_target, test_data$predicted_RV)
# ---------------- Random Forest Ends -------------



# ---------------- Boosting Model Begins -------------
# boosting using the gbm package
# The gbm package requires a specific format for the formula, as below: 

# fmclassgbm <- as.formula(as.integer(CAC40>0)~CAC40_LAG1 + CAC40_LAG2 + 
#                            DAX_LAG1+ DAX_LAG2 + SP500_LAG1 + SP500_LAG2 + 
#                            MSCIEM_LAG1 + MSCIEM_LAG2 + EURUSD_LAG1 + EURUSD_LAG2)

fmclassgbm <- as.formula(RV_target~ + close_price + open_to_close + bv + medrv + rk_parzen + rk_th2 + 
                           rk_twoscale + rsv + rv10 + rv5_ss + MA_RV1 + MA_RV5 + MA_RV10 + 
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
fitboost1 <- gbm( fmclassgbm, data = train_data, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 1,
                  shrinkage = 0.001, train.fraction = .6)
fitboost2 <- gbm( fmclassgbm, data = train_data, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 2,
                  shrinkage = 0.001, train.fraction = .6)
fitboost3 <- gbm( fmclassgbm, data = train_data, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 3,
                  shrinkage = 0.001, train.fraction = .6)
fitboost4 <- gbm( fmclassgbm, data = train_data, n.trees = 10000, 
                  distribution = "gaussian", interaction.depth = 4,
                  shrinkage = 0.001, train.fraction = .6)
fitboost5 <- gbm( fmclassgbm, data = train_data, n.trees = 10000, 
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


# checking accuracy in the train_dataing set - commands for the first boosting model
fitboost1.predtrain_data <- as.factor(predict(fitboost1,newdata = train_data, n.trees = fitboost1.optTrees))
postResample(pred = as.numeric(fitboost1.predtrain_data), obs = as.numeric(train_data$RV_target))
fitboost2.predtrain_data <- as.factor(predict(fitboost2,newdata = train_data, n.trees = fitboost2.optTrees))
postResample(pred = as.numeric(fitboost2.predtrain_data), obs = as.numeric(train_data$RV_target))
fitboost3.predtrain_data <- as.factor(predict(fitboost3,newdata = train_data, n.trees = fitboost3.optTrees))
postResample(pred = as.numeric(fitboost3.predtrain_data), obs = as.numeric(train_data$RV_target))
fitboost4.predtrain_data <- as.factor(predict(fitboost4,newdata = train_data, n.trees = fitboost4.optTrees))
postResample(pred = as.numeric(fitboost4.predtrain_data), obs = as.numeric(train_data$RV_target))
fitboost5.predtrain_data <- as.factor(predict(fitboost5,newdata = train_data, n.trees = fitboost5.optTrees))
postResample(pred = as.numeric(fitboost5.predtrain_data), obs = as.numeric(train_data$RV_target))


# checking accuracy in the test_data set
fitboost1.predtest_data <- as.factor(predict(fitboost1,newdata = test_data, n.trees = fitboost1.optTrees))
postResample(pred = as.numeric(fitboost1.predtest_data), obs = as.numeric(test_data$RV_target))
fitboost2.predtest_data <- as.factor(predict(fitboost2,newdata = test_data, n.trees = fitboost2.optTrees))
postResample(pred = as.numeric(fitboost2.predtest_data), obs = as.numeric(test_data$RV_target))
fitboost3.predtest_data <- as.factor(predict(fitboost3,newdata = test_data, n.trees = fitboost3.optTrees))
postResample(pred = as.numeric(fitboost3.predtest_data), obs = as.numeric(test_data$RV_target))
fitboost4.predtest_data <- as.factor(predict(fitboost4,newdata = test_data, n.trees = fitboost4.optTrees))
postResample(pred = as.numeric(fitboost4.predtest_data), obs = as.numeric(test_data$RV_target))
fitboost5.predtest_data <- as.factor(predict(fitboost5,newdata = test_data, n.trees = fitboost5.optTrees))
postResample(pred = as.numeric(fitboost5.predtest_data), obs = as.numeric(test_data$RV_target))


set.seed(123)
fitboost1.refit <- gbm( fmclassgbm, data = train_data, n.trees = fitboost1.optTrees, 
                        distribution = "gaussian", interaction.depth = 1,
                        shrinkage = 0.001, train.fraction = .6)
fitboost2.refit <- gbm( fmclassgbm, data = train_data, n.trees = fitboost2.optTrees, 
                        distribution = "gaussian", interaction.depth = 2,
                        shrinkage = 0.001, train.fraction = .6)
fitboost3.refit <- gbm( fmclassgbm, data = train_data, n.trees = fitboost3.optTrees, 
                        distribution = "gaussian", interaction.depth = 3,
                        shrinkage = 0.001, train.fraction = .6)
fitboost4.refit <- gbm( fmclassgbm, data = train_data, n.trees = fitboost4.optTrees, 
                        distribution = "gaussian", interaction.depth = 4,
                        shrinkage = 0.001, train.fraction = .6)
fitboost5.refit <- gbm( fmclassgbm, data = train_data, n.trees = fitboost5.optTrees, 
                        distribution = "gaussian", interaction.depth = 5,
                        shrinkage = 0.001, train.fraction = .6)

# checking accuracy in the test_data set
fitboost1.refit.predtest_data <- as.factor(predict(fitboost1.refit,newdata = test_data, n.trees = fitboost1.optTrees))
cm <-postResample(pred = as.numeric(fitboost1.refit.predtest_data), obs = as.numeric(test_data$RV_target))
test_data.errors$boost1 <- 1 - cm[2]
fitboost2.refit.predtest_data <- as.factor(predict(fitboost2.refit,newdata = test_data, n.trees = fitboost2.optTrees))
cm <-postResample(pred = as.numeric(fitboost2.refit.predtest_data), obs = as.numeric(test_data$RV_target))
test_data.errors$boost2 <- 1 - cm[2]
fitboost3.refit.predtest_data <- as.factor(predict(fitboost3.refit,newdata = test_data, n.trees = fitboost3.optTrees))
cm <-postResample(pred = as.numeric(fitboost3.refit.predtest_data), obs = as.numeric(test_data$RV_target))
test_data.errors$boost3 <- 1 - cm[2]
fitboost4.refit.predtest_data <- as.factor(predict(fitboost4.refit,newdata = test_data, n.trees = fitboost4.optTrees))
cm <-postResample(pred = as.numeric(fitboost4.refit.predtest_data), obs = as.numeric(test_data$RV_target))
test_data.errors$boost4 <- 1 - cm[2]
fitboost5.refit.predtest_data <- as.factor(predict(fitboost5.refit,newdata = test_data, n.trees = fitboost5.optTrees))
cm <-postResample(pred = as.numeric(fitboost5.refit.predtest_data), obs = as.numeric(test_data$RV_target))
test_data.errors$boost5 <- 1 - cm[2]

# --------------  Boosting Model Ends --------------




# --------------  Bagging Model Begins -----------------
bagging_rv= randomForest(RV_target~., data = train_data, 
                         mtry = 10,
                         nodesize = 10,
                         maxnodes=25, 
                         importance = TRUE, 
                         ntree= 1000, 
                         keep.forest = TRUE)

test_data$predicted_bagging_RV <- as.numeric(predict(bagging_rv, test_data))

RSQUARE(test_data$RV_target, test_data$predicted_bagging_RV)

MSE(test_data$RV_target, test_data$predicted_bagging_RV)

summary(bagging_rv)

print(bagging_rv)

print(varImp(bagging_rv))

plot(test_data[, c("RV_target", "predicted_bagging_RV")], 
     col=c("black", "red"),
     lwd = c(1,1), 
     main = "Actual vs predicted RV", 
     legend.loc = "topleft")

test_data.errors$bagging = RSQUARE(test_data$RV_target, test_data$predicted_bagging_RV)
# ---------------- Bagging Model Ends -------------

# comparison of a single pruned tree, bagging, random forest and the boosting models
# Boosting model with with a level of interaction of two is the best one
op <- par(cex = 1)
barplot(as.matrix(test_data.errors))