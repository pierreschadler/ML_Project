rm(list=ls())
library(data.table)
library(rpart)          # one of the main ways to train_data classification and regression trees in R
library(caret)      
library(randomForest)   # for bagging and random forests
library(ROCR) 
library(car)
library(gbm)            # generalized boosting regression modeling
library(xts)             # time series objects
library(dplyr)           # illustrate dplyr and piping
library(PerformanceAnalytics)
library(plotmo)
library(glmnet)


#Import data and create a subset dataframe with all needed variables

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


# format Date, compute returns and basic plots to visualize data
rv_subset$Date <- as.Date(rv_subset$Date)
rv_subset <- as.xts(rv_subset)
rv_subset$RV <- rv_subset$RV^.5 * sqrt(252)
rv_subset$rets <- CalculateReturns(rv_subset$close_price)


par(mfrow = c(3, 1) )
plot(rv_subset$rets)
plot(rv_subset$open_to_close)
plot(rv_subset$RV)

# Here we will create a multitude of new factors from the data
# These factors are lagged moving average of the existing variables
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

# lagging the moving averages:
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


#Creating the train_dataing and test_dataing dataframes
train_data <- rv_subset["/2019", ]
test_data<-  rv_subset["2020/", ]
# Removes all NA values in the dataframe
train_data <- na.omit(train_data)


#Creating the formula of the model, we want to explain RV by all the sub-factors we created
temp_fm <- as.formula(RV~ + MA_RV1 + MA_RV5 + MA_RV10 + 
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

#Creating a temporary linear model
temp_ols.fit<-lm(temp_fm, data=train_data)
#We can see that we have a good adjusted R-squared of 0.927 however the majority
#of the factors are not significant in the analysis we can therefore remove them
#We can plot the factors that have a p-value superior of 0.05 that we will remove
summary(temp_ols.fit)
par(mfrow = c(1, 2) )
plot(summary(temp_ols.fit)[["coefficients"]][,4], main = "Factors with p_vales >  0.05 before cleaning ")
abline(h = 0.05)
#We can also see that we have abnormal large VIFs 
vif(temp_ols.fit)
plot(vif(temp_ols.fit), main = "Factors with VIFs >  10 before cleaning")
abline(h = 10)


#Here is a cleaned formula with only significant factors
fm<-as.formula(RV~ + MA_RV1  + MA_RV10 
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
par(mfrow = c(1, 2) )
plot(summary(ols.fit)[["coefficients"]][,4], main = "Factors with p_vales >  0.05 after cleaning")
abline(h = 0.05)
#There are still large VIFs hewever for the sake of this analysis we will continue
vif(ols.fit)
plot(vif(ols.fit), main = "Factors with VIFs >  10 after cleaning")
abline(h = 10)


#Saving the predicted values of the OLS model and plotting them agaist the obvervations
train_data$pred_ols.fit <- as.numeric(predict(ols.fit))

#Compute R2 and RMSE
postResample(pred = as.numeric(train_data$pred_ols.fit), obs = as.numeric(train_data$RV))


# We create the xtrain_data et ytrain_data matrix to train_data the following models
xtrain_data <- model.matrix(fm, data = train_data)
ytrain_data <- train_data$RV

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
ytest_data <- test_data$RV

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









