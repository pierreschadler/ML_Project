#install.packages("data.table")      # easy to load large files
#install.packages("xts")             # time series objects
#install.packages("dplyr")           # illustrate dplyr and piping
#install.packages("PerformanceAnalytics")
#install.packages("randomForest")

library(randomForest)
library(data.table)      # easy to load large files
library(xts)             # time series objects
library(dplyr)           # illustrate dplyr and piping
library(PerformanceAnalytics)
library(caret)

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

# Computing MA of Close price
#rv_subset$MA_CLOSEPRICE1 <- frollmean(rv_subset$close_price, 1)
#rv_subset$MA_CLOSEPRICE5 <- frollmean(rv_subset$close_price, 5)
#rv_subset$MA_CLOSEPRICE10 <- frollmean(rv_subset$close_price, 10)
#rv_subset$MA_CLOSEPRICE22 <- frollmean(rv_subset$close_price, 22)
#rv_subset$MA_CLOSEPRICE50 <- frollmean(rv_subset$close_price, 50)

# lag the moving averages:
#rv_subset$MA_RV1 <- stats::lag(rv_subset$MA_RV1, 1)
#rv_subset$MA_RV5 <- stats::lag(rv_subset$MA_RV5, 1)
#rv_subset$MA_RV10 <- stats::lag(rv_subset$MA_RV10, 1)
#rv_subset$MA_RV22 <- stats::lag(rv_subset$MA_RV22, 1)
#rv_subset$MA_RV50 <- stats::lag(rv_subset$MA_RV50, 1)

#rv_subset$MA_RET1 <- stats::lag(rv_subset$MA_RET1, 1)
#rv_subset$MA_RET5 <- stats::lag(rv_subset$MA_RET5, 1)
#rv_subset$MA_RET10 <- stats::lag(rv_subset$MA_RET10, 1)
#rv_subset$MA_RET22 <- stats::lag(rv_subset$MA_RET22, 1)
#rv_subset$MA_RET50 <- stats::lag(rv_subset$MA_RET50, 1)

#rv_subset$MA_INTDAYRET1 <- stats::lag(rv_subset$MA_INTDAYRET1, 1)
#rv_subset$MA_INTDAYRET5 <- stats::lag(rv_subset$MA_INTDAYRET5, 1)
#rv_subset$MA_INTDAYRET10 <- stats::lag(rv_subset$MA_INTDAYRET10, 1)
#rv_subset$MA_INTDAYRET22 <- stats::lag(rv_subset$MA_INTDAYRET22, 1)
#rv_subset$MA_INTDAYRET50 <- stats::lag(rv_subset$MA_INTDAYRET50, 1)

##rv_subset$MA_bv1 <- stats::lag(rv_subset$MA_bv1, 1)
#rv_subset$MA_bv5 <- stats::lag(rv_subset$MA_bv5, 1)
#rv_subset$MA_bv10 <- stats::lag(rv_subset$MA_bv10, 1)
#rv_subset$MA_bv22 <- stats::lag(rv_subset$MA_bv22, 1)
#rv_subset$MA_bv50 <- stats::lag(rv_subset$MA_bv50, 1)

#rv_subset$MA_medrv1 <- stats::lag(rv_subset$MA_medrv1, 1)
#rv_subset$MA_medrv5 <- stats::lag(rv_subset$MA_medrv5, 1)
#rv_subset$MA_medrv10 <- stats::lag(rv_subset$MA_medrv10, 1)
#rv_subset$MA_medrv22 <- stats::lag(rv_subset$MA_medrv22, 1)
#rv_subset$MA_medrv50 <- stats::lag(rv_subset$MA_medrv50, 1)

#rv_subset$MA_rk_parzen1 <- stats::lag(rv_subset$MA_rk_parzen1, 1)
##rv_subset$MA_rk_parzen5 <- stats::lag(rv_subset$MA_rk_parzen5, 1)
#rv_subset$MA_rk_parzen10 <- stats::lag(rv_subset$MA_rk_parzen10, 1)
#rv_subset$MA_rk_parzen22 <- stats::lag(rv_subset$MA_rk_parzen22, 1)
#rv_subset$MA_rk_parzen50 <- stats::lag(rv_subset$MA_rk_parzen50, 1)

#rv_subset$MA_rk_th21 <- stats::lag(rv_subset$MA_rk_th21, 1)
#rv_subset$MA_rk_th25 <- stats::lag(rv_subset$MA_rk_th25, 1)
#rv_subset$MA_rk_th210 <- stats::lag(rv_subset$MA_rk_th210, 1)
#rv_subset$MA_rk_th222 <- stats::lag(rv_subset$MA_rk_th222, 1)
#rv_subset$MA_rk_th250 <- stats::lag(rv_subset$MA_rk_th250, 1)

#rv_subset$MA_rk_twoscale1 <- stats::lag(rv_subset$MA_rk_twoscale1, 1)
#rv_subset$MA_rk_twoscale5 <- stats::lag(rv_subset$MA_rk_twoscale5, 1)
#rv_subset$MA_rk_twoscale10 <- stats::lag(rv_subset$MA_rk_twoscale10, 1)
#rv_subset$MA_rk_twoscale22 <- stats::lag(rv_subset$MA_rk_twoscale22, 1)
#rv_subset$MA_rk_twoscale50 <- stats::lag(rv_subset$MA_rk_twoscale50, 1)

#rv_subset$MA_rsv1 <- stats::lag(rv_subset$MA_rsv1, 1)
#rv_subset$MA_rsv5 <- stats::lag(rv_subset$MA_rsv5, 1)
#rv_subset$MA_rsv10 <- stats::lag(rv_subset$MA_rsv10, 1)
#rv_subset$MA_rsv22 <- stats::lag(rv_subset$MA_rsv22, 1)
#rv_subset$MA_rsv50 <- stats::lag(rv_subset$MA_rsv50, 1)

#rv_subset$MA_rv101 <- stats::lag(rv_subset$MA_rv101, 1)
#rv_subset$MA_rv105 <- stats::lag(rv_subset$MA_rv105, 1)
#rv_subset$MA_rv1010 <- stats::lag(rv_subset$MA_rv1010, 1)
#rv_subset$MA_rv1022 <- stats::lag(rv_subset$MA_rv1022, 1)
#rv_subset$MA_rv1050 <- stats::lag(rv_subset$MA_rv1050, 1)

#rv_subset$MA_rv51 <- stats::lag(rv_subset$MA_rv51, 1)
#rv_subset$MA_rv55 <- stats::lag(rv_subset$MA_rv55, 1)
#rv_subset$MA_rv510 <- stats::lag(rv_subset$MA_rv510, 1)
#rv_subset$MA_rv522 <- stats::lag(rv_subset$MA_rv522, 1)
#rv_subset$MA_rv550 <- stats::lag(rv_subset$MA_rv550, 1)

##rv_subset$MA_CLOSEPRICE1 <- stats::lag(rv_subset$MA_CLOSEPRICE1, 1)
#rv_subset$MA_CLOSEPRICE5 <- stats::lag(rv_subset$MA_CLOSEPRICE5, 1)
#rv_subset$MA_CLOSEPRICE10 <- stats::lag(rv_subset$MA_CLOSEPRICE10, 1)
#rv_subset$MA_CLOSEPRICE22 <- stats::lag(rv_subset$MA_CLOSEPRICE22, 1)
#rv_subset$MA_CLOSEPRICE50 <- stats::lag(rv_subset$MA_CLOSEPRICE50, 1)

#rv_subset$CLOSEPRICE1 <- rv_subset$close_price - rv_subset$CLOSEPRICE1
#rv_subset$CLOSEPRICE5 <- rv_subset$close_price - rv_subset$CLOSEPRICE5
#rv_subset$CLOSEPRICE10 <- rv_subset$close_price - rv_subset$CLOSEPRICE10
#rv_subset$CLOSEPRICE22 <- rv_subset$close_price - rv_subset$CLOSEPRICE22
#rv_subset$CLOSEPRICE50 <- rv_subset$close_price - rv_subset$CLOSEPRICE50


# -------- additions ---------

#rv_subset <- subset(rv_subset, select = -close_price)
#rv_subset <- subset(rv_subset, select = -rets)

# ------- additions end ---------
rv_subset <- na.omit(rv_subset)
train_data <- rv_subset["/2019", ]
test_data <-  rv_subset["2020/", ]



#train_data <- na.omit(train_data)



# -------------- Random Forest Begins --------------


rv_rf <- randomForest(RV_target ~ .,
                      data = train_data,
                      importance=TRUE)

test_data$predicted_RV <- as.numeric(predict(rv_rf, test_data))

RSQUARE = 
  function(y_actual,y_predict){
    1 - sum( (y_actual-y_predict)^2)/sum( (y_actual-mean(y_actual))^2)
  }

MSE = function(y_actual, y_predict){
  sqrt(mean((y_actual-y_predict)^2))
}

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

# ---------------- Random Forest Ends -------------
