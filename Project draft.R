library(data.table)      # easy to load large files
library(xts)             # time series objects
library(dplyr)           # illustrate dplyr and piping
library(PerformanceAnalytics)

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
rv_subset$rets <- CalculateReturns(rv_subset$close_price)

par(mfrow = c(3, 1) )
plot(rv_subset$rets)
plot(rv_subset$open_to_close)
plot(rv_subset$RV)

# calculate moving averages 1, 5, and 22 days
rv_subset$MA_RV1 <- frollmean(rv_subset$RV, 1)
rv_subset$MA_RV5 <- frollmean(rv_subset$RV, 5)
rv_subset$MA_RV22 <- frollmean(rv_subset$RV, 22)

# lag the moving averages:
rv_subset$MA_RV1 <- stats::lag(rv_subset$MA_RV1, 1)
rv_subset$MA_RV5 <- stats::lag(rv_subset$MA_RV5, 1)
rv_subset$MA_RV22 <- stats::lag(rv_subset$MA_RV22, 1)

train_data <- rv_subset["/2019", ]
test_data <-  rv_subset["2020/", ]



train_data <- na.omit(train_data)


RVHAR_RV <- lm(RV ~ MA_RV1 + MA_RV5 + MA_RV22,bv,
             data = train_data)

summary(HAR_RV)

train_data$pred_HAR_RV <- as.numeric(predict(HAR_RV))



plot(train_data["2019", 
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
RSQUARE = function(y_actual,y_predict){
  1 - sum( (y_actual-y_predict)^2)/sum( (y_actual-mean(y_actual))^2)
}

MSE(train_data$RV, train_data$pred_HAR_RV)



RSQUARE(train_data$RV, train_data$pred_HAR_RV)

