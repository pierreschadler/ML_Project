#  Bagging Start
bag = randomForest(RV_target~., data = train_data, 
                   mtry = 68,
                   maxnodes=200, 
                   importance = TRUE, 
                   ntree= 1000, 
                   keep.forest = TRUE)
#Check the importance of each parameter for the model to explain RV
importance(bag)

varImpPlot(bag)

train_data$pred_BAG_RV <- as.numeric(predict(bag))

train_data = na.omit(train_data)

plot(train_data["2019", 
                c("RV", "pred_BAG_RV")], 
     col=c("black", "red"),
     lwd = c(1,1), 
     main = "Actual vs predicted RV", 
     legend.loc = "topleft")
RSQUARE(test_data$RV_target, test_data$predicted_RV)

MSE(test_data$RV_target, test_data$predicted_RV)

summary(rv_rf)

print(rv_rf)

print(varImp(rv_rf))