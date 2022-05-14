

    From: A. Alhazred alhazr3d@kingsporthill.com

    To: You you@kingsporthill.com

    Date: Fri 22/04/2022 10:00

    Subject: RV modeling - Urgent

     Hi,

    Welcome to the firm once again. My team are working on a new trading strategy for equity indices which involves arbitraging between implied and realized volatility (RV). If we have a good forecast of tomorrow’s RV, we can improve the profitability of our trading strategy. I remember from your interview that you took a course in machine learning at business school and you can code in R. Therefore, I would like you and your team to build some machine learning models in R to forecast RV. You can download state-of-the-art RV estimates for different equity indices from the \url{https://realized.oxford-man.ox.ac.uk/}. Your team will work on the [INSERT SELECTED INDEX] stock market index. You should focus on the 5-minute RV estimate (rv5 in the data set).

    RV is a special type of non-parametric volatility estimator that uses high-frequency data. Each day’s volatility measure depends only on data from that day. If you need more information, please see the documentation part of the site. Your benchmark model should be the Heterogeneous Autoregressive Realized Volatility (HAR-RV). The main reference is Corsi (2009), which you can find in our shared drive. The model is quite simple. Basically, it’s just a linear regression of the RV at time t on the lagged 1-day, 5-day and 22-day moving averages of RV. See equation (8) in the paper.

    I would like your team to create alternative models to try to improve on the HAR model. First you should try creating some additional predictors. Some ideas:

        Lagged moving averages of RV using different windows (other than 1, 5, and 22 days)

        Lagged moving averages of daily returns (calculate returns using the $close\_price$ variable)

        Lagged moving averages of the intraday returns (see the variable $open\_to\_close$ in the data set)

    You should test linear models as well as machine learning models using these predictors. Some suggestions of models to try (of course, feel free to try other models):

        The original HAR-RV benchmark model

        Extended HAR-RV model (add additional lags/variables)

        Extended HAR-RV model using regularization (ridge and/or lasso)

        Bagging/Random Forests

        Boosting

    I need you to test these models and provide an opinion if any of them can beat the simple HAR in terms of forecasting RV out of sample. If so, we will consider using it for our trading strategy. You should estimate the models using a long sample (let’s say from January 2000 to December 2019). Be mindful to use only this period for model development and to select any parameters. Once you’re happy with the models you have created (and only then), forecast RV in the remaining data. I would like to see some tables summarizing statistics (R2, mean squared error) in the training and test samples for all models. I also want to see some nice graphs comparing the actual RV with the predicted RV in each period. Finally, based on your results, I would like a recommendation of the best model to use, and whether it’s worth it to use machine learning for the purposes of RV forecasting. Please be careful with the usual problems in machine learning, such as overfitting and look-ahead bias (that is, the variables used to predict tomorrow’s RV must be observable today, so in general you need to lag them). I have added to our shared drive an initial R script that creates the basic HAR model to help you get started.

    A. Alhazred

    Head of Systematic Equity Strategies
