load('train_validation_split.RData')

X.train.s <- scale(X.train)

##### Train multiple models
library(randomForest)
library(e1071)
library(h2o)

##### RF
start.time <- Sys.time()
rf_model <- randomForest(x=X.train.s,
    y=y.train,
    keep.forest=TRUE)
end.time <- Sys.time()
print(end.time - start.time)
save(rf_model, file="rf_model_custom.RData")
# Time difference of 13.50562 mins

##### SVM (Gaussian)
start.time <- Sys.time()
svm_model_gauss <- svm(X.train.s, 
    y.train, 
    type='C', 
    kernel='radial', 
    probability=TRUE)
end.time <- Sys.time()
print(end.time - start.time)
save(svm_model_gauss, file="svm_model_gauss_custom.RData")
# Time difference of 5.731219 mins

##### SVM (sigmoid)
start.time <- Sys.time()
svm_model_sigmoid <- svm(X.train.s, 
    y.train, 
    type='C', 
    kernel='sigmoid', 
    probability=TRUE)
end.time <- Sys.time()
print(end.time - start.time)
save(svm_model_sigmoid, file="svm_model_sigmoid_custom.RData")
# Time difference of 6.28119 mins

##### SVM (linear)
start.time <- Sys.time()
svm_model_linear <- svm(X.train.s, 
    y.train, 
    type='C', 
    kernel='linear', 
    probability=TRUE)
end.time <- Sys.time()
print(end.time - start.time)
save(svm_model_linear, file="svm_model_linear_custom.RData")
# Time difference of 22.38755 mins

##### DNN
localH2O <- h2o.init(ip = "localhost", 
    port = 54321, 
    startH2O = TRUE, 
    nthreads = -1,
    max_mem_size='6g')

# add labels into the df, and then push it into the h2o instance
train_h2o <- as.h2o(localH2O, data.frame(y=y.train, X.train.s), key='train')

# train the DNN model
start.time <- Sys.time()
model <- h2o.deeplearning(
    x = 2:63,  # column numbers for predictors
    y = 1,   # column number for label
    data = train_h2o, # data in H2O format
    activation = "TanhWithDropout", # or 'Tanh'
    input_dropout_ratio = 0.2, # % of inputs dropout
    hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
    balance_classes = TRUE, 
    hidden = c(50,50,50), # three layers of 50 nodes
    epochs = 100) # max. no. of epochs
end.time <- Sys.time()
print(end.time - start.time)
# Time difference of 2.13140 min
h2o.saveModel(model, getwd())
h2o.shutdown(localH2O)

##### Logistic
train.all <- data.frame(y=y.train, x=X.train.s)
start.time <- Sys.time()
logistic_model <- glm(y~., data=train.all, family = binomial)
end.time <- Sys.time()
print(end.time - start.time)
save(logistic_model, file="logistic_model_custom.RData")
# Time difference of 4.196407 secs
