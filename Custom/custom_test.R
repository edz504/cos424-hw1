library(e1071)
##### Test SVM (linear) on testing set
load('train_validation_split.RData')
load('test_split.RData')
load('svm_model_linear_full_custom.RData')

# get training set's mean and variance for feature normalization
X <- rbind(X.train, X.valid)

tr.mean <- apply(X, FUN=mean, 2)
tr.var <- apply(X, FUN=var, 2)
X.test.s <- (as.matrix(X.test) - t(replicate(nrow(X.test), 
    tr.mean))) %*% diag(1 / tr.var) 

y.test.pred.svm.linear <- predict(svm_model_linear_full, X.test.s, probability=TRUE)
probs.full <- attr(y.test.pred.svm.linear, "probabilities")[, 2]

# calculate straight accuracy
acc.test <- 1 - sum(abs(y.test - round(probs.full))) / length(y.test)

# calculate the 2-class log loss metric
ll.test <- -(1 / length(y.test)) * sum(y.test * log(probs.full))