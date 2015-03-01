####### Evaluate accuracy and log-loss metric for RF, SVM (3 kernels), ANN on validation set and pick best
library(randomForest)
library(e1071)
library(h2o)

load('train_validation_split.RData')
# get training set's mean and variance for feature normalization
tr.mean <- apply(X.train, FUN=mean, 2)
tr.var <- apply(X.train, FUN=var, 2)
# scale the validation set by the training features
X.valid.s <- (as.matrix(X.valid) - t(replicate(nrow(X.valid), tr.mean))) %*% diag(1 / tr.var) 

ev <- data.frame(valid_acc=numeric(), log_loss=numeric())
load("rf_model_custom.RData")
load("svm_model_gauss_custom.RData")
load("svm_model_sigmoid_custom.RData")
load("svm_model_linear_custom.RData")
y.valid.pred.rf <- predict(rf_model, X.valid.s)
# calculate straight accuracy
acc.rf <- 1 - sum(abs(y.valid - round(y.valid.pred.rf))) / length(y.valid)
# calculate the 2-class log loss metric
ll.rf <- -(1 / length(y.valid)) * sum(y.valid * log(y.valid.pred.rf))
ev[1,] <- c(acc.rf, ll.rf)

y.valid.pred.svm.gauss <- predict(svm_model_gauss, X.valid.s, probability=TRUE)
probs.svm.gauss <- attr(y.valid.pred.svm.gauss, "probabilities")[, 2]
acc.svm.gauss <- 1 - sum(abs(y.valid - round(probs.svm.gauss))) / length(y.valid)
ll.svm.gauss <- -(1 / length(y.valid)) * sum(y.valid * log(probs.svm.gauss))
ev[2,] <- c(acc.svm.gauss, ll.svm.gauss)

y.valid.pred.svm.sigmoid <- predict(svm_model_sigmoid, X.valid.s, probability=TRUE)
probs.svm.sigmoid <- attr(y.valid.pred.svm.sigmoid, "probabilities")[, 2]
acc.svm.sigmoid <- 1 - sum(abs(y.valid - round(probs.svm.sigmoid))) / length(y.valid)
ll.svm.sigmoid <- -(1 / length(y.valid)) * sum(y.valid * log(probs.svm.sigmoid))
ev[3,] <- c(acc.svm.sigmoid, ll.svm.sigmoid)

y.valid.pred.svm.linear <- predict(svm_model_linear, X.valid.s, probability=TRUE)
probs.svm.linear <- attr(y.valid.pred.svm.linear, "probabilities")[, 2]
acc.svm.linear <- 1 - sum(abs(y.valid - round(probs.svm.linear))) / length(y.valid)
ll.svm.linear <- -(1 / length(y.valid)) * sum(y.valid * log(probs.svm.linear))
ev[4,] <- c(acc.svm.linear, ll.svm.linear)

localH2O <- h2o.init(ip = "localhost", 
    port = 54321, 
    startH2O = TRUE, 
    nthreads = -1,
    max_mem_size='6g')

model.path <- paste(getwd(), '/DeepLearning_84047039b6f8c36ca11d6e60657045e1',
    sep='')
model <- h2o.loadModel(localH2O, model.path)
valid_h2o <- as.h2o(localH2O, X.valid.s, key='valid')
y.valid.pred.h2o <- h2o.predict(model, valid_h2o)
probs.h2o <- as.matrix(y.valid.pred.h2o)[, ncol(y.valid.pred.h2o)]
acc.h2o <- 1 - sum(abs(y.valid - round(probs.h2o))) / length(y.valid)
ll.h2o <- -(1 / length(y.valid)) * sum(y.valid * log(probs.h2o))
ev[5,] <- c(acc.h2o, ll.h2o)

rownames(ev) <- c('RF', 'SVM_gauss', 'SVM_sigmoid', 'SVM_linear', 'DNN')
save(ev, file='ev.RData')