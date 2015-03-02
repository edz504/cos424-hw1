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

ev <- data.frame(valid_acc=numeric(), 
    log_loss=numeric(),
    precision=numeric(),
    recall=numeric(),
    f1=numeric())

load("rf_model_custom.RData")
load("svm_model_gauss_custom.RData")
load("svm_model_sigmoid_custom.RData")
load("svm_model_linear_custom.RData")
load("logistic_model_custom.RData")


y.valid.pred.rf <- predict(rf_model, X.valid.s)
# calculate straight accuracy
acc.rf <- 1 - sum(abs(y.valid - round(y.valid.pred.rf))) / length(y.valid)
# calculate the 2-class log loss metric
ll.rf <- -(1 / length(y.valid)) * sum(y.valid * log(y.valid.pred.rf))
# calculate precision, recall, f1
tp <- length(intersect(which(y.valid == 1), 
    which(round(y.valid.pred.rf) == 1)))
fp <- length(intersect(which(y.valid == 0),
    which(round(y.valid.pred.rf) == 1)))
fn <- length(intersect(which(y.valid == 1),
    which(round(y.valid.pred.rf) == 0)))
prec.rf <- tp / (tp + fp)
rec.rf <- tp / (tp + fn)
f1.rf <- 2 * (prec.rf * rec.rf) / (prec.rf + rec.rf)

ev[1,] <- c(acc.rf, ll.rf, prec.rf, rec.rf, f1.rf)

y.valid.pred.svm.gauss <- predict(svm_model_gauss, X.valid.s, probability=TRUE)
probs.svm.gauss <- attr(y.valid.pred.svm.gauss, "probabilities")[, 2]
acc.svm.gauss <- 1 - sum(abs(y.valid - round(probs.svm.gauss))) / length(y.valid)
ll.svm.gauss <- -(1 / length(y.valid)) * sum(y.valid * log(probs.svm.gauss))
tp <- length(intersect(which(y.valid == 1), 
    which(round(probs.svm.gauss) == 1)))
fp <- length(intersect(which(y.valid == 0),
    which(round(probs.svm.gauss) == 1)))
fn <- length(intersect(which(y.valid == 1),
    which(round(probs.svm.gauss) == 0)))
prec.svm.gauss <- tp / (tp + fp)
rec.svm.gauss <- tp / (tp + fn)
f1.svm.gauss <- 2 * (prec.svm.gauss * rec.svm.gauss) / (prec.svm.gauss + rec.svm.gauss)

ev[2,] <- c(acc.svm.gauss, ll.svm.gauss, prec.svm.gauss, rec.svm.gauss, f1.svm.gauss)

y.valid.pred.svm.sigmoid <- predict(svm_model_sigmoid, X.valid.s, probability=TRUE)
probs.svm.sigmoid <- attr(y.valid.pred.svm.sigmoid, "probabilities")[, 2]
acc.svm.sigmoid <- 1 - sum(abs(y.valid - round(probs.svm.sigmoid))) / length(y.valid)
ll.svm.sigmoid <- -(1 / length(y.valid)) * sum(y.valid * log(probs.svm.sigmoid))
tp <- length(intersect(which(y.valid == 1), 
    which(round(probs.svm.sigmoid) == 1)))
fp <- length(intersect(which(y.valid == 0),
    which(round(probs.svm.sigmoid) == 1)))
fn <- length(intersect(which(y.valid == 1),
    which(round(probs.svm.sigmoid) == 0)))
prec.svm.sigmoid <- tp / (tp + fp)
rec.svm.sigmoid <- tp / (tp + fn)
f1.svm.sigmoid <- 2 * (prec.svm.sigmoid * rec.svm.sigmoid) / (prec.svm.sigmoid + rec.svm.sigmoid)

ev[3,] <- c(acc.svm.sigmoid, ll.svm.sigmoid, prec.svm.sigmoid, rec.svm.sigmoid, f1.svm.sigmoid)

y.valid.pred.svm.linear <- predict(svm_model_linear, X.valid.s, probability=TRUE)
probs.svm.linear <- attr(y.valid.pred.svm.linear, "probabilities")[, 2]
acc.svm.linear <- 1 - sum(abs(y.valid - round(probs.svm.linear))) / length(y.valid)
ll.svm.linear <- -(1 / length(y.valid)) * sum(y.valid * log(probs.svm.linear))
tp <- length(intersect(which(y.valid == 1), 
    which(round(probs.svm.linear) == 1)))
fp <- length(intersect(which(y.valid == 0),
    which(round(probs.svm.linear) == 1)))
fn <- length(intersect(which(y.valid == 1),
    which(round(probs.svm.linear) == 0)))
prec.svm.linear <- tp / (tp + fp)
rec.svm.linear <- tp / (tp + fn)
f1.svm.linear <- 2 * (prec.svm.linear * rec.svm.linear) / (prec.svm.linear + rec.svm.linear)

ev[4,] <- c(acc.svm.linear, ll.svm.linear, prec.svm.linear, rec.svm.linear, f1.svm.linear)

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
tp <- length(intersect(which(y.valid == 1), 
    which(round(probs.h2o) == 1)))
fp <- length(intersect(which(y.valid == 0),
    which(round(probs.h2o) == 1)))
fn <- length(intersect(which(y.valid == 1),
    which(round(probs.h2o) == 0)))
prec.h2o <- tp / (tp + fp)
rec.h2o <- tp / (tp + fn)
f1.h2o <- 2 * (prec.h2o * rec.svm.sigmoid) / (prec.h2o + rec.h2o)
ev[5,] <- c(acc.h2o, ll.h2o, prec.h2o, rec.h2o, f1.h2o)

# fix naming issues for columns in logit
train.all <- data.frame(y=y.train, x=scale(X.train))
valid.all <- data.frame(y=y.valid, x=X.valid.s)
colnames(valid.all) <- colnames(train.all)

y.valid.pred.logistic <- predict(logistic_model, valid.all, type = "response")
acc.logistic <- 1 - sum(abs(y.valid - round(y.valid.pred.logistic))) / length(y.valid)
ll.logistic <- -(1 / length(y.valid)) * sum(y.valid * log(y.valid.pred.logistic))
tp <- length(intersect(which(y.valid == 1), 
    which(round(y.valid.pred.logistic) == 1)))
fp <- length(intersect(which(y.valid == 0),
    which(round(y.valid.pred.logistic) == 1)))
fn <- length(intersect(which(y.valid == 1),
    which(round(y.valid.pred.logistic) == 0)))
prec.logistic <- tp / (tp + fp)
rec.logistic <- tp / (tp + fn)
f1.logistic <- 2 * (prec.logistic * rec.svm.sigmoid) / (prec.logistic + rec.logistic)
ev[6,] <- c(acc.logistic, ll.logistic, prec.logistic, rec.logistic, f1.logistic)

rownames(ev) <- c('RF', 'SVM_gauss', 'SVM_sigmoid', 'SVM_linear', 'DNN', 'Logistic')
save(ev, file='ev.RData')

pred_prob <- data.frame(
    truth = y.valid,
    rf = y.valid.pred.rf,
    svm_gauss = probs.svm.gauss,
    svm_sigmoid = probs.svm.sigmoid,
    svm_linear = probs.svm.linear,
    dnn = probs.h2o,
    logistic = y.valid.pred.logistic)
write.csv(pred_prob, file='pred_prob.csv',
    row.names=FALSE)