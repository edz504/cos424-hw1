library(e1071)

# custom_train_valid.R + custom_valid_valid.R showed that the SVM with linear kernel got the highest validation accuracy, so we train that on all of the training data
load('train_validation_split.RData')

X <- rbind(X.train, X.valid)
y <- c(y.train, y.valid)

# scale the training set by the training features
X.s <- scale(X)

# remove extra variables for memory limitations
rm(X)
rm(X.train)
rm(X.valid)
rm(y.train)
rm(y.valid)
rm(valid.inds)

##### Train SVM (linear) on full training set
start.time <- Sys.time()
svm_model_linear_full <- svm(X.s, 
    y, 
    type='C', 
    kernel='linear', 
    probability=TRUE)
end.time <- Sys.time()
print(end.time - start.time)
save(svm_model_linear_full, file="svm_model_linear_full_custom.RData")
