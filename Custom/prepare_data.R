train.1 <- read.csv('custom_features_train.csv', header=FALSE)
test.1 <- read.csv('custom_features_test.csv', header=FALSE)

train.y.text <- read.csv('train_emails_classes_100.txt', header=FALSE)[, 1]
test.y.text <- read.csv('test_emails_classes_0.txt', header=FALSE)[, 1]

train.y <- rep(NA, length(train.y.text))
test.y <- rep(NA, length(test.y.text))
train.y[which(train.y.text == 'NotSpam')] <- 0
train.y[which(train.y.text == 'Spam')] <- 1
test.y[which(test.y.text == 'NotSpam')] <- 0
test.y[which(test.y.text == 'Spam')] <- 1

tr.mean <- apply(train.1, FUN=mean, 2)
ts.mean <- apply(test.1, FUN=mean, 2)

# column 42 of train and test have Inf values, replace them with column-wise mean
inf.inds.tr <- which(train.1[, 42] == Inf)
train.1[inf.inds.tr, 42] <- mean(train.1[-inf.inds.tr, 42])
inf.inds.ts <- which(test.1[, 42] == Inf)
test.1[inf.inds.ts, 42] <- mean(test.1[-inf.inds.ts, 42])

# remove zero-variance columns
tr.var <- apply(train.1, FUN=var, 2)
ts.var <- apply(test.1, FUN=var, 2)
all(which(tr.var == 0) == which(ts.var == 0)) # same columns
train.2 <- train.1[, -which(tr.var == 0)]
test.2 <- test.1[, -which(ts.var == 0)]

# split train into train and validation (no cross-validation)
VALIDATION_FRAC <- 0.2
valid.inds <- sample(seq(1, nrow(train.2)), nrow(train.2) * VALIDATION_FRAC)

X.train <- train.2[-valid.inds, ]
y.train <- train.y[-valid.inds]

X.valid <- train.2[valid.inds, ]
y.valid <- train.y[valid.inds]

save(X.train, y.train, X.valid, y.valid, valid.inds,
    file='train_validation_split.RData')

X.test <- test.2
y.test <- test.y
save(X.test, y.test, file='test_split.RData')