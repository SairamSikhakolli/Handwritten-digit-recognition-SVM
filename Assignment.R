############################ Assignment #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#A classic problem in the field of pattern recognition is that of handwritten digit recognition. 
#Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. 
#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

#####################################################################################

# 2. Data Understanding: 

# Number of Instances: 60000
# Number of Attributes:  785 

#3. Data Preparation: 


#Loading Neccessary libraries


library(caret)

library(kernlab)

library(dplyr)

library(readr)

library(ggplot2)

library(gridExtra)

library(doParallel)

#Loading given train and test Data

mnist_train <- read.csv("mnist_train.csv",header = F, stringsAsFactors = F)
mnist_test <- read.csv("mnist_test.csv",header=F,stringsAsFactors = F)

#Understanding Dimensions of both the datasets

dim(mnist_train) # 60000   785
dim(mnist_test) # 10000   785

#Checking the Structure of the datasets

str(mnist_train)
str(mnist_test)

#printing first few rows of both the datasets

head(mnist_train)
head(mnist_test)

#Exploring the data

summary(mnist_train)
summary(mnist_test)

#We hve 0's in the dataset, this signifies there is no area on the pixel fr the digit.
#but we can't blindly remove them, as they part of the pixel and they uniquely determine the area

#checking missing value

sapply(mnist_train, function(x) sum(is.na(x))) #no NA's
sapply(mnist_test, function(x) sum(is.na(x)))  #no NA's

#Making our target class to factor

mnist_train$V1<-factor(mnist_train$V1)
mnist_test$V1<-factor(mnist_test$V1)

summary(mnist_train$V1) #We hve the values from 0 to 9

# Sampling 15% of the train and test data to make computation faster
  
set.seed(100)
train.indices = sample(1:nrow(mnist_train), 0.15*nrow(mnist_train))
train = mnist_train[train.indices, ]
test.indices = sample(1:nrow(mnist_test), 0.15*nrow(mnist_test))
test = mnist_test[test.indices, ]

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$V1)

#Accuracy : 0.9227 

#Using RBF Kernel
Model_RBF <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)
Model_RBF

#Hyperparameter : sigma =  1.63483744485576e-07
#cost C = 1 

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$V1)

#Accuracy : 0.9567


############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

# Making grid of "sigma" and C values. 
grid <- expand.grid(.sigma=c(1.62483744485576e-07, 1.64483744485576e-07), .C=c(0.5,1,2))

#parallel processing
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl,allowparallel=TRUE)

stopCluster(cl)

print(fit.svm)

plot(fit.svm)
#The final values used for the model were sigma = 1.644837e-07 and C = 2.

# Validating the model results on test data
evaluate_non_linear<- predict(fit.svm, test)
confusionMatrix(evaluate_non_linear, test$V1)
#Final Accuracy of the model is : 0.96  
