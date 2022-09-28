library(neuralnet)
library(nnet) ## nnet () not used here as it does not support multilayer networks
library(caret)
library(e1071)

# Read accidents data
titanic <- read.csv("Desktop/kaggle/train.csv")
titanic1 <-read.csv("Desktop/kaggle/test.csv")
titanic<-titanic[,-c(1,4,9,11)]
titanic1<-titanic1[,-c(3,8,10)]
titanic$Age[is.na(titanic$Age)] <- mean(titanic$Age, na.rm = TRUE)
titanic1$Age[is.na(titanic1$Age)] <- mean(titanic1$Age, na.rm = TRUE)
titanic <- titanic[-c(62,830), ]



dummy <- dummyVars("~Sex+Embarked",titanic)
titanica_Add <- data.frame(predict(dummy,titanic))
titanic <- cbind(titanic, titanica_Add)

dummy <- dummyVars("~.",titanic1)
titanic1 <- data.frame(predict(dummy,titanic1))

titanic$Survived <- as.character(titanic$Survived)
titanic$'1' <- titanic$Survived == '1'
titanic$'0' <- titanic$Survived == '0'

process <- preProcess(titanic[,c(2,4,5,6,7)], method=c("range"))
titanic <- predict(process, titanic)

process1 <- preProcess(titanic1[,c(2,5,6,7,8)], method=c("range"))
titanic1 <- predict(process1, titanic1)

# Partition the Data
set.seed(123)
train.index <- sample(c(1:dim(titanic)[1]), dim(titanic)[1]*0.7)  
train.df <- titanic[train.index, ]
valid.df <- titanic[-train.index, ]

# --- If above gives error, change the value in set.seed () 
## run nn with 2 hidden nodes
# use hidden - with a vector of integers specifying number of hidden nodes in each layer

nn <- neuralnet(train.df$'1'+train.df$'0' ~ 
                Pclass+Sexfemale+Sexmale+Age+SibSp+Parch
                +Fare+EmbarkedC+EmbarkedQ+EmbarkedS, data = train.df,
                linear.output = F,hidden = 3)    

# Plot the Neural Network 
plot(nn, rep = "best") # Review the Neural Network

# Print the  Weights
#nn$weights
# Display predictions
#prediction(nn)

# Now, use this neural net to predict -- and see how well it fits the training and validation data
# Note that traindata has several dummy variables
predict <- compute(nn, data.frame(train.df))
predicted.class = apply(predict$net.result, 1, which.max) - 1
confusionMatrix(as.factor(ifelse(predicted.class =="1", "0", "1")), as.factor(train.df$Survived))

# Now, predict using validation dataset
predict <- compute(nn, data.frame(valid.df))
predicted.class = apply(predict$net.result, 1, which.max) - 1
confusionMatrix(as.factor(ifelse(predicted.class =="1", "0", "1")), as.factor(valid.df$Survived))

#upload result
predict1 <- compute(nn, data.frame(titanic1))
df1<- data.frame(titanic1$PassengerId,predict1)
names(df1)<-c("PassengerId","Survived")
write.csv(df,file="Rupload4.csv",row.names = F)

upload <- predict(nn,titanic1)[1:418]
df<- data.frame(titanic1$PassengerId,upload)
names(df)<-c("PassengerId","Survived")
write.csv(df,file="Rupload5.csv",row.names = F)


rm(list=ls())
