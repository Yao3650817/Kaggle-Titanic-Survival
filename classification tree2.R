#import library and data
library(rpart); library(rpart.plot)
library(forecast); library(caret); library(e1071)
titanic <- read.csv("Desktop/kaggle/train.csv")
titanic2 <- read.csv("Desktop/kaggle/test.csv")

# Drop columns.
titanic <- titanic[ , -c(4,9,11)]  
titanic <- titanic[ , -c(1)]
titanic2 <- titanic2[ , -c(3,8,10)] 

#split data
set.seed(1)  
train.index <- sample(c(1:dim(titanic)[1]), dim(titanic)[1]*0.7)  
train.df <- titanic[train.index, ]
valid.df <- titanic[-train.index, ]

#train data
#cv.ct <- rpart(Survived ~ ., data = train.df, method = "class",    ####  
#               cp = 0, minsplit = 5, xval = 5)   
cv.ct <- rpart(Survived ~ ., data = titanic, method = "class", cp = 0.00001, minsplit = 1, xval = 5) 
#choose the best model
printcp(cv.ct)

pruned.ct <- prune(cv.ct, cp = 0.00292398)

#pruned.ct <- prune(cv.ct,                                          ###
                   #cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])


length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])

#test set accuracy
pruned.ct.pred.valid <- predict(pruned.ct,valid.df,type = "class")
confusionMatrix(pruned.ct.pred.valid, as.factor(valid.df$Survived))

#upload result
upload <- predict(pruned.ct,titanic2)[1:418]
df<- data.frame(titanic2$PassengerId,upload)
names(df)<-c("PassengerId","Survived")
write.csv(df,file="Rupload1.csv",row.names = F)

rm(list=ls())