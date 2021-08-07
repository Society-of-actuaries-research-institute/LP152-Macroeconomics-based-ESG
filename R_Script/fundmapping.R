# Check if required packages have been installed. If not install them
packages <- c("rpart", "rpart.plot", "FNN", "neuralnet", "gbm", "glmnet", "caret")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}

# Load the libraries
library(rpart)
library(rpart.plot)
library(FNN)
library(neuralnet)
library(gbm)
library(glmnet)
library(caret)

# Set working directory. This needs to be changed according to your file system
setwd("C:/dsge/r6")

# Set random number generator
set.seed(123)

start_time <- Sys.time()

################################################################
# Multifactor regression #######################################
################################################################

# Read csv file that contains historical economic factor and asset returns
rawdata <- read.csv("input/inputmap.csv", header=TRUE, sep=",", dec=".")

# select X variables
Xnames <- c("R_","pi_c_","dy_","dc_","di_","dE_","dimp_","dex_","dS_","dw_")

# select Y variables
Ynames <- names(rawdata)[!names(rawdata) %in% c(Xnames,"Year","Quarter","Recession","pi_i_","pi_d_","dimp_","dex_","dE_","dS_","dw_","dy_star_","pi_star_","R_star_","aaadefault")]

# define a dataframe that will be used to store the results
modeloutput <- data.frame(y=character(),
				 RMSE_train=double(), 		#RMSE based on training data
                 R2_train=double(),   		#R-squared based on training data
                 R2Adjust_train=double(), 	#Adjusted R-squared based on training data
				 RMSE=double(),				#RMSE based on valiation data
                 R2=double(),				#R-squared based on valiation data
                 R2Adjust=double(),			#Adjusted R-squared based on validation data
				 df=integer(),				#degree of freedom
				 fvar=double(),				#total variance
				 resfvar=double(),			#residual variance
				 rvar=double(),				#recession variance
				 resrvar=double(),			#residual recession variance
				 corr=double(),				#correlation coefficient
				 rescorr=double(),			#correlation coefficient during recessions
                 Models=character(),		#model type
                 stringsAsFactors=FALSE)


lag <-2 # how many lags to be used for regressions. A lag of 2 means x, x(-1), and x(-2) will be used in the regression.
residuals <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))		#store residuals of linear regression
residualsgbm <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))		#store residuals of GBM
residualsann <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))		#store residuals of Artificial NNs
residualsridge <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))	#store residuals of Ridge regression
residualslasso <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))	#store residuals of Lasso regression
residualselastic <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))	#store residuals of Elastic Net
#set column names of these data frames that store residuals
colnames(residuals) <- Ynames											
colnames(residualsgbm) <- Ynames
colnames(residualsann) <- Ynames
colnames(residualsridge) <- Ynames
colnames(residualslasso) <- Ynames
colnames(residualselastic) <- Ynames

lambda = 0.5	#inital weight for regularization in Lasso and Ridge Regression. It is not used 

################################################################
# Model Calibration using training/validation split ############
################################################################

#You may see some warning messages populated for certain y variables and models. They may indicate some issues which can
#be caused by data scarcity. However, the model assessment is based on validation data and therefore the warnings may not
#be very meaningful for model selection.

for (y in Ynames) {

	Traindata <- rawdata[,names(rawdata) %in% c(y,Xnames,"Recession")]
	Xnames1 <- names(Traindata)[!names(Traindata) %in% c("Recession")] #get a list of variables to generate lagged variables
	
	#generate lagged variables
	if (lag > 0) {
		for (i in c(1:lag)){
			for (varname in Xnames1){
				Traindata[(i+1):nrow(Traindata),paste0(varname,i)] <- Traindata[1:(nrow(Traindata)-i),varname]
				Traindata[1:i,paste0(varname,i)] <- NA
			}
		}
	}
	
	#remove data records that have NA. If the lag equals 2, the first two records will be removed as x(-1) and x(-2) have no values
	Traindata <- na.omit(Traindata[(lag+1):nrow(Traindata),])
	
	Xnames2 <- names(Traindata)[!names(Traindata) %in% c(y,"Recession")]
	
	#generate the formula used for calibration
	f <-as.formula(paste(y,"~",paste(Xnames2,collapse="+")))
	print(f)

	#split the data into training (80%) and validation (20%)
	idx <- sample(seq(1, 2), size = nrow(Traindata), replace = TRUE, prob = c(.8, .2))
	training <- Traindata[idx==1,]
	validation <- Traindata[idx==2,]

	withCallingHandlers({
		#linear regression
		mdl <- "linear regression"
		lr<-lm(f, data=training)
		lrpredict <- predict(lr,training)
		rmse_train<-sqrt(mean((training[,y]-lrpredict)^2))
		r2_train<-1-sum((training[,y]-lrpredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		lrpredict <- predict(lr,validation)
		rmse<-sqrt(mean((validation[,y]-lrpredict)^2))
		r2<-1-sum((validation[,y]-lrpredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		#aic<-AIC(lr)
		#bic<-BIC(lr)
		df<-nrow(training)-length(Xnames2)-1
		lrpredict <- predict(lr, Traindata)
		fvar <- var(lrpredict)
		resfvar <- var(lrpredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-lrpredict)
		resrvar <- var((Traindata[,y]-lrpredict)[Traindata$Recession==1])
		corr<-cor(Traindata[,y]-lrpredict,lrpredict)
		rescorr<-cor((Traindata[,y]-lrpredict)[Traindata$Recession==1],lrpredict[Traindata$Recession==1])
		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr, "LM")
		write.csv(summary(lr)$coefficients,paste0("lr_",y,".csv"))
		residuals[,y]<-c(rep(NA,nrow(residuals)-length(lr$residuals)),lr$residuals)
		
		#ridge regression
		mdl <- "ridge regression"
		ridge<-glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,y]), alpha = 0, lambda = lambda)
		cv.out <- cv.glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,y]), alpha = 0)
		lambda <- cv.out$lambda.min
		ridgepredict <- predict(ridge, s = lambda, newx = as.matrix(training[Xnames2]))
		rmse_train<-sqrt(mean((training[,y]-ridgepredict)^2))
		r2_train<-1-sum((training[,y]-ridgepredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		ridgepredict <- predict(ridge, s = lambda, newx = as.matrix(validation[Xnames2]))
		rmse<-sqrt(mean((validation[,y]-ridgepredict)^2))
		r2<-1-sum((validation[,y]-ridgepredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		#aic<-AIC(lr)
		#bic<-BIC(lr)
		df<-nrow(training)-length(Xnames2)-1
		ridgepredict <- predict(ridge, s = lambda, newx = as.matrix(Traindata[Xnames2]))
		fvar <- var(ridgepredict)
		resfvar <- var(ridgepredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-ridgepredict)
		resrvar <- var((Traindata[,y]-ridgepredict)[Traindata$Recession==1])
		corr<-cor(Traindata[,y]-ridgepredict,ridgepredict)
		rescorr<-cor((Traindata[,y]-ridgepredict)[Traindata$Recession==1],ridgepredict[Traindata$Recession==1])
		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr, "Ridge")
		write.csv(rbind(as.matrix(ridge$a0), as.matrix(ridge$beta)),paste0("ridge_",y,".csv"))
		residual_ridge <- Traindata[,y]-ridgepredict
		residualsridge[,y]<-c(rep(NA,nrow(residualsridge)-length(residual_ridge)),residual_ridge)

		#lasso regression
		mdl <- "lasso"
		lasso<-glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,y]), alpha = 1, lambda = lambda)
		cv.out <- cv.glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,y]), alpha = 1)
		lambda <- cv.out$lambda.min
		lassopredict <- predict(lasso, s = lambda, newx = as.matrix(training[Xnames2]))
		rmse_train<-sqrt(mean((training[,y]-lassopredict)^2))
		r2_train<-1-sum((training[,y]-lassopredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		lassopredict <- predict(lasso, s = lambda, newx = as.matrix(validation[Xnames2]))
		rmse<-sqrt(mean((validation[,y]-lassopredict)^2))
		r2<-1-sum((validation[,y]-lassopredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		#aic<-AIC(lr)
		#bic<-BIC(lr)
		df<-nrow(training)-length(Xnames2)-1
		lassopredict <- predict(lasso, s = lambda, newx = as.matrix(Traindata[Xnames2]))
		fvar <- var(lassopredict)
		resfvar <- var(lassopredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-lassopredict)
		resrvar <- var((Traindata[,y]-lassopredict)[Traindata$Recession==1])
		corr<-cor(Traindata[,y]-lassopredict,lassopredict)
		rescorr<-cor((Traindata[,y]-lassopredict)[Traindata$Recession==1],lassopredict[Traindata$Recession==1])
		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr, "lasso")
		write.csv(rbind(as.matrix(lasso$a0), as.matrix(lasso$beta)),paste0("lasso_",y,".csv"))
		residual_lasso <- Traindata[,y]-lassopredict
		residualslasso[,y]<-c(rep(NA,nrow(residualslasso)-length(residual_lasso)),residual_lasso)

		#elastic net
		mdl <- "elastic net"
		elastic<-train(f, data = training, method = "glmnet", trControl = trainControl("cv", number = 10), tuneLength = 10)
		alpha <- elastic$bestTune[1]
		lambda <- elastic$bestTune[2]
		elastic<-glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,y]), alpha = alpha, lambda = lambda)
		elasticpredict <- predict(elastic, newx = as.matrix(training[Xnames2]))
		rmse_train<-sqrt(mean((training[,y]-elasticpredict)^2))
		r2_train<-1-sum((training[,y]-elasticpredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		elasticpredict <- predict(elastic, s = lambda, newx = as.matrix(validation[Xnames2]))
		rmse<-sqrt(mean((validation[,y]-elasticpredict)^2))
		r2<-1-sum((validation[,y]-elasticpredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		#aic<-AIC(lr)
		#bic<-BIC(lr)
		df<-nrow(training)-length(Xnames2)-1
		elasticpredict <- predict(elastic, s = lambda, newx = as.matrix(Traindata[Xnames2]))
		fvar <- var(elasticpredict)
		resfvar <- var(elasticpredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-elasticpredict)
		resrvar <- var((Traindata[,y]-elasticpredict)[Traindata$Recession==1])
		corr<-cor(Traindata[,y]-elasticpredict,elasticpredict)
		rescorr<-cor((Traindata[,y]-elasticpredict)[Traindata$Recession==1],elasticpredict[Traindata$Recession==1])
		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr, "elastic")
		write.csv(rbind(as.matrix(elastic$a0), as.matrix(elastic$beta)),paste0("elastic_",y,".csv"))
		residual_elastic <- Traindata[,y]-elasticpredict
		residualselastic[,y]<-c(rep(NA,nrow(residualselastic)-length(residual_elastic)),residual_elastic)
		
		#cart
		mdl <- "CART"
		cart = rpart(f, data = training, cp = 10^(-3),minsplit = 10)
		cartpredict <- predict(cart, training)
		rmse_train<-sqrt(mean((training[,y]-cartpredict)^2))
		r2_train<-1-sum((training[,y]-cartpredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*(length(unique(cart$frame$var))-1)/(nrow(training)-(length(unique(cart$frame$var))-1)-1)
		cartpredict <- predict(cart, validation)
		rmse<-sqrt(mean((validation[,y]-cartpredict)^2))
		r2<-1-sum((validation[,y]-cartpredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*(length(unique(cart$frame$var))-1)/(nrow(training)-(length(unique(cart$frame$var))-1)-1)
		#rpart.plot(cart)
		df<-nrow(training)-(length(unique(cart$frame$var))-1)-1
		cartpredict <- predict(cart, Traindata)
		fvar <- var(cartpredict)
		resfvar <- var(cartpredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-cartpredict)
		resrvar <- var((Traindata[,y]-cartpredict)[Traindata$Recession==1])
		corr<-cor((Traindata[,y]-cartpredict),cartpredict)
		rescorr<-cor((Traindata[,y]-cartpredict)[Traindata$Recession==1],cartpredict[Traindata$Recession==1])

		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr, "CART")

		#knn
		mdl <- "KNN"
		knnreg <- knn.reg(train = training[,names(validation) %in% Xnames2] , test=training[,names(training) %in% Xnames2], y=training[,y], k=5, algorithm = "kd_tree")
		rmse_train<-sqrt(mean((training[,y]-knnreg$pred)^2))
		r2_train<-1-sum((training[,y]-knnreg$pred)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		knnreg <- knn.reg(train = training[,names(validation) %in% Xnames2] , test=validation[,names(validation) %in% Xnames2], y=training[,y], k=5, algorithm = "kd_tree")
		rmse<-sqrt(mean((validation[,y]-knnreg$pred)^2))
		r2<-1-sum((validation[,y]-knnreg$pred)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		df<-nrow(training)-length(Xnames2)-1
		knnreg <- knn.reg(train = training[,names(validation) %in% Xnames2] , test=Traindata[,names(Traindata) %in% Xnames2], y=training[,y], k=5, algorithm = "kd_tree")
		fvar <- var(knnreg$pred)
		resfvar <- var(knnreg$pred[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-knnreg$pred)
		resrvar <- var((Traindata[,y]-knnreg$pred)[Traindata$Recession==1])
		corr<-cor((Traindata[,y]-knnreg$pred),knnreg$pred)
		rescorr<-cor((Traindata[,y]-knnreg$pred)[Traindata$Recession==1],knnreg$pred[Traindata$Recession==1])

		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr,"KNN")

		#gbm
		mdl <- "GBM"
		set.seed(123) #reset random seed as gbm may use random subset
		gbmreg<-gbm(f, data=training, distribution = "gaussian", interaction.depth=6, n.minobsinnode = 2, bag.fraction=0.7, n.trees = 100)
		gbmpredict <- predict(gbmreg, training, n.trees = 50)
		rmse_train<-sqrt(mean((training[,y]-gbmpredict)^2))
		r2_train<-1-sum((training[,y]-gbmpredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		gbmpredict <- predict(gbmreg, validation, n.trees = 50)
		rmse<-sqrt(mean((validation[,y]-gbmpredict)^2))
		r2<-1-sum((validation[,y]-gbmpredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		df<-nrow(training)-length(Xnames2)-1
		gbmpredict <- predict(gbmreg, Traindata, n.trees = 50)	
		fvar <- var(gbmpredict)
		resfvar <- var(gbmpredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-gbmpredict)
		resrvar <- var((Traindata[,y]-gbmpredict)[Traindata$Recession==1])
		corr<-cor((Traindata[,y]-gbmpredict),gbmpredict)
		rescorr<-cor((Traindata[,y]-gbmpredict)[Traindata$Recession==1],gbmpredict[Traindata$Recession==1])
		residual_gbm <- Traindata[,y]-gbmpredict
		residualsgbm[,y]<-c(rep(NA,nrow(residualsgbm)-length(residual_gbm)),residual_gbm)

		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr,"gbm")

		#ann
		mdl <- "ANN"
		set.seed(123)
		if (y == "oil"){ #the oil data needs more complicated ANN
			ann <- neuralnet(f, data=data.matrix(training), hidden=c(10,10,5), linear.output=TRUE, stepmax = 2000000, threshold=0.03, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 5000)
		}else{
			ann <- neuralnet(f, data=data.matrix(training), hidden=c(10,5), linear.output=TRUE, stepmax = 2000000, threshold=0.01, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 5000)
		}

		#ann <- neuralnet(f, data=data.matrix(training), hidden=c(10,5), linear.output=TRUE, stepmax = 2000000, threshold=th, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 5000)
		#ann <- neuralnet(f, data=data.matrix(Traindata), hidden=c(10), linear.output=TRUE, stepmax = 200000, threshold=0.5, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 100)
		#ann <- neuralnet(f, data=data.matrix(Traindata), hidden=c(5), linear.output=TRUE, stepmax = 200000, threshold=0.5, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 100)
		#ann <- neuralnet(f, data=data.matrix(Traindata), hidden=c(5), linear.output=TRUE, stepmax = 200000, threshold=0.5, act.fct = "logistic", likelihood = TRUE, lifesign ="full", lifesign.step = 100)
		annpredict <- predict(ann,training)
		rmse_train<-sqrt(mean((training[,y]-annpredict)^2))
		r2_train<-1-sum((training[,y]-annpredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		annpredict <- predict(ann,validation)
		rmse<-sqrt(mean((validation[,y]-annpredict)^2))
		r2<-1-sum((validation[,y]-annpredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		df<-nrow(training)-length(Xnames2)-1

		annpredict <- predict(ann,Traindata)
		
		tryCatch(
		{
			fvar <- var(annpredict)
			resfvar <- var(annpredict[Traindata$Recession==1])
			rvar <- var(Traindata[,y]-annpredict)
			resrvar <- var((Traindata[,y]-annpredict)[Traindata$Recession==1])
			corr<-cor((Traindata[,y]-annpredict),annpredict)
			rescorr<-cor((Traindata[,y]-annpredict)[Traindata$Recession==1],annpredict[Traindata$Recession==1])
		},
			error = function(ex) {
	#			print("errors")
				fvar <- NA
				resfvar <- NA
				rvar <- NA
				resrvar <- NA
				corr <- NA
				rescorr <- NA
			}
		)
		residual_ann <- Traindata[,y]-annpredict
		residualsann[,y]<-c(rep(NA,nrow(residualsann)-length(residual_ann)),residual_ann)
		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr,"ANN")
	}, warning = function(ex) { 
		message(paste0("There are some issues happened for variable: ", y, "; model: ", mdl))
		message("Here's the original warning message:")
		message(ex)
		cat("\n")
	})

}

#output results and residuals
write.csv(modeloutput,paste0("modeloutput1.csv"))
write.csv(residuals,paste0("residuals1.csv"))
write.csv(residualsridge,paste0("residuals1ridge.csv"))
write.csv(residualslasso,paste0("residuals1lasso.csv"))
write.csv(residualselastic,paste0("residuals1elastic.csv"))
write.csv(residualsgbm,paste0("residuals1gbm.csv"))
write.csv(residualsann,paste0("residuals1ann.csv"))

# Calculate numbers in Table 7 in the report
modeloutput$RMSE <- as.numeric(modeloutput$RMSE)
modeloutput$RMSE_train <- as.numeric(modeloutput$RMSE_train)
modeloutput$R2 <- as.numeric(modeloutput$R2)
modeloutput$R2_train <- as.numeric(modeloutput$R2_train)
aggregate(modeloutput[,colnames(modeloutput) %in% c("RMSE_train", "R2_train", "RMSE", "R2")], by=list(modeloutput$Model), FUN = mean, na.rm=TRUE)
	
################################################################
# Elastic Net on All Data ######################################
################################################################
#After we chose Elastic Net as our best model, we apply it to all data to capture more information
#This will be used for the simulation

#define output structure as before
modeloutput <- data.frame(y=character(),
				 RMSE_train=double(),
                 R2_train=double(),
                 R2Adjust_train=double(),
				 RMSE=double(),
                 R2=double(),
                 R2Adjust=double(),
				 df=integer(),
				 fvar=double(),
				 resfvar=double(),
				 rvar=double(),
				 resrvar=double(),
				 corr=double(),
				 rescorr=double(),
                 Models=character(),
                 stringsAsFactors=FALSE)


#set lag, residuals data structure
lag <-2
residualselastic <- matrix(,nrow=nrow(rawdata)-lag,ncol=length(Ynames))
colnames(residualselastic) <- Ynames

#run elastic net model. It will generate a csv file (elastic_y.csv) that store the calibrated model for each y.
#The mapping.csv contains the values of all elastic net calibrated models.

for (y in Ynames) {
	Traindata <- rawdata[,names(rawdata) %in% c(y,Xnames,"Recession")]
	
	Xnames1 <- names(Traindata)[!names(Traindata) %in% c("Recession")]

	if (lag > 0) {
		for (i in c(1:lag)){
			for (varname in Xnames1){
				Traindata[(i+1):nrow(Traindata),paste0(varname,i)] <- Traindata[1:(nrow(Traindata)-i),varname]
				Traindata[1:i,paste0(varname,i)] <- NA
			}
		}
	}
	
	Traindata <- na.omit(Traindata[(lag+1):nrow(Traindata),])
	
	Xnames2 <- names(Traindata)[!names(Traindata) %in% c(y,"Recession")]
	
	f <-as.formula(paste(y,"~",paste(Xnames2,collapse="+")))
	
	print(f)

	training <- Traindata
	validation <- Traindata

	withCallingHandlers({
		#elastic net
		mdl <- "Elastic Net"
		elastic<-train(f, data = training, method = "glmnet", trControl = trainControl("cv", number = 10), tuneLength = 10)
		alpha <- elastic$bestTune[1]
		lambda <- elastic$bestTune[2]
		elastic<-glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,y]), alpha = alpha, lambda = lambda)
		elasticpredict <- predict(elastic, newx = as.matrix(training[Xnames2]))
		rmse_train<-sqrt(mean((training[,y]-elasticpredict)^2))
		r2_train<-1-sum((training[,y]-elasticpredict)^2)/sum((training[,y]-mean(training[,y]))^2)
		r2adjust_train <- r2_train-(1-r2_train)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		elasticpredict <- predict(elastic, s = lambda, newx = as.matrix(validation[Xnames2]))
		rmse<-sqrt(mean((validation[,y]-elasticpredict)^2))
		r2<-1-sum((validation[,y]-elasticpredict)^2)/sum((validation[,y]-mean(validation[,y]))^2)
		r2adjust <- r2-(1-r2)*length(Xnames2)/(nrow(training)-length(Xnames2)-1)
		#aic<-AIC(lr)
		#bic<-BIC(lr)
		df<-nrow(training)-length(Xnames2)-1
		elasticpredict <- predict(elastic, s = lambda, newx = as.matrix(Traindata[Xnames2]))
		fvar <- var(elasticpredict)
		resfvar <- var(elasticpredict[Traindata$Recession==1])
		rvar <- var(Traindata[,y]-elasticpredict)
		resrvar <- var((Traindata[,y]-elasticpredict)[Traindata$Recession==1])
		corr<-cor(Traindata[,y]-elasticpredict,elasticpredict)
		rescorr<-cor((Traindata[,y]-elasticpredict)[Traindata$Recession==1],elasticpredict[Traindata$Recession==1])
		modeloutput[nrow(modeloutput)+1,] <- c(y, rmse_train, r2_train, r2adjust_train, rmse, r2, r2adjust, df, fvar, resfvar, rvar, resrvar, corr,rescorr, "elastic")
		write.csv(rbind(as.matrix(elastic$a0), as.matrix(elastic$beta)),paste0("elastic_",y,".csv"))
		residual_elastic <- Traindata[,y]-elasticpredict
		residualselastic[,y]<-c(rep(NA,nrow(residualselastic)-length(residual_elastic)),residual_elastic)
	}, warning = function(ex) { 
		message(paste0("There are some issues happened for variable: ", y, "; model: ", mdl))
		message("Here's the original warning message:")
		message(ex)
		cat("\n")
	})
}

write.csv(modeloutput,paste0("modeloutput3.csv")) 
#this file contains the "tvar	trvar ivar	irvar	tcorr	rcorr" used in mapping.csv
# tvar:	 fvar				#total variance
# trvar: resfvar			#residual variance
# ivar:	 rvar				#recession variance
# irvar: resrvar			#residual recession variance
# tcorr: corr				#correlation coefficient
# rcorr: rescorr			#correlation coefficient during recessions
write.csv(residualselastic,paste0("residuals1elastic3.csv"))
	
################################################################
# predict recession ############################################
################################################################
rawdata <- read.csv("input/inputmap.csv", header=TRUE, sep=",", dec=".")
Xnames <- c("pi_c_","dy_","dc_","di_","dE_")
Ynames <- "Recession"

#set output data structure
modeloutput <- data.frame(y=character(),
                 Precision_train=double(), #precision based on training data
                 Recall_train=double(),	   #recall based on training data
                 FMeasure_train=double(),  #F measure based on training data
                 Precision_valid=double(), #precision based on validation data
                 Recall_valid=double(),    #recall based on validation data
                 FMeasure_valid=double(),  #F measure based on validation data
                 Models=character(),	   #Model type
                 stringsAsFactors=FALSE)

lag <-2
Traindata <- rawdata[,names(rawdata) %in% c(Ynames,Xnames)]
Xnames1 <- names(Traindata)[!names(Traindata) %in% c(Ynames)]

# create lagged variables
if (lag > 0) {
	for (i in c(1:lag)){
		for (varname in Xnames1){
			Traindata[(i+1):nrow(Traindata),paste0(varname,i)] <- Traindata[1:(nrow(Traindata)-i),varname]
			Traindata[1:i,paste0(varname,i)] <- NA
		}
	}
}

# 
Traindata <- na.omit(Traindata[(lag+1):nrow(Traindata),])
Xnames2 <- names(Traindata)[!names(Traindata) %in% c(Ynames)]

# create formula
f <-as.formula(paste(Ynames,"~",paste(Xnames2,collapse="+")))
print(f)

# set random seed
set.seed(123)

# split data into training and validation
idx <- sample(seq(1, 2), size = nrow(Traindata), replace = TRUE, prob = c(.8, .2))
training <- Traindata[idx==1,]
validation <- Traindata[idx==2,]

#linear regression
lr<-lm(f, data=training)
predict <- ifelse(lr$fitted.values>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

predict <- ifelse(predict(lr,validation)>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[1,] <- c(Ynames,precision, recall, F, precision_v, recall_v, F_v, "linear")
write.csv(summary(lr)$coefficients,paste0("lr_",Ynames,".csv"))

# Generalized Linear Model (GLM) Logistic model
glmr <- glm(f, data=training, family=binomial)
glmpredict <- predict(glmr, training)
predict <- ifelse(glmpredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

glmpredict <- predict(glmr, validation)
predict <- ifelse(glmpredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[2,] <- c(Ynames, precision, recall, F, precision_v, recall_v, F_v, "glm")
write.csv(glmr$coefficients,paste0("glmr_",Ynames,".csv"))

#cart
cart = rpart(f, data = training, cp = 10^(-3),minsplit = 10,, method = "class")
cartpredict <- predict(cart, training)
predict <- ifelse(cartpredict[,2]>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

cartpredict <- predict(cart, validation)
predict <- ifelse(cartpredict[,2]>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[3,] <- c(Ynames, precision, recall, F, precision_v, recall_v, F_v, "cart")

#knn
knnreg <- knn.reg(train = training[,names(training) %in% Xnames2] , test=training[,names(training) %in% Xnames2], y=training[,Ynames], k=5, algorithm = "kd_tree")
predict <- ifelse(knnreg$pred>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

knnreg <- knn.reg(train = training[,names(training) %in% Xnames2] , test=validation[,names(validation) %in% Xnames2], y=training[,Ynames], k=5, algorithm = "kd_tree")
predict <- ifelse(knnreg$pred>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[4,] <- c(Ynames, precision, recall, F, precision_v, recall_v, F_v, "KNN")

#ann
set.seed(123)

ann <- neuralnet(f, data=data.matrix(training), hidden=c(5), linear.output=TRUE, stepmax = 100000, threshold=0.001, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 1000)
#ann <- neuralnet(f, data=data.matrix(Traindata), hidden=c(10), linear.output=TRUE, stepmax = 200000, threshold=0.5, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 100)
#ann <- neuralnet(f, data=data.matrix(Traindata), hidden=c(5), linear.output=TRUE, stepmax = 200000, threshold=0.5, act.fct = "tanh", likelihood = TRUE, lifesign ="full", lifesign.step = 100)
#ann <- neuralnet(f, data=data.matrix(Traindata), hidden=c(5), linear.output=TRUE, stepmax = 200000, threshold=0.5, act.fct = "logistic", likelihood = TRUE, lifesign ="full", lifesign.step = 100)

predict <- ifelse(ann$net.result[[1]][,1]>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

predict <- ifelse(compute(ann, data.matrix(validation))$net.result[,1]>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[5,] <- c(Ynames, precision, recall, F, precision_v, recall_v, F_v, "ANN")

#lasso
lasso<-glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,Ynames]), alpha = 1, lambda = lambda)
cv.out <- cv.glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,Ynames]), alpha = 1)
lambda <- cv.out$lambda.min
lassopredict <- predict(lasso, s = lambda, newx = as.matrix(training[Xnames2]))
predict <- ifelse(lassopredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

lassopredict <- predict(lasso, s = lambda, newx = as.matrix(validation[Xnames2]))
predict <- ifelse(lassopredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[6,] <- c(Ynames,precision, recall, F, precision_v, recall_v, F_v, "lasso")
write.csv(rbind(as.matrix(lasso$a0), as.matrix(lasso$beta)),paste0("lasso_",Ynames,".csv"))

#ridge
ridge<-glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,Ynames]), alpha = 0, lambda = lambda)
cv.out <- cv.glmnet(as.matrix(training[,Xnames2]), as.matrix(training[,Ynames]), alpha = 0)
lambda <- cv.out$lambda.min
ridgepredict <- predict(ridge, s = lambda, newx = as.matrix(training[Xnames2]))
predict <- ifelse(ridgepredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- training[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("Training: The precision is",precision)
paste("Training: The recall is",recall)
paste("Training: The F is",F)

ridgepredict <- predict(ridge, s = lambda, newx = as.matrix(validation[Xnames2]))
predict <- ifelse(ridgepredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[7,] <- c(Ynames,precision, recall, F, precision_v, recall_v, F_v, "lasso")
write.csv(rbind(as.matrix(lasso$a0), as.matrix(lasso$beta)),paste0("ridge_",Ynames,".csv"))


#Logistic model is the best based on the validation result. Let's use the full dataset to get the parameters.
#It generates the file "glm_all_Recession.csv" which contains the calibrated model used in esg.R (lines 276 - 278)
glmr <- glm(f, data=Traindata, family=binomial)
glmpredict <- predict(glmr, Traindata)
predict <- ifelse(glmpredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- Traindata[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision<-tp/predictyes
recall<-tp/(tp+fn)
F = 2*precision*recall/(precision+recall)
paste("The precision is",precision)
paste("The recall is",recall)
paste("The F is",F)

glmpredict <- predict(glmr, validation)
predict <- ifelse(glmpredict>0.5, 1, 0)
predictyes <- sum(predict==1)
predictno<-sum(predict==0)
actual <- validation[,Ynames]
actualwhenpredyes <- actual[predict==1]
actualwhenpredno <- actual[predict==0]
tp <- sum(actualwhenpredyes==1)
tn <- sum(actualwhenpredno==0)
fp <- length(actualwhenpredyes) - tp
fn <- length(actualwhenpredno) - tn
precision_v<-tp/predictyes
recall_v<-tp/(tp+fn)
F_v = 2*precision_v*recall_v/(precision_v+recall_v)
paste("Validation: The precision is",precision_v)
paste("Validation: The recall is",recall_v)
paste("Validation: The F is",F_v)

modeloutput[8,] <- c(Ynames, precision, recall, F, precision_v, recall_v, F_v, "glm_all")
write.csv(glmr$coefficients,paste0("glm_all_",Ynames,".csv"))

write.csv(modeloutput,paste0("modeloutput2.csv"))

################################################################
# Correlation and Cholesky Decomposition #######################
################################################################

#loop to repair correlation matrix for non-positive definite
repairall <- function(C){

	tryCatch(
	{
		chol(C)
		sa<-1
		return (sa)
	},
		error = function(ex) {
		sa<-0
		return (sa)
		}
	)
}	

#repair correlation matrix for non-positive definite
repaircorr<-function(C){
	# compute eigenvectors/-values
	E   <- eigen(C, symmetric = TRUE)   
	V   <- E$vectors
	D   <- E$values

	# replace negative eigenvalues by 0.001
	D   <- pmax(D,0.001)

	# reconstruct correlation matrix
	BB  <- V %*% diag(D) %*% t(V)

	# rescale correlation matrix
	T   <- 1/sqrt(diag(BB))
	TT  <- outer(T,T)
	C   <- BB * TT
	return (C)
}

#correlation matrix for expansion periods
um<-"pairwise.complete.obs" #"pairwise.complete.obs" "complete.obs" "all.obs" "na.or.complete"
alldata <- read.csv("residuals1elastic3.csv", header=TRUE, sep=",", dec=".")
alldata$Recession <- rawdata$Recession[3:length(rawdata$Recession)] #lag = 2
cordata <- alldata[alldata$Recession == 0,]
cordata <- alldata[,!names(alldata) %in% c("Recession", "X")]
cordata <- cordata[complete.cases(cordata),]
normalcorr <- cor(cordata, use = um, method = "pearson")
normalcorr[is.na(normalcorr)]<-0
icount <-0
while (repairall(normalcorr)==0) {
	normalcorr <- repaircorr(normalcorr)
	icount <- icount+1
	print(icount)
}
normalchol <- chol(normalcorr)

#correlation matrix for recession periods
cordata <- alldata[alldata$Recession == 1,]
cordata <- cordata[,!names(cordata) %in% c("Recession","X")]
recessioncorr <- cor(cordata, use = um, method = "pearson")
recessioncorr[is.na(recessioncorr)]<-0
icount <-0
while (repairall(recessioncorr)==0) {
	recessioncorr <- repaircorr(recessioncorr)
	icount <- icount+1
	print(icount)
}
recessionchol <- chol(recessioncorr)

# Y variables
Xnames <- c("R_","pi_c_","dy_","dc_","di_","dE_","dimp_","dex_","dS_","dw_")
Ynames <- names(rawdata)[!names(rawdata) %in% c(Xnames,"Year","Quarter","Recession","pi_i_","pi_d_","dimp_","dex_","dE_","dS_","dw_","dy_star_","pi_star_","R_star_","aaadefault")]

colnames(normalcorr) <- Ynames
colnames(normalchol) <- Ynames
colnames(recessioncorr) <- Ynames
colnames(recessionchol) <- Ynames
write.csv(normalcorr,paste0("normalcorr.csv"),row.names = FALSE)		
write.csv(normalchol,paste0("normalchol.csv"),row.names = FALSE)		#will be used as input file for esg
write.csv(recessioncorr,paste0("recessioncorr.csv"),row.names = FALSE)
write.csv(recessionchol,paste0("recessionchol.csv"),row.names = FALSE)	#will be used as input file for esg


################################################################
# Macroeconomic factor VAR #####################################
################################################################
library(vars)
rawdata <- read.csv("input/varinput.csv", header=TRUE, sep=",", dec=".")
Xnames <- c("R_","pi_c_","pi_i_","pi_d_","dy_","dc_","di_","dimp_","dex_","dE_","dS_","dw_","dy_star_","pi_star_","R_star_")
Traindata <- rawdata[,names(rawdata) %in% c(Xnames)]
Traindata <- Traindata[complete.cases(Traindata),]
	
var1 <- VAR(Traindata, p = 1, type = "const") #both
stab1 <- stability(var1, h = 0.15, dynamic = FALSE, rescale = TRUE) #type = c("OLS-CUSUM", "Rec-CUSUM", "Rec-MOSUM","OLS-MOSUM", "RE", "ME", "Score-CUSUM", "Score-MOSUM", "fluctuation"),
plot(stab1)

write.csv(summary(var1)$corres, "corres.csv")

#correlation matrix for macroeconomic factors
um<-"pairwise.complete.obs" #"pairwise.complete.obs" "complete.obs" "all.obs" "na.or.complete"
normalcorr <- read.csv("corres.csv", header=TRUE, sep=",", dec=".")
normalcorr[is.na(normalcorr)]<-0
normalcorr <- normalcorr[,c(2:ncol(normalcorr))]
icount <-0
while (repairall(normalcorr)==0) {
	normalcorr <- repaircorr(normalcorr)
	icount <- icount+1
	#print(icount)
}
macrochol <- chol(normalcorr)
write.csv(macrochol,paste0("macrochol.csv"),row.names = FALSE)

################################################################
# Fit data to Normal distribution ##############################
################################################################
rawdata <- read.csv("input/varinput.csv", header=TRUE, sep=",", dec=".")
Xnames <- c("R_","pi_c_","pi_i_","pi_d_","dy_","dc_","di_","dimp_","dex_","dE_","dS_","dw_","dy_star_","pi_star_","R_star_")

LL <- 0
for (i in Xnames) {
	Traindata <- rawdata[,names(rawdata) %in% i]
	Traindata <- as.numeric(Traindata)
	Traindata <- Traindata[!is.na(Traindata)]
	fit <- fitdistr(Traindata, densfun="normal")
	LL<-fit$loglik+LL
}

print(LL)

end_time <- Sys.time()
print(paste0("total run time is ", end_time - start_time, "mins"))

