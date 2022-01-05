library(ggplot2)
library(splines)

#' Convolution in R
#' This is a function about how to calculate convolution of two vectors in R.
#' @param Y a vector being the observations of the response variable
#' @param X a vec being the variable of interest
#' @param Z a matrix being the covariates 
#' @param tau00 a vector being the estimated knots
#' @param quantile the critical value based on your interest, default is 1.96
#' @return a ggplot2 object
#' @examples
#' library(lsknots)
#' set.seed(123456)
#' n=200#sample size
#' theta = c(5, -1, 3, 5,-9,6 , 0.3, 0.6, 0.8, 0.9)
#' tau00=c(0.3,0.5,0.85,0.95)
#' theta = c(9, -5, 6, 7,-8,-3 , 0.1, 0.3, 0.5, 0.7)
#' tau00=c(0.05,0.45,0.55,0.8)
#' eta = 0.5 
#' sig = 0.03
#' a=  rnorm(n,0,1)
#' Z = rnorm(n,0,2) 
#' u = (a+Z)/sqrt(5)
#' X= pnorm(u,mean=0,sd=1) 
#' e = rnorm (n,0,sd=sig)  
#' Y = theta[1] + theta[2]*X + theta[3]*as.numeric(X > theta[7]) * ( X - theta[7] ) + theta[4] * as.numeric(X > theta[8]) * ( X - theta[8] ) + theta[5] * as.numeric(X > theta[9]) * ( X - theta[9] )  + theta[6] * as.numeric(X > theta[10]) * ( X - theta[10] )+ eta*Z+ e 
#' ls_plot(Y,X,Z,tau00,1.96)
ls_plot <- function(Y,X,Z,tau00,quantile=1.96)
{
  A=matrix(c(X,Z),ncol=2)
  k = knots_find(Y,A,tau00)
  z = as.matrix(Z)
  sdout = knots_sd(Y, X, z,k)
  xrange <- range(X)
  X.grid <- seq(from=xrange[1],to=xrange[2],by=0.0001)
  fit <- lm(Y~bs(X,knots=k,degree=1))
  pred=predict(fit,newdata=list(X=X.grid),se=T)
  Iv = interval(quantile, k, sdout)
  
  df1 = data.frame(X,Y)
  df2 = data.frame(X.grid,pred=pred$fit)
  
  gp = ggplot() + 
    geom_point(data=df1, aes(x=X,y=Y)) + 
    geom_vline(xintercept=k, linetype='dashed') + 
    geom_line(data=df2,aes(x=X.grid, y=pred)) +
    geom_rect(aes(xmin=Iv$Left,xmax=Iv$Right,ymin=-Inf,ymax=Inf,fill=k) ,alpha= .5)+
    xlab("the factor of interest") +
    ggtitle("linear spline model")+
    theme(plot.title=element_text(hjust=0.5))+
    scale_x_continuous(breaks = seq(-0.5,2,0.2)) +
    scale_y_continuous(breaks = seq(0,10,2))
  
  return(gp)
}

