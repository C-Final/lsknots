#define _USE_MATH_DEFINES
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

//dummy_vec
arma::vec dummy_vec(arma::vec x, double tau)
{
  uvec q1=find(x > tau);
  uvec q2=find(x<=tau);
  x.elem(q1) -=tau;
  x.elem(q2).zeros();
  return x;
}

//dummy_vec1
arma::vec dummy_vec1(arma::vec x, double tau)
{
  uvec q1=find(x > tau);
  uvec q2=find(x<=tau);
  x.elem(q1).ones();
  x.elem(q2).zeros();
  return x;
}

//dummy_matrix
arma::mat dummy_matrix(arma::vec x, arma::vec tau_foo)
{
  int k=tau_foo.n_elem;
  int n=x.n_elem;
  mat indicator_X(n,k);
  for (int i=0; i<k;i++)
  {
    indicator_X.col(i)=dummy_vec1(x,tau_foo(i));
  }
  return indicator_X;
}

//matrix_multiply
arma::mat matrixMult(arma::mat A)
{
  int Acol=A.n_cols;
  int Arow=A.n_rows;
  
  mat D(Acol,Acol,fill::zeros);
  
  for (int i=0; i<Acol; i++)
  {
    for (int j=0;j<Acol;j++)
    {
      double s=0;
      for (int k=0; k<Arow;k++)
      {
        s += A(k,i)*A(k,j);
      }
      D(i,j)=s;
    }
  }
  
  return D;
}

//solve the inverse using svd
mat svdl(mat A){
  mat U, V;
  vec s;
  svd_econ(U,s,V,A);
  return V*diagmat(1/s)*U.t();
}

//' OLS with estimated knots
//' This is a function about how to calculate the OLS estimator with given knots.
//' @param Y a vector being the observations of the response variable
//' @param X a matrix with the first column being the variable of interest and the left column being the covariates 
//' @param tau_iter a vector being the estimated knots 
//' @return estimate of the coefficients and response variable
//' @examples
//' library(lsknots)
//' set.seed(12345)
//' n = 1000
//' eta = 0.5 
//' sig = 0.03
//' a=  rnorm(n,0,1)
//' Z = rnorm(n,0,2) 
//' u = (a+Z)/sqrt(5)
//' X= pnorm(u,mean=0,sd=1) 
//' e = rnorm (n,0,sd=sig)  
//' Y = theta[1] + theta[2]*X + theta[3] * as.numeric(X > theta[7]) * ( X - theta[7] ) + theta[4] * as.numeric(X > theta[8]) * ( X - theta[8] ) + theta[5] * as.numeric(X > theta[9]) * ( X - theta[9] )  + theta[6] * as.numeric(X > theta[10]) * ( X - theta[10] )+ eta*Z+ e 
//' A=matrix(c(X,Z),ncol=2)
//' tau00=c(0.3,0.5,0.85,0.95)
//' yhat =knots_ols(Y,A,tau0)$y_hat
//' betahat=knots_ols(Y,A,tau0)$beta_hat
// [[Rcpp::export]]
Rcpp::List knots_ols(arma::vec Y, arma::mat X, arma::vec tau_iter) 
{
  int k=tau_iter.n_elem; // number of knots
  int n=X.n_rows; // number of observations
  mat X_knot(n,k);
  vec one(n,fill::ones);
  X.insert_cols(0,one);
  
  for (int i=0;i<k;i++)
  {
    X_knot.col(i)=dummy_vec(X.col(1),tau_iter(i));
  }
  X.insert_cols(2,X_knot);
  
  vec beta_hat=svdl(X)*Y;
  vec y_hat=X*beta_hat;
  
  return Rcpp::List::create(Rcpp::Named("beta_hat")=beta_hat,
                            Rcpp::Named("y_hat")=y_hat);
}


//' updated NR methods to find knots
//' This is a function about how to find knots in spline linear regression
//' @param Y a vector being the observations of the response variable
//' @param X a matrix with the first column being the variable of interest and the left column being the covariates 
//' @param tau_iter a vector being the estimated knots
//' @param epsilon threshold for error 
//' @examples
//' library(lsknots)
//' set.seed(12345)
//' n = 1000
//' eta = 0.5 
//' sig = 0.03
//' a=  rnorm(n,0,1)
//' Z = rnorm(n,0,2) # Z: confounder/other covariate
//' u = (a+Z)/sqrt(5)
//' X= pnorm(u,mean=0,sd=1) #X: factor of interest with change-point effects: uniform (0,1)
//' e = rnorm (n,0,sd=sig)  #e: error term
//' Y = theta[1] + theta[2]*X + theta[3] * as.numeric(X > theta[7]) * ( X - theta[7] ) + theta[4] * as.numeric(X > theta[8]) * ( X - theta[8] ) + theta[5] * as.numeric(X > theta[9]) * ( X - theta[9] )  + theta[6] * as.numeric(X > theta[10]) * ( X - theta[10] )+ eta*Z+ e 
//' tau00=c(0.3,0.5,0.85,0.95)
//' A=matrix(c(X,Z),ncol=2)
//' knots_find(Y,A,tau00)
// [[Rcpp::export]]
arma::vec knots_find(arma::vec Y, arma::mat X, arma::vec tau_iter, double epsilon=0.000001)
{ 
  vec tau0=tau_iter;
  int k=tau_iter.n_elem; // number of knots
  vec tau1(k);
  int n=X.n_rows; // number of observations
  double eps=1.0;
  int num=0;
  vec one(n,fill::ones);
  X.insert_cols(0,one);
  
  while(eps>epsilon && num<1000)
  {
    //step1
    mat X_knot(n,k);
    for (int i=0;i<k;i++)
    {
      X_knot.col(i)=dummy_vec(X.col(1),tau0(i));
    }
    mat X_mid=X;
    X_mid.insert_cols(2,X_knot);
   
    vec beta_hat=svdl(X_mid)*Y;
    vec y_hat=X_mid*beta_hat;
    
    //step2
    mat indicator_tau=dummy_matrix(X_mid.col(1),tau0);
    vec U_i=beta_hat.rows(2,2+k-1)%((Y-y_hat).t()*indicator_tau).t()/n;
    mat J_i=diagmat(beta_hat.rows(2,2+k-1))*matrixMult(indicator_tau)*diagmat(beta_hat.rows(2,2+k-1))/n;
    
    tau1=tau0-inv(J_i)*U_i;
    eps=norm(tau1-tau0);
    num +=1;
    tau0=tau1;
  }
  
  return tau1;
}

//' function to find the estimated standard deviation for knots
//' This is a function about how to find the estimated standard deviation for knots in spline linear regression
//' @param y a vector being the observations of the response variable
//' @param X a vec being the variable of interest
//' @param Z a matrix being the covariates 
//' @param tau_out a vector being the estimated knots
//' @return sd_out a vector being the estimated standard deviation for knots
//' @examples
//' library(lsknots)
//' set.seed(12345)
//' n = 1000
//' eta = 0.5 
//' sig = 0.03
//' a=  rnorm(n,0,1)
//' Z = rnorm(n,0,2) # Z: confounder/other covariate
//' u = (a+Z)/sqrt(5)
//' X= pnorm(u,mean=0,sd=1) #X: factor of interest with change-point effects: uniform (0,1)
//' e = rnorm (n,0,sd=sig)  #e: error term
//' Y = theta[1] + theta[2]*X + theta[3] * as.numeric(X > theta[7]) * ( X - theta[7] ) + theta[4] * as.numeric(X > theta[8]) * ( X - theta[8] ) + theta[5] * as.numeric(X > theta[9]) * ( X - theta[9] )  + theta[6] * as.numeric(X > theta[10]) * ( X - theta[10] )+ eta*Z+ e 
//' tau_out = knots_find(Y,A,tau00)
//' sdout = knots_sd(Y, X, as.matrix(Z),tau_out)
// [[Rcpp::export]]
arma::vec knots_sd(arma::vec y,arma::vec X,arma::mat Z,arma::vec tau_out) {
  int n=tau_out.n_elem;
  int m=X.n_elem;
  mat x_tau(m,n);
  mat I_tau(m,n);
  for(int i(0);i<m;i++){
    for (int j(0);j<n;j++){
      int q=X(i)>tau_out(j);
      if(q){
        I_tau(i,j)=q;
        x_tau(i,j)=q*(X(i)-tau_out(j));
      }
    }
  }
  x_tau.insert_cols(0,X);
  vec one(m,fill::ones);
  x_tau.insert_cols(0,one);
  x_tau=join_rows(x_tau,Z);
  
  int p=x_tau.n_cols;
  mat xtx=x_tau.t()*x_tau;
  vec xty=x_tau.t()*y;
  vec betahat=solve(xtx,xty);
  vec res=y-x_tau*betahat;
  
  double s=0.0;
  for(int i(0);i<m;i++){
    s+=res(i)*res(i);
  }
  double sighat=sqrt(s/(m-p));
  vec theta0(n+p);
  for(int i(0);i<p;i++){
    theta0(i)=betahat(i);
  }
  for(int i(0);i<n;i++){
    theta0(i+p)=tau_out(i);
  }
  
  mat Jn1=x_tau.t()*x_tau/m;
  mat diag_theta0=diagmat(theta0(span(2,n+1)));
  mat temp=I_tau*diag_theta0;
  mat Jn2=-x_tau.t()*temp/m;
  mat Jn3=Jn2.t();
  mat Jn4=temp.t()*temp/m;
  
  //solve svd
  mat J=svdl(Jn4-Jn3*svdl(Jn1)*Jn2);
  mat V=sighat*sighat*J/(m-p-n);
  vec sd_out=sqrt(V.diag(0));
  
  return sd_out;
}

// [[Rcpp::export]]
Rcpp::List interval(double quantile,arma::vec tau_out,arma::vec sd_out) 
{
  return Rcpp::List::create(Rcpp::Named("Value")=tau_out,
                            Rcpp::Named("Left")=tau_out-quantile*sd_out,
                            Rcpp::Named("Right")=tau_out+quantile*sd_out);
}