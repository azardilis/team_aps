f <- function(x,l,n,k){
  return(l*(k^n)/(k^n+x^n))
}

logsen <- function(x,k,n){
  return(-n*x^n/(k^n+x^n))
}

png("/Users/paulrubenstein/Documents/Computational_Biology_MPhil_2013/Functional Genomics/team_aps/doc/hill.png")
plot(1:100,f(1:100,1,1,1),ty='l',main=expression(paste("Hill function f with ",lambda,",K,n=1")),
     ylab="f(x)",xlab="x")
dev.off()

png("/Users/paulrubenstein/Documents/Computational_Biology_MPhil_2013/Functional Genomics/team_aps/doc/log-log.png")
plot(1:100,logsen(1:100,1,1),ty='l',main=expression(paste("Log-log sensitivity of Hill function with ",lambda,",K,n=1")),
     ylab="log-log sensitivity",xlab="x")
dev.off()