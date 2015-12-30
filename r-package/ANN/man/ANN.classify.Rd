\name{ANN.classify}
\alias{ANN.classify}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
ANN.classify(inputTestData, calibratedWeights = computedWeights$trainedWeights, calibratedHidden = computedWeights$trainedHidden, kernel = kernel, verboseClassify = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{inputTestData}{
%%     ~~Describe \code{inputTestData} here~~
}
  \item{calibratedWeights}{
%%     ~~Describe \code{calibratedWeights} here~~
}
  \item{calibratedHidden}{
%%     ~~Describe \code{calibratedHidden} here~~
}
  \item{kernel}{
%%     ~~Describe \code{kernel} here~~
}
  \item{verboseClassify}{
%%     ~~Describe \code{verboseClassify} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (inputTestData, calibratedWeights = computedWeights$trainedWeights, 
    calibratedHidden = computedWeights$trainedHidden, kernel = kernel, 
    verboseClassify = TRUE) 
{
    kernelFunction <- match.fun(kernel)
    inputTestDataDim <- dim(inputTestData)
    inputTestLengthC <- inputTestDataDim[1]
    inputTestLengthR <- inputTestDataDim[2]
    testDistanceList <- vector()
    classes <- vector()
    print("=========================================")
    print("-- Classifying Data ---------------------")
    print("=========================================")
    for (i in 1:inputTestLengthC) {
        if (verboseClassify == TRUE) {
            print("-- New Sample ---------------------------")
            print("   Forward-Propagating...")
            print(paste("       Test Sample", i))
        }
        inputTestSample <- inputTestData[i, ]
        testHiddenLayer <- kernelFunction(inputTestSample \%*\% 
            calibratedWeights)
        testHiddenLayer <- c(.insert(testHiddenLayer, element = 1))
        testOutput <- kernelFunction(testHiddenLayer \%*\% calibratedHidden)
        classes <- c(classes, testOutput)
        if (verboseClassify == TRUE) {
            print(paste("       Computed Value:        ", .decimals(testOutput, 
                8)))
            print(paste("       Computed Class:        ", round(testOutput)))
            print("-- Sample Done --------------------------")
        }
    }
    print("=========================================")
    print("-- Classifications Complete -------------")
    print("=========================================")
    list(inferences = classes, classes = round(classes))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
