\name{ANN.validate}
\alias{ANN.validate}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
ANN.validate(inputValidData, validTargets = vector(), calibratedWeights = computedWeights$trainedWeights, calibratedHidden = computedWeights$trainedHidden, kernel = kernel, verboseValidate = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{inputValidData}{
%%     ~~Describe \code{inputValidData} here~~
}
  \item{validTargets}{
%%     ~~Describe \code{validTargets} here~~
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
  \item{verboseValidate}{
%%     ~~Describe \code{verboseValidate} here~~
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
function (inputValidData, validTargets = vector(), calibratedWeights = computedWeights$trainedWeights, 
    calibratedHidden = computedWeights$trainedHidden, kernel = kernel, 
    verboseValidate = TRUE) 
{
    kernelFunction <- match.fun(kernel)
    inputValidDataDim <- dim(inputValidData)
    inputValidLengthC <- inputValidDataDim[1]
    inputValidLengthR <- inputValidDataDim[2]
    validDistanceList <- vector()
    classes <- vector()
    validHit <- 0
    validMiss <- 0
    validGrand <- 0
    print("=========================================")
    print("-- Validating Model ---------------------")
    print("=========================================")
    for (i in 1:inputValidLengthC) {
        validGrand <- validGrand + 1
        if (verboseValidate == TRUE) {
            print("-- New Sample ---------------------------")
            print("   Forward-Propagating...")
            print(paste("       Valid Sample", i))
        }
        inputValidSample <- inputValidData[i, ]
        validHiddenLayer <- kernelFunction(inputValidSample \%*\% 
            calibratedWeights)
        validHiddenLayer <- c(.insert(validHiddenLayer, element = 1))
        validOutput <- kernelFunction(validHiddenLayer \%*\% calibratedHidden)
        classes <- c(classes, validOutput)
        if (verboseValidate == TRUE) {
            print(paste("       Computed Value:        ", .decimals(validOutput, 
                8)))
            print(paste("       Computed Class:        ", round(validOutput)))
        }
        if (length(validTargets > 0)) {
            validRounded <- round(validOutput)
            validError <- 0.5 * (validTargets[i] - validOutput)^2
            validDistance <- abs(validTargets[i] - validOutput)
            validDistanceList <- c(validDistanceList, validDistance)
            if (verboseValidate == TRUE) {
                print(paste("       Known Class:           ", 
                  validTargets[i]))
                print(paste("       Raw Distance:          ", 
                  .decimals(validDistance, 8)))
                print(paste("       Computed SSE:          ", 
                  .decimals(validError, 8)))
            }
            if (abs(validRounded - validTargets[i]) == 0) {
                validHit <- validHit + 1
                if (verboseValidate == TRUE) 
                  print("       Rounded Hit Status:     Hit! :)")
            }
            else {
                validMiss <- validMiss + 1
                if (verboseValidate == TRUE) 
                  print("       Rounded Hit Status:     Miss :(")
            }
            if (verboseValidate == TRUE) {
                print(paste("           Valid Hits / Total: ", 
                  validHit, "/", validGrand))
                print(paste("           Valid Hit Percent:  ", 
                  .decimals((validHit/validGrand) * 100, 2)))
                print("-- Sample Done --------------------------")
            }
        }
    }
    print("=========================================")
    print("-- Validation Complete ------------------")
    print("=========================================")
    print("-- Report -------------------------------")
    print("   Rounded Hits:")
    print(paste("       Valid Hits / Total:     ", validHit, 
        "/", validGrand))
    print(paste("       Valid Hit Percent:      ", .decimals((validHit/validGrand) * 
        100, 2)))
    list(inferences = classes, classes = round(classes), hits = validHit, 
        total = validGrand, percent = .decimals((validHit/validGrand) * 
            100, 2))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
