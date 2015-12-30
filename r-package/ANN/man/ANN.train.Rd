\name{ANN.train}
\alias{ANN.train}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
ANN.train(epochs = 2, inputTrainingData, trainingTargets, kernel = "sigmoid", etaP = 0.1, etaH = 0.01, hiddenNodes = 20, dataWeightsLimit = 0.05, hiddenWeightsLimit = 0.5, plotData = list(), visualizeWeights = FALSE, classification = TRUE, verboseTrain = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{epochs}{
%%     ~~Describe \code{epochs} here~~
}
  \item{inputTrainingData}{
%%     ~~Describe \code{inputTrainingData} here~~
}
  \item{trainingTargets}{
%%     ~~Describe \code{trainingTargets} here~~
}
  \item{kernel}{
%%     ~~Describe \code{kernel} here~~
}
  \item{etaP}{
%%     ~~Describe \code{etaP} here~~
}
  \item{etaH}{
%%     ~~Describe \code{etaH} here~~
}
  \item{hiddenNodes}{
%%     ~~Describe \code{hiddenNodes} here~~
}
  \item{dataWeightsLimit}{
%%     ~~Describe \code{dataWeightsLimit} here~~
}
  \item{hiddenWeightsLimit}{
%%     ~~Describe \code{hiddenWeightsLimit} here~~
}
  \item{plotData}{
%%     ~~Describe \code{plotData} here~~
}
  \item{visualizeWeights}{
%%     ~~Describe \code{visualizeWeights} here~~
}
  \item{classification}{
%%     ~~Describe \code{classification} here~~
}
  \item{verboseTrain}{
%%     ~~Describe \code{verboseTrain} here~~
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
function (epochs = 2, inputTrainingData, trainingTargets, kernel = "sigmoid", 
    etaP = 0.1, etaH = 0.01, hiddenNodes = 20, dataWeightsLimit = 0.05, 
    hiddenWeightsLimit = 0.5, plotData = list(), visualizeWeights = FALSE, 
    classification = TRUE, verboseTrain = TRUE) 
{
    kernelFunction <- match.fun(kernel)
    inputTrainingDataDim <- dim(inputTrainingData)
    inputTrainingLengthC <- inputTrainingDataDim[1]
    inputTrainingLengthR <- inputTrainingDataDim[2]
    trainingWeights <- matrix(runif(((inputTrainingLengthR) * 
        hiddenNodes), min = -dataWeightsLimit, max = dataWeightsLimit), 
        nrow = inputTrainingLengthR, ncol = hiddenNodes, byrow = T)
    hiddenWeights <- matrix(.generateVector(hiddenNodes + 1, 
        lower = -hiddenWeightsLimit, upper = hiddenWeightsLimit), 
        nrow = hiddenNodes + 1, ncol = 1, byrow = T)
    geneExpression <- data.frame()
    if (length(plotData) > 0) {
        if (plotData$SSE == TRUE) {
            par(mfrow = c(1, 1))
            plot(x = NULL, y = NULL, xlim = c(1, epochs), ylim = c(0, 
                1), cex.main = 0.8, ylab = "Classification SSE", 
                xlab = "Epoch", main = "Scatter Plot of Sample Classification \n SSE by Epoch")
        }
        else if (plotData$distance == TRUE) {
            par(mfrow = c(1, 1))
            plot(x = NULL, y = NULL, xlim = c(1, epochs), ylim = c(0, 
                1), cex.main = 0.8, ylab = "Classification Error Distance", 
                xlab = "Epoch", main = "Scatter Plot of Sample Classification \n Error Distance by Epoch")
        }
    }
    if (verboseTrain == FALSE) {
        print("=========================================")
        print("-- Training . . .    --------------------")
        print("=========================================")
    }
    for (n in 1:epochs) {
        distanceList <- vector()
        iRandomization <- sample(inputTrainingLengthC)
        hit <- 0
        miss <- 0
        grand <- 0
        meanSSE <- vector()
        meanDistance <- vector()
        classifications <- vector()
        if (verboseTrain == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Start"))
            print("=========================================")
        }
        for (i in iRandomization) {
            grand <- grand + 1
            total <- i * n
            if (verboseTrain == TRUE) {
                print("-- New Sample ---------------------------")
                print("   Forward-Propagating...")
                print(paste("       Sample", i, "of Epoch", n))
            }
            hiddenLayer <- kernelFunction(inputTrainingData[i, 
                ] \%*\% trainingWeights)
            hiddenLayer <- c(.insert(hiddenLayer, element = 1))
            output <- kernelFunction(hiddenLayer \%*\% hiddenWeights)
            classifications <- c(classifications, output)
            error <- 0.5 * (trainingTargets[i] - output)^2
            distance <- abs(trainingTargets[i] - output)
            distanceList <- c(distanceList, distance)
            rounded <- round(output)
            if (verboseTrain == TRUE) {
                print(paste("       Known Class:           ", 
                  trainingTargets[i]))
                print(paste("       Computed Class:        ", 
                  round(output)))
                print(paste("       Computed Value:        ", 
                  .decimals(output, 8)))
                print(paste("       Raw Distance:          ", 
                  .decimals(distance, 8)))
                print(paste("       Computed SSE:          ", 
                  .decimals(error, 8)))
            }
            if (abs(rounded - trainingTargets[i]) == 0) {
                hit <- hit + 1
                if (verboseTrain == TRUE) 
                  print("       Rounded Hit Status:     hit! :)")
            }
            else {
                miss <- miss + 1
                if (verboseTrain == TRUE) 
                  print("       Rounded Hit Status:     miss :(")
            }
            if (verboseTrain == TRUE) {
                print(paste("           Epoch Hits / Total:", 
                  hit, "/", grand))
                print(paste("           Epoch Hit Percent: ", 
                  .decimals((hit/grand) * 100, 2)))
                print("   Back-Propagating...")
            }
            deltaP <- drop((trainingTargets[i] - output) * output * 
                (1 - output))
            deltaH <- (hiddenLayer * (1 - hiddenLayer) * hiddenWeights * 
                deltaP)[-c(1)]
            hiddenChange <- etaH * hiddenLayer * deltaP
            updatedHidden <- as.matrix(hiddenLayer) + as.matrix(hiddenChange)
            hiddenLayer <- updatedHidden
            weightChange <- etaP * t(inputTrainingData[i, , drop = FALSE]) \%*\% 
                deltaH
            updatedWeights <- trainingWeights + weightChange
            trainingWeights <- updatedWeights
            if (length(plotData) > 0) {
                if (plotData$SSE == TRUE) {
                  meanSSE <- c(.insert(meanSSE, element = error))
                  points(x = n, y = error, col = "orange")
                }
                else if (plotData$distance == TRUE) {
                  meanDistance <- c(.insert(meanDistance, element = distance))
                  points(x = n, y = distance, col = "blue")
                }
            }
            if (verboseTrain == TRUE) 
                print("-- Sample Done --------------------------")
        }
        if (verboseTrain == TRUE) {
            print("=========================================")
            print(paste("-- Epoch", n, "Done"))
            print("=========================================")
        }
        if (length(plotData) > 0) {
            if (plotData$SSE == TRUE) {
                points(x = n, y = mean(meanSSE), col = "green")
            }
            else if (plotData$distance) {
                points(x = n, y = mean(meanDistance), col = "red")
            }
        }
    }
    if (verboseTrain == FALSE) {
        print("=========================================")
        print("-- Training Complete     ----------------")
        print("=========================================")
        print("-- Report -------------------------------")
        print("   Rounded Hits Last Epoch:")
        print(paste("       Train Hits / Total:     ", hit, "/", 
            grand))
        print(paste("       Train Hit Percent:      ", .decimals((hit/grand) * 
            100, 2)))
    }
    if (length(plotData) > 0) {
        if (plotData$SSE == TRUE) {
            legend(1, 1, legend = c("Mean SSE", "SSE per Sample"), 
                pch = 1, col = c("green", "orange"), cex = 0.6)
        }
        else if (plotData$distance == TRUE) {
            legend(1, 1, legend = c("Mean Error Distance", "Error Distance per Sample"), 
                pch = 1, col = c("red", "blue"), cex = 0.6)
        }
        else if (plotData$weightMeans$plot == TRUE) {
            geneExpression <- as.data.frame(t(apply(trainingWeights, 
                1, mean)))[-c(1)]
            names(geneExpression) <- geneNames
            image(t(as.matrix(geneExpression)), axis = FALSE, 
                main = "Heat Map of Mean Weights Computed for \n Gene Expression Levels", 
                axes = FALSE)
        }
    }
    list(classifications = classifications, trainedWeights = trainingWeights, 
        trainedHidden = hiddenWeights, kernel = kernel, meanGeneExpression = geneExpression)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
