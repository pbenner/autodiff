
library(mvtnorm)
library(rjson)
library(cairoDevice)

# ------------------------------------------------------------------------------

library(tikzDevice)

options(tikzMetricPackages = c(
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usetikzlibrary{calc}",
            "\\usepackage{amssymb}"))

# tikz utility
# ------------------------------------------------------------------------------

plot.tikz <- function(filename, expr, width=7.5, height=5, mar=c(5,4,3,2), bareBones=FALSE) {
  cmd <- substitute(expr)
  tikz(file=filename, standAlone=FALSE, bareBones=bareBones, width=width, height=height)
  # remove margin
  par(mar=mar, las=1)
  eval(cmd)
  dev.off()
  system(sprintf("sed -i 's/fill opacity=0.00,//' %s", filename))
  system(sprintf("sed -i 's/\\path[clip].*//' %s", filename))
}

# ------------------------------------------------------------------------------

sigma0 <- matrix(c(1.2, 0.25, 0.25, 1.2), 2)
sigma1 <- matrix(c(0.6, 0.40, 0.40, 0.6), 2)
mu0   <- c(-1.0,  1.0)
mu1   <- c( 1.0, -1.3)

# samples
# ------------------------------------------------------------------------------
if (FALSE) {
    x0 <- rmvnorm(100, mean=mu0, sigma=sigma0)
    x1 <- rmvnorm(100, mean=mu1, sigma=sigma1)

# export
# ------------------------------------------------------------------------------

    write.table(x0, file="roc.x0.table", row.names=F, col.names=F)
    write.table(x1, file="roc.x1.table", row.names=F, col.names=F)
}
# plot mixture
# ------------------------------------------------------------------------------

plot.mixture <- function(i, x0, x1, main="", cex.main=1.0, ...) {
    file0 <- sprintf("roc.result/roc.normal0.%d.json", i)
    file1 <- sprintf("roc.result/roc.normal1.%d.json", i)

    t0 <- fromJSON(file=file0)$Parameters
    t1 <- fromJSON(file=file1)$Parameters

    mu0h    <- as.numeric(t0$Mu)
    mu1h    <- as.numeric(t1$Mu)
    sigma0h <- matrix(t0$Sigma, t0$N, t0$N, byrow=TRUE)
    sigma1h <- matrix(t1$Sigma, t1$N, t1$N, byrow=TRUE)

    # threshold
    tp <- c(0.2, -0.8)
    t <- dmvnorm(tp, mean=mu1h, sigma=sigma1h, log=TRUE) -
         dmvnorm(tp, mean=mu0h, sigma=sigma0h, log=TRUE)
    # resolution
    n <- 50

    # decision boundary
    f <- function(v) {
        v <- drop(v)
        dmvnorm(v, mean=mu1h, sigma=sigma1h, log=TRUE) -
            dmvnorm(v, mean=mu0h, sigma=sigma0h, log=TRUE) - t
    }
    xp <- seq(-4.5, 4.5, length=n)
    yp <- seq(-4.5, 4.5, length=n)
    zp <- outer(xp, yp, function(x,y) apply(cbind(x,y), 1, f))

    # densities
    xy.grid <- expand.grid(xp, yp)
    density0 <- matrix(apply(xy.grid, 1, function(x) dmvnorm(x, mean = mu0h, sigma = sigma0h)), ncol = n)
    density1 <- matrix(apply(xy.grid, 1, function(x) dmvnorm(x, mean = mu1h, sigma = sigma1h)), ncol = n)

    contour(xp, yp, zp, levels=0, drawlabels = FALSE, lwd = 2, lty = 2, main=main, cex.main=cex.main, ...)
    contour(xp, yp, density0, nlevels = 5, drawlabels = FALSE, add = TRUE)
    contour(xp, yp, density1, nlevels = 5, drawlabels = FALSE, add = TRUE, lty=3)
    points(x0, pch=1)
    points(x1, pch=19)
    mtext(side=1, text=expression('X'['i']), line=3.0, las=0)
    mtext(side=2, text=expression('X'['j']), line=2.5, las=0)

}

plot.roc <- function(i, type="l", main="", cex.main=1.0, ...) {
    file <- sprintf("roc.result/roc.%d.table", i)

    t <- read.table(file)
    fpr <- t$V2
    tpr <- t$V3

    plot(fpr, tpr, type=type, xlim=c(0,1), ylim=c(0,1), main=main, cex.main=cex.main, ...)
    lines(c(0,1), c(0,1))
}

plot.fpr <- function(i, type="l", ...) {
    file <- sprintf("roc.result/roc.%d.table", i)

    t <- read.table(file)
    thr <- t$V1
    fpr <- t$V2

    plot(thr, fpr, type=type, ...)
}

# plot latex
# ------------------------------------------------------------------------------
if (FALSE) {
    x0 <- read.table("roc.x0.table")
    x1 <- read.table("roc.x1.table")

    for (i in 0:9) {
        filename <- sprintf("roc.plots/density.%03d.tex", i)
        print(filename)
        plot.tikz(filename, {
            plot.mixture(i*10, x0, x1, xlab="", ylab="")
        }, bareBones=TRUE, width=4, height=3.5, mar=c(5, 3, 1, 1))
    }

    for (i in 0:9) {
        filename <- sprintf("roc.plots/roc.%03d.tex", i)
        print(filename)
        plot.tikz(filename, {
            plot.roc(i*10, xlab="False Positive Rate", ylab="")
            mtext(side = 2, text = "True Positive Rate", line = 4, las=3)
        }, bareBones=TRUE, width=4, height=3.5, mar=c(5, 6, 1, 1))
    }
}
# plot gif
# ------------------------------------------------------------------------------
if (TRUE) {
    x0 <- read.table("roc.x0.table")
    x1 <- read.table("roc.x1.table")

    for (i in 0:99) {
        filename <- sprintf("roc.plots/plot.%03d.png", i)
        print(filename)
        Cairo_png(filename=filename,
            width=8,
            height=4,
            pointsize=12)
        par(mfrow=c(1,2), mar=c(4, 4, 2, 1))
        plot.mixture(i, x0, x1, xlab="", ylab="", main="Sample Space")
        plot.roc(i, xlab="False Positive Rate", ylab="", main="ROC curve")
        mtext(side = 2, text = "True Positive Rate", line = 2.5, las=3)
        dev.off()
    }
    # convert to gif:
    system("convert -delay 5 -loop 0 roc.plots/plot.???.png roc.plots/plot.gif")
}
