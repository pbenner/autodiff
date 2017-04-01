t.rprop  <- read.table("rosenbrock.rprop.table", header=FALSE)
t.bfgs   <- read.table("rosenbrock.bfgs.table", header=FALSE)
t.newton <- read.table("rosenbrock.newton.table", header=FALSE)

# ------------------------------------------------------------------------------

a <- 1
b <- 100
f <- function(x1, x2) (a - x1)^2 + b*(x2 - x1^2)^2

x <- seq(-1.5, 2, length=200)
y <- seq(-0.5, 3, length=200)
z <- outer(x, y, f)

# ------------------------------------------------------------------------------

png("rosenbrock.png", height=350, width=900)
par(mfrow=c(1,3), cex=0.9)

image(x, y, z, col=heat.colors(200)[40:200], main="Rprop")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.rprop,  type="b")

image(x, y, z, col=heat.colors(200)[40:200], main="BFGS")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.bfgs,  type="b")

image(x, y, z, col=heat.colors(200)[40:200], main="Modified Newton")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.newton,  type="b")

dev.off()

# ------------------------------------------------------------------------------

pdf("rosenbrock.pdf", height=7, width=18)
par(mfrow=c(1,3), cex=1.5)

image(x, y, z, col=heat.colors(200)[40:200], main="Rprop")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.rprop,  type="b")

image(x, y, z, col=heat.colors(200)[40:200], main="BFGS")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.bfgs,  type="b")

image(x, y, z, col=heat.colors(200)[40:200], main="Modified Newton")
contour(x, y, z, add=T, col="white", nlevels=20)
points(t.newton,  type="b")

dev.off()
