
# Matyas
# ------------------------------------------------------------------------------

t <- read.table("bfgs_test1.table", header=FALSE)

f <- function(x1, x2) 0.26*(x1^2 + x2^2) - 0.48*x1*x2

x <- seq(-3, 3, length=100)
y <- x
z <- outer(x, y, f)

image(x, y, z, col=terrain.colors(100))
contour(x, y, z, add=T, col="white")
points(t, type="b")

# Rosenbrock
# ------------------------------------------------------------------------------

t <- read.table("bfgs_test2.table", header=FALSE)

a <- 1
b <- 100
f <- function(x1, x2) (a - x1)^2 + b*(x2 - x1^2)^2

x <- seq(-1.5, 2, length=200)
y <- seq(-0.5, 3, length=200)
z <- outer(x, y, f)

image(x, y, z, col=heat.colors(200)[40:200])
contour(x, y, z, add=T, col="white", nlevels=20)
points(t, type="b")
