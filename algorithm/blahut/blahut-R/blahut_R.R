
# Usage:
# source("PATH/blahut_R.R", chdir = TRUE)

# ------------------------------------------------------------------------------

blahut.dir <- getwd()
blahut <- function(n, channel, px.init, lambda = 1.0, binary = sprintf("%s/blahut-R", blahut.dir)) {
    channel.str <- capture.output(str(t(channel), vec.len=length(channel)))
    px.init.str <- capture.output(str(px.init, vec.len=length(px.init)))

    # create fifo
    output <- sprintf("%s.out", binary)
    # create pipes
    cmd <- sprintf("%s -i %d -l %f - > %s 2>&1", binary, n, lambda, output)
    p   <- pipe(cmd)
    # execute command
    writeLines(sprintf("%s\n%s\n", channel.str, px.init.str), con = p)
    # get and parse output
    eval(parse(output))
}

# ------------------------------------------------------------------------------
if (FALSE) {

    channel <- matrix(
        c(0.60, 0.30, 0.10,
          0.70, 0.10, 0.20,
          0.50, 0.05, 0.45),
        3, 3)

    px.init <- c(1/3, 1/3, 1/3)

    px <- blahut(channel, px.init)

}
