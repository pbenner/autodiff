library(latexreadme)
library(knitr)

rmd1 = file.path("README_input.md")
rmd2 = file.path("README.md")

parse_latex(rmd1,
            rmd2,
            git_username  = "pbenner",
            git_reponame  = "autodiff/master/demo/channel/README",
            git_branch    = "",
            raw_git_site  = "https://raw.githubusercontent.com",
            insert_string = paste0("\n<img src=\"%s%s\" ", "alt=\"\" ", "height=\"", 60, "\">\n")
#            insert_string = paste0("\n<img src=\"%s%s\" ", "alt=\"\" ", ">\n")
            )

system("mv eq_no_* README")
