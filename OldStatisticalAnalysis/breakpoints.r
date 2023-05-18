library("strucchange")
# read data
ExpCurves <- read.csv(file = "ExpCurves.csv")

# for each technology read the data
for (i in unique(ExpCurves$Tech)){
    cost <- log10(ExpCurves[ExpCurves$Tech == i, 'Unit.cost'])
    prod <- log10(ExpCurves[ExpCurves$Tech == i, 'Cumulative.production'])
    # for technologies with enough data points run a structural change analysis 
    # with maximum 2 breaks (i.e., 3 regression lines)
    if (length(cost)>19){
        # compute breakpoints
        bp.cost <- breakpoints(cost ~ prod, breaks = 2)
        # fit new lines in each partition
        fm1 <- lm(cost ~ prod * breakfactor(bp.cost))
        filename <- paste('./figs/', substr(i,1,nchar(i)-4), '.png', sep="")
        png(file=filename)
        print(breakpoints(bp.cost))
        plot(prod, cost, type="p")
        lines(prod, fitted(fm1))
        title(main = i, )
        dev.off()
    }
}
