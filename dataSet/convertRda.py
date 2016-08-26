import rpy2
import rpy2.robjects as robjects


rload = robjects.r['load']
rload('./Weekly.rda')
rwrite = robjects.r['write.csv']
rwrite(robjects.r.Weekly, file="Weekly.csv")
