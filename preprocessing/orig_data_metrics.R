cohorts <- readRDS(file.path(".", "data", "PCa_cohorts.Rds"))
library('Biobase')
colnames(pData(cohorts$CancerMap_2017_Luca))
colnames(pData(cohorts$Stockholm_2016_Ross_Adams))
#pData(cohorts$Belfast_2018_Jain)$PATH_T_STAGE
pData(cohorts$Stockholm_2016_Ross_Adams)$PATH_T_STAGE
pData(cohorts$CancerMap_2017_Luca)
sum(is.na(pData(cohorts$Stockholm_2016_Ross_Adams)$AGE))/nrow(pData(cohorts$Stockholm_2016_Ross_Adams))
sum(is.na(pData(cohorts$Stockholm_2016_Ross_Adams)$PATH_T_STAGE))/nrow(pData(cohorts$Stockholm_2016_Ross_Adams))
pData(cohorts$Belfast_2018_Jain)$PATH_T_STAGE

any(is.na(pData(cohorts$Belfast_2018_Jain)$PATH_T_STAGE))
any(is.na(pData(cohorts$Belfast_2018_Jain)$CLIN_T_STAGE))

type(as.numeric(pData(cohorts$Atlanta_2014_Long)$BCR_STATUS))


exprs(cohorts[[1]])
ncol(exprs(cohorts$CamCap_2016_Ross_Adams))

for (i in 1:9) {
  print(names(cohorts)[i])
  print(nrow(exprs(cohorts[[i]])))
  print(ncol(exprs(cohorts[[i]])))
  print('next')
}
