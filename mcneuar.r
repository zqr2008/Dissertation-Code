
jresult_table$control=(jresult_table$control_J==jresult_table$真实死活)
jresult_table$experi=(jresult_table$experi_J==jresult_table$真实死活)

mcnemar.test(jresult_table$control,jresult_table$experi,correct = F)
table(jresult_table$control,jresult_table$experi)


Seniorresult_table$control=(Seniorresult_table$control_S==Seniorresult_table$真实死活)
Seniorresult_table$experi=(Seniorresult_table$experi_S==Seniorresult_table$真实死活)
mcnemar.test(Seniorresult_table$control,Seniorresult_table$experi,correct = F)
