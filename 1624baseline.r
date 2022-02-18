library(memisc)
library(magrittr)
library(tidyverse)
library(compareGroups)




data0 = as.data.set(spss.system.file("C:/Users/mjdee/Desktop/JI-2020前/结果/真的最终版.sav"))
data = as.data.frame(data0)
data<-data%>% filter(ALIVE_7D_S=="1" | ALIVE_7D_S=="0")
data$outcome_7d<-factor(data$ALIVE_7D_S,levels = c("1","0"),labels = c("存活","死亡"))
data$gender<-factor(data$gendar,levels = c("M","F","7"," "),labels = c("M","F","M","F"))
data$gender[is.na(data$gender)]<-"F"
data$vent_status<-factor(data$O2_DEVICE_FIRST,
                         levels = c("NASAL CANNULE","NONE(ROOM AIR","OXYGEN MASK","VENTURI MASK","VENTILATOR"),
                         labels = c("非呼吸机","非呼吸机","非呼吸机","非呼吸机","呼吸机"))
data$vent_status[is.na(data$vent_status)]<-"非呼吸机"
data$胸痛<-factor(data$chest_pain,levels = c("1","2"),labels = c("有","无"))
data$胸痛[is.na(data$胸痛)]<-"无"

data$腹痛<-factor(data$abdominal_pain,levels = c("1","2"),labels = c("有","无"))
data$腹痛[is.na(data$腹痛)]<-"无"

data$呼吸困难<-factor(data$dyspnea,levels = c("1","2"),labels = c("有","无"))
data$呼吸困难[is.na(data$呼吸困难)]<-"无"

data$意识改变<-factor(data$Altered_mental_status,levels = c("1","2"),labels = c("有","无"))
data$意识改变[is.na(data$意识改变)]<-"无"

data$呼吸系统<-factor(data$呼吸疾病除外肺炎,levels = c("1","2"),labels = c("有","无"))
data$呼吸系统[is.na(data$呼吸系统)]<-"无"

data$循环系统<-factor(data$循环系统疾病,levels = c("1","2"),labels = c("有","无"))
data$循环系统[is.na(data$循环系统)]<-"无"

data$消化系统<-factor(data$消化系统疾病,levels = c("1","2"),labels = c("有","无"))
data$消化系统[is.na(data$消化系统)]<-"无"

data$肿瘤系统<-factor(data$肿瘤病史,levels = c("1","2"),labels = c("有","无"))
data$肿瘤系统[is.na(data$肿瘤系统)]<-"无"

data$脑血管系统<-factor(data$脑血管病,levels = c("1","2"),labels = c("有","无"))
data$脑血管系统[is.na(data$脑血管系统)]<-"无"


data$泌尿系统<-factor(data$泌尿生殖系统疾病,levels = c("1","2"),labels = c("有","无"))
data$泌尿系统[is.na(data$泌尿系统)]<-"无"


baseline1624<-descrTable(outcome_7d~gender+age+GCS_FIRST+RR_FIRST+HR_FIRST
                         +SBP_FIRST+白细胞+血小板+血红蛋白+钾+钠+尿素+肌酐+总胆红素+血糖+PH_即刻+
                           PO2_FIRST+vent_status+胸痛+腹痛+呼吸困难+意识改变
                          +呼吸系统+循环系统+消化系统+肿瘤系统+脑血管系统+泌尿系统,
                             data=data,method=NA,show.all=TRUE, 
                         hide = c(gender = "F",vent_status ="非呼吸机",
                                  胸痛="无",腹痛="无",呼吸困难="无",意识改变="无",
                                  呼吸系统="无",循环系统="无",消化系统="无",
                                  肿瘤系统="无",脑血管系统="无",泌尿系统="无"))

export2word(baseline1624, file='C:/Users/mjdee/Desktop/JI-2020/毕业论文/table_baseline1624.docx')
