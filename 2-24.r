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
data$xiuke_first<-(data$HR_FIRST)/(data$SBP_FIRST)
data$FiO2_FIRST<-as.numeric(data$FiO2_FIRST)

fact_funct<-function(x){factor(x,levels = c("1","2"),labels = c("有","无"))}
change_na<-function(x){x[is.na(x)]<-"无"}

baseline_table<-data
baseline_table<- baseline_table%>% mutate_at(
  c("chest_pain","abdominal_pain","Chest_tightness","dyspnea","fever","syncope",
    "fatigue","palpitation","Hematemesis","Bloody_stools",
    "Altered_mental_status","headache","vomiting",
    "TRAUMA_YN","DISCH_DX_RESP","DISCH_DX_INJURY",
    "DISCH_DX_NEOPLASMS","Cancer_Therapy","ACTIVE_MALIGNANCY",
    "Hematologic_cancer","Metastatic_cancer",         
    "DISCH_DX_ABNORMAL_NOS","DISCH_DX_CEREBROVASC",            
    "DISCH_DX_FLU_PNEUMONIA","DISCH_DX_CHRONIC_LOWER_RESP",    
    "DISCH_DX_CIRC_DISEASE","chronic_heart_failureIV",
    "DISCH_DX_DIGESTIVE_DISEASE","Cirrhosis",     
    "DISCH_DX_GU_DISEASE","DISCH_DX_OTHER_DISEASE",        
    "STEROID_THERAPY","艾滋病","Infection",
    "Use_Vasoactive_Drugs"),.fun=fact_funct)

baseline_table<- baseline_table%>% mutate_at(
  c("chest_pain","abdominal_pain","Chest_tightness","dyspnea","fever","syncope",
    "fatigue","palpitation","Hematemesis","Bloody_stools",
    "Altered_mental_status","headache","vomiting",
    "TRAUMA_YN","DISCH_DX_RESP","DISCH_DX_INJURY",
    "DISCH_DX_NEOPLASMS","Cancer_Therapy","ACTIVE_MALIGNANCY",
    "Hematologic_cancer","Metastatic_cancer",         
    "DISCH_DX_ABNORMAL_NOS","DISCH_DX_CEREBROVASC",            
    "DISCH_DX_FLU_PNEUMONIA","DISCH_DX_CHRONIC_LOWER_RESP",    
    "DISCH_DX_CIRC_DISEASE","chronic_heart_failureIV",
    "DISCH_DX_DIGESTIVE_DISEASE","Cirrhosis",     
    "DISCH_DX_GU_DISEASE","DISCH_DX_OTHER_DISEASE",        
    "STEROID_THERAPY","艾滋病","Infection",
    "Use_Vasoactive_Drugs"),.fun=~replace_na(., "无"))


baseline_table<- baseline_table%>% mutate_at(
  c("Planed_Admit_ERD","X._arrhythmia","Hypovolemic_hemorrhagic_shock",
      "Hypovolemic_non_hemorrhagic_shoc","Septic_shock",                   
      "Anaphylactic_shock","Mix_shoch","Live_failure","Seizures",                        
      "coma","stupor","obtunded","Agitation","Vigilance_disturbance",
      "Confusion","Focal_neurologic_deficit","Intracranial_effect",
      "Acute_Abdomen","SAP"),.fun=~replace_na(., "N"))

baseline1624<-descrTable(outcome_7d~gender+age+chest_pain+abdominal_pain+
                           Chest_tightness+dyspnea+fever+syncope+
                           fatigue+palpitation+Hematemesis+Bloody_stools+
                           Altered_mental_status+headache+vomiting+
                           TRAUMA_YN+DISCH_DX_RESP+DISCH_DX_INJURY+
                           DISCH_DX_NEOPLASMS+Cancer_Therapy+ACTIVE_MALIGNANCY+
                           Hematologic_cancer+Metastatic_cancer+         
                           DISCH_DX_ABNORMAL_NOS+DISCH_DX_CEREBROVASC+            
                           DISCH_DX_FLU_PNEUMONIA+DISCH_DX_CHRONIC_LOWER_RESP+    
                           DISCH_DX_CIRC_DISEASE+chronic_heart_failureIV+
                           DISCH_DX_DIGESTIVE_DISEASE+Cirrhosis+     
                           DISCH_DX_GU_DISEASE+DISCH_DX_OTHER_DISEASE+        
                           STEROID_THERAPY+艾滋病+Infection+
                           Use_Vasoactive_Drugs+Planed_Admit_ERD+
                           X._arrhythmia+Hypovolemic_hemorrhagic_shock+
                           Hypovolemic_non_hemorrhagic_shoc+Septic_shock+                   
                           Anaphylactic_shock+Mix_shoch+Live_failure+Seizures+                        
                           coma+stupor+obtunded+Agitation+Vigilance_disturbance+
                           Confusion+Focal_neurologic_deficit+Intracranial_effect+
                           Acute_Abdomen+SAP+GCS_FIRST+RR_FIRST+HR_FIRST+SBP_FIRST+SpO2_FIRST+
                           FiO2_FIRST+vent_status+O2_FLOW_RATE_FIRST+xiuke_first+
                           PH_即刻+PO2_FIRST+白细胞+肌酐+钾+钠+尿素+血糖+
                           血红蛋白+血小板+总胆红素,
                           data=baseline_table,method=NA,show.all=TRUE,
                           hide.no=c("无","N"))


export2word(baseline1624, file='C:/Users/mjdee/Desktop/JI-2020/毕业论文/table_baseline1624.docx')
