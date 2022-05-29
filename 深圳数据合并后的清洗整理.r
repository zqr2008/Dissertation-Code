library(tidyverse)
library(readxl)
library(compareGroups)
library(mice)
library(visdat)

#加载的数据集为pandas将原始数据及化验合并的，原始数据已在redcap随访
#无法在化验中找到的患者均剔除
X2021_12_22merge <- read_excel("C:/Users/mjdee/Desktop/JI-2020/JI-深圳/2021_12_22merge.xlsx")

shenzhencleandata<-X2021_12_22merge

shenzhencleandata<-filter(shenzhencleandata,alive_7d==1 | alive_7d==0) #去除失随访15人
shenzhencleandata<-filter(shenzhencleandata,rr_first>0|is.na(rr_first)) #去除来诊呼吸心跳骤停2人
shenzhencleandata<-filter(shenzhencleandata,name_patient!="刘裕兰") #去除缺失过多2人
shenzhencleandata<-dplyr::mutate_at(shenzhencleandata, .vars = vars(45:49), .fun = function(x) ifelse(is.na(x), 0, x))


szclean<-rename(shenzhencleandata,
                chest_pain=ed_chief_complaint___1,
                abdominal_pain=ed_chief_complaint___2,
                Chest_tightness=ed_chief_complaint___3,
                dyspnea=ed_chief_complaint___4,
                fever=ed_chief_complaint___5,
                syncope=ed_chief_complaint___6,
                fatigue=ed_chief_complaint___7,
                palpitation=ed_chief_complaint___8,
                Hematemesis=ed_chief_complaint___9, 
                Bloody_stools=ed_chief_complaint___10,
                Altered_mental_status=ed_chief_complaint___11,
                headache=ed_chief_complaint___12,
                vomiting=ed_chief_complaint___13,
                injury=external_cause)


szclean<-mutate(szclean,O2_FLOW_RATE_FIRST=(fio2_first-21)/4,xiuke_first=(hr_first/sbp_first))

szclean<-select(szclean,
                age_patient,
                chest_pain,
                abdominal_pain,
                Chest_tightness,
                dyspnea,
                fever,
                syncope,
                fatigue,
                palpitation,
                Hematemesis,
                Bloody_stools,
                Altered_mental_status,
                headache,
                vomiting,
                trauma_yn,
                disch_dx_resp,
                injury,
                disch_dx_neoplasms,
                cancer_therapy,
                active_malignancy,
                hematologic_cancer,
                metastatic_cancer,
                disch_dx_abnormal_nos,
                disch_dx_cerebrovasc,
                disch_dx_flu_pneumonia,
                disch_dx_chronic_lower_resp,
                disch_dx_circ_disease,
                chronic_heart_failureiv,
                disch_dx_digestive_disease,
                cirrhosis,
                disch_dx_gu_disease,
                disch_dx_other_disease,
                steroid_therapy,
                disch_dx_aids,
                infection,
                use_vasoactive_drugs,
                planed_admit_erd,
                arrhythmia,
                hypovolemic_hemorrhagic_shock,
                hypovolemic_non_hemorrhagic_shoc,
                septic_shock,
                anaphylactic_shock,
                mix_shoch,
                live_failure,
                seizures,
                coma,
                stupor,
                obtunded,
                agitation,
                vigilance_disturbance,
                confusion,
                focal_neurologic_deficit,
                intracranial_effect,
                acute_abdomen,
                sap,
                gcs_first,
                rr_first,
                hr_first,
                sbp_first,
                spo2_first,
                fio2_first,
                o2_device_first,
                O2_FLOW_RATE_FIRST,
                ph_first_y,
                po2_first_y,
                wbc,
                creatinine_umol_l,
                potassium,
                sodium,
                bun_mmol_l,
                glucose,
                hgb,
                plt,
                tbil,
                xiuke_first,
                alive_7d)
                
impdat <-szclean
impdat <- mice(szclean,m=5,method=c("pmm")) 
summary(impdat)
dataforML<-complete(impdat,2) 
dataforML<-dplyr::mutate_at(dataforML, .vars = vars(63), .fun = function(x) ifelse(is.na(x), 3, x))



shenzhencleandata$gender<-factor(shenzhencleandata$gender,levels = c("1","2"),labels = c("男","女"))
shenzhencleandata$alive_7d<-factor(shenzhencleandata$alive_7d,levels = c("1","0"),labels = c("存活","死亡"))


baselineshenzhen<-descrTable(alive_7d~gender+age_patient+gcs_first+rr_first
                             +hr_first+sbp_first+wbc+plt+hgb+potassium,
           data=shenzhencleandata,method=NA,show.all=TRUE,hide = c(gender = "女"))

export2word(baselineshenzhen, file='C:/Users/mjdee/Desktop/JI-2020/JI-深圳/table1.docx')

