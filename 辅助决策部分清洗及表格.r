---
title: "puth_clean"
author: "zqr2008"
date: "2022/1/17"
output: html_document
---
library(tidyverse)
library(dlookr)
library(mice)

puth<-read.csv("C:/Users/mjdee/Desktop/_DATA_2022-01-17_0221.csv",as.is = TRUE, encoding = 'UTF-8')
puth<-filter(puth,hospital_name==1)
puth<-filter(puth,alive_7d==1 | alive_7d==0)
puth$id_patient<-as.character(puth$id_patient)
puth$X.U.FEFF.record_id<-as.character(puth$X.U.FEFF.record_id)
puth$id_patient<-str_pad(puth$id_patient,12,side="left","0")
puth<-mutate(puth,identity=paste(X.U.FEFF.record_id,id_patient,sep ="-"))

puth_clean<-rename(puth,
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
puth_clean<-mutate(puth_clean,O2_FLOW_RATE_FIRST=(fio2_first-21)/4,xiuke_first=(hr_first/sbp_first))
puth_clean<-select(puth_clean,name_patient,identity,
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
                ph_first,
                po2_first,
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

puth_clean$active_malignancy<-factor(puth_clean$active_malignancy,levels = c("1","0"),labels = c("0","1"))
puth_clean<-dplyr::mutate_at(puth_clean, .vars = vars(19:22), .fun = function(x) ifelse(is.na(x), 0, x))
puth_clean$alive_7d<-factor(puth_clean$alive_7d,levels = c("1","0"),labels = c("0","1"))

impdat <-puth_clean
impdat <-mice(puth_clean,m=5,method=c("pmm")) 
dataforML<-complete(impdat,3)
dataforML<-dplyr::mutate_at(dataforML, .vars = vars(65), .fun = function(x) ifelse(is.na(x), 3, x))

write.table(dataforML,"C:/Users/mjdee/Desktop/JI-2020/ML/PUTH2.csv",row.names=FALSE,col.names=TRUE,sep=",")

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:

```{r cars}
summary(cars)
```

## Including Plots

You can also embed plots, for example:

```{r pressure, echo=FALSE}
plot(pressure)
```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
