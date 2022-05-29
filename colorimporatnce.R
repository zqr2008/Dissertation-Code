#此部分把重要性排序使用swim plot的方法画出来
library(RColorBrewer) # 调色板
library(scales)       # 将y轴转化成百分比格式
library(tidyverse)    # 作图
library(RColorBrewer)
library(swimplot)
library(readxl)
library(reshape2)
library(showtext)

windowsFonts(HEL=windowsFont("Helvetica CE 55 Roman"),
             TNM=windowsFont("Times New Roman"),
             ARL=windowsFont("Arial"),
             songti=windowsFont("Adobe 宋体 Std L"))



importance_order_ <- read_excel("C:/Users/mjdee/Desktop/JI-2020/ML/importance_order!!!.xlsx",
                                col_names =  TRUE )
importance<-importance_order_

df <- data.frame(matrix(unlist(importance), nrow =dim(importance)[1], byrow = FALSE))
df$id <- seq_len(nrow(df))
df<-dplyr::mutate_at(df, .vars = vars(2:12), .fun = function(x) as.numeric(x))
df<-dplyr::mutate_at(df, .vars = vars(13), .fun = function(x) as.character(x))
df<-df%>%mutate(第1次随机=X2,第2次随机=X2+X3,第3次随机=X2+X3+X4,第4次随机=X2+X3+X4+X5,
                第5次随机=X2+X3+X4+X5+X6,第6次随机=X2+X3+X4+X5+X6+X7,
                第7次随机=X2+X3+X4+X5+X6+X7+X8,第8次随机=X2+X3+X4+X5+X6+X7+X8+X9,
                第9次随机=X2+X3+X4+X5+X6+X7+X8+X9+X10,
                第10次随机=X2+X3+X4+X5+X6+X7+X8+X9+X10+X11)

df<-df[1:25,-(2:12)]
df<-melt(df,id=c("X1","id"))

swim<-swimmer_plot(df = df,
             id = "id",
             end = "value",
             name_fill = "variable",
             col = 1, 
             alpha = 0.75,
             width = 0.85)+
  scale_fill_manual(name = "variable", 
                    values = c("第1次随机" = "seagreen4", "第2次随机" = "lightskyblue3", 
                               "第3次随机" = "plum4","第4次随机"="brown",
                               "第5次随机"="firebrick2","第6次随机" = "darkslategrey",
                               "第7次随机"="midnightblue","第8次随机"="royalblue1",
                               "第9次随机"="saddlebrown","第10次随机"="tan2"
                               )) +
  labs(y = "变量得分",
       x = "变量名称")+swimmer_text(df_text = df,id="id",label="X1",adj.y = 0,
                                adj.x = -450,size=7,family="songti")+
  guides(fill=guide_legend(title="随机次数"))+
  theme(axis.title.y=element_text(family="songti",face="bold",size=20), 
        axis.title.x=element_text(family="songti",face="bold",size = 20),
        legend.title =element_text(family = "songti", face="bold",size = 20),
        legend.text=element_text(family = "songti", face="bold",size = 20),
        plot.title=element_text(family="songti",face="bold")) 
