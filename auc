library(ggplot2)
library(patchwork)
library(ggthemes)
library(haven)
library(reshape2)
library(reshape)
library(readxl)
library(ggsci)



windowsFonts(TNM = windowsFont("Times New Roman"))

AUC2 <- read_excel("C:/Users/mjdee/Desktop/JI-2020/毕业论文/shenzhen_auc.xlsx")
AUC2 <- data.frame(AUC2)
result<-melt(AUC2,id=c("algo"))
other<-result[0:16,]
auc_ci<-result[17:28,]
auc_ci<-cast(auc_ci,algo~variable)

theme_bar <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black',size=1),
          panel.border=element_rect(fill='transparent', color='transparent',size=1),
          panel.grid=element_blank(),#去网格线
          axis.title.x=element_text(face = "bold",size = 20,family = "TNM"),
          axis.title.y=element_text(face = "bold",size = 20,family = "TNM"),#y轴标签加粗及字体大小
          axis.text = element_text(face = "bold",size = 20,family = "TNM"),#坐标轴刻度标签加粗
          # axis.ticks = element_line(color='black'),#坐标轴刻度线
          # axis.ticks.margin = unit(0.8,"lines"),
          legend.title=element_blank(),#去除图例标题
          # legend.justification=c(1,0),#图例在画布的位置(绘图区域外)
          legend.position=c(0.50, 0.95),#图例在绘图区域的位置
          # legend.position='top',#图例放在顶部
          legend.direction = "horizontal",#设置图例水平放置
          # legend.spacing.x = unit(2, 'cm'),
          legend.text = element_text(face = "bold",family = "TNM",size = 13,margin = margin(r=20)),
          legend.background = element_rect( linetype="solid",colour ="black")
          #legend.margin=margin(0,0,0,0)#图例与绘图区域边缘的距离
          # legend.box.margin =margin(-10,0,0,0)
    )
  
}





pic1<-ggplot(data=other, mapping=aes(x = algo, y = value,fill=variable))+
  geom_bar(stat="identity",position=position_dodge(0.75),width=0.6)+
  coord_cartesian(ylim=c(0.3,1.0))+labs(x = "机器学习算法", y = "参数表现")+
  scale_y_continuous(expand = c(0, 0))+#消除x轴与绘图区的间隙
  scale_fill_nejm()+theme_bar()+geom_text(aes(label=sprintf('%.2f',value),y=value+0.01),
                                          position=position_dodge(0.7),vjust=-0.2,hjust=0.55,check_overlap = FALSE,family = "TNM",size=6)

pic2<-ggplot(data=auc_ci, mapping=aes(x = algo, y = AUC))+
  geom_bar(stat="identity", width=0.3, position= "dodge",aes(fill=algo))+
  coord_cartesian(ylim=c(0.3,1.0))+labs(x = "机器学习算法", y = "曲线下面积")+scale_fill_nejm()+
  scale_y_continuous(expand = c(0, 0))+theme_bar()+
  geom_errorbar(aes(ymin=auc_lower, ymax=auc_higher,width=0.2))+
  geom_text(aes(label=sprintf('%.2f(%.2f-%.2f)',AUC,auc_lower,auc_higher),y=AUC+0.14),
            position=position_dodge(0.7),vjust=0.8,hjust=0.55,check_overlap = TRUE,family = "TNM",size=7)+
  theme(legend.position='none')


pic2/pic1
