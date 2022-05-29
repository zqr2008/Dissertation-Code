combine hospital 2 to JI format
data processing:
1.对应表：
#以下为两个df中变量对应关系！
#df4['#白细胞计数']=df3['wbc']
#df4['肌酐[Cr]']=df3['creatinine_umol_l']
#df4['#钾[K]']=df3['potassium']
#df4['#钠[Na]']=df3['sodium']
#df4['尿素[urea]']=df3['bun_mmol_l']
#df4['葡萄糖[GLU]']=df3['glucose']
#df4['#血红蛋白']=df3['hgb']
#df4['#血小板计数']=df3['plt']
#df4['总胆红素[TB]']=df3['tbil']
#df4['纤维蛋白原[FIB]']=df3['fib']
#df4['D-二聚体[D-D]']=df3['ddimer']
#df4['嗜酸粒细胞百分率']=df3['e_fraction']
#df4['嗜酸粒细胞绝对值']=df3['e']
#df4['红细胞压积[HCT]']=df3['hct']
#df4['平均血红蛋白含量']=df3['mch']
#df4['平均血红蛋白浓度']=df3['mchc']
#df4['中性粒细胞百分率']=df3['n_fraction']
#df4['中性粒细胞绝对值']=df3['n']
#df4['血小板压积']=df3['thrombocytocrit']
#df4['淋巴细胞绝对值']=df3['l']
#df4['淋巴细胞百分率']=df3['l_fraction']
#df4['#丙氨酸氨基转移酶[ALT]']=df3['alt']
#df4['#天门冬氨基转移酶[AST]']=df3['ast']
#df4['#碱性磷酸酶[ALP]']=df3['alp']	
#df4['钙[Ca]']=df3['ca']
#df4['肌酸激酶[CK]']=df3['ck']
#df4['肌酸激酶MB亚型质量[CK-MB mass]']=df3['ck_mb']
#df4['二氧化碳结合力[CO2-CP]']=df3['tco2']
#df4['#乳酸脱氢酶[LDH]']=df3['ldh']
#df4['活化部分凝血激酶时间[APTT]']=df3['aptt']
#df4['PT国际标准化比值[INR]']=df3['inr']
#df4['凝血酶原时间[PT]']=df3['pt']
#df4['PT活动度[PT%]']=df3['a']
#df4['凝血酶凝固时间[TT]']=df3['tt']
#df4['#白蛋白[Alb]']=df3['alb']
#df4['降钙素原[PCT]']=df3['pct']

2.化验多出一部分，是以"多"的版本

3.df对应关系
df:原始化验读取表dataframe
df1:更正化验表基础信息名称后的dataframe
df4:将df1按要求变形后的表
df_index:将基础信息摘出来
df_labtitle:将缺失的lab项目摘出来
df2:将不需要的东西删除后拼接的新daaframe
df3:设置索引和df4一致
df_selected:将df4中目标变量挑选出来	
df_merged:df3和df_selected合并化验的结果，用'name_patient'为index,作内连接
df5:选择真正的可以匹配上的结果，后缀为y
