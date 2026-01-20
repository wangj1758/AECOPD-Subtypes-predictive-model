import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 加载模型
model_path = "stacking_Classifier_model.pkl"
try:
    stacking_classifier = joblib.load(model_path)
except:
    st.error("⚠️ 模型文件未找到，请确保 stacking_Classifier_model.pkl 在同一目录下")
    st.stop()

# 设置页面配置和标题
st.set_page_config(
    layout="wide", 
    page_title="AECOPD亚型预测系统", 
    page_icon="🏥"
)

st.title("🏥 AECOPD出院后1年内急性加重再住院亚型预测系统")
st.write("""
基于Stacking集成学习模型，预测AECOPD患者出院后1年内急性加重再住院的4个亚型。
本系统整合了**6个关键临床特征**，为临床决策提供辅助支持。
""")

# 左侧侧边栏输入区域
st.sidebar.header("📋 临床特征输入")
st.sidebar.write("请输入患者的6项关键临床特征：")

# 定义特征输入（连续变量）
st.sidebar.subheader("理化指标 (4项)")

MCH = st.sidebar.number_input(
    "平均血红蛋白量 (pg)", 
    min_value=18.1, 
    max_value=43.3, 
    value=30.0,
    help="正常范围: 27-34 pg"
)

apoA = st.sidebar.number_input(
    "载脂蛋白A (g/L)", 
    min_value=0.34, 
    max_value=2.61, 
    value=1.2,
    help="正常范围: 1.0-1.6 g/L"
)

uric_acid = st.sidebar.number_input(
    "尿酸 (μmol/L)", 
    min_value=71.0, 
    max_value=731.3, 
    value=300.0,
    help="正常范围: 男性208-428, 女性155-357 μmol/L"
)

FVC = st.sidebar.number_input(
    "FVC占预计值的百分比 (%)", 
    min_value=22.92, 
    max_value=139.45, 
    value=80.0,
    help="正常范围: ≥80%"
)

# 定义特征输入（二分类变量）
st.sidebar.subheader("中医证候信息 (2项)")

fever = st.sidebar.selectbox(
    "发热", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="是否出现发热症状"
)

tan_re = st.sidebar.selectbox(
    "痰热壅肺证", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="中医辨证是否为痰热壅肺证"
)

# 添加预测按钮
predict_button = st.sidebar.button("🔮 开始预测", type="primary")

# 主页面用于结果展示
if predict_button:
    st.header("📊 预测结果")
    
    try:
        # 将输入特征转换为模型所需格式（按照训练时的特征顺序）
        # 训练时的正确特征顺序：平均血红蛋白量, 发热, 痰热壅肺证, 载脂蛋白A, 尿酸, FVC_BEST/PRED
        input_array = np.array([
            MCH,           # 平均血红蛋白量
            fever,         # 发热 (注意：移到第2位)
            tan_re,        # 痰热壅肺证 (注意：移到第3位)
            apoA,          # 载脂蛋白A (注意：移到第4位)
            uric_acid,     # 尿酸 (注意：移到第5位)
            FVC            # FVC占预计值的百分比 (注意：移到第6位)
        ]).reshape(1, -1)

        # 模型预测
        prediction = stacking_classifier.predict(input_array)[0]
        prediction_proba = stacking_classifier.predict_proba(input_array)[0]
        
        # 模型的类别标签是1,2,3,4，需要映射到概率数组索引
        # prediction_proba[0] 对应亚型1, prediction_proba[1] 对应亚型2, 以此类推
        
        # 亚型映射及1年内急性加重再住院率（使用1-4作为键，与模型预测结果对应）
        # 风险等级为三级分类：高风险、中风险、低风险
        subtype_info = {
            1: {
                "name": "亚型1", 
                "readmission_rate": 19.2, 
                "risk_level": "高风险",
                "risk_color": "🔴",
                "description": "痰热壅肺证合并发热型 - 高风险亚型，需要密切随访和积极干预"
            },
            2: {
                "name": "亚型2", 
                "readmission_rate": 14.5, 
                "risk_level": "中风险",
                "risk_color": "🟡",
                "description": "无痰热壅肺证及发热型 - 中风险亚型，建议定期随访和预防性治疗"
            },
            3: {
                "name": "亚型3", 
                "readmission_rate": 14.0, 
                "risk_level": "中风险",
                "risk_color": "🟡",
                "description": "发热无痰热壅肺证型 - 中风险亚型，建议定期随访和预防性治疗"
            },
            4: {
                "name": "亚型4", 
                "readmission_rate": 10.1, 
                "risk_level": "低风险",
                "risk_color": "🟢",
                "description": "痰热壅肺证无发热型 - 低风险亚型，建议常规随访"
            }
        }

        # 预测结果即为亚型编号(1-4)
        predicted_subtype = prediction
        predicted_proba = prediction_proba[predicted_subtype - 1]  # 概率数组索引从0开始

        # 显示预测结果
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.success(f"### 预测亚型：{subtype_info[predicted_subtype]['name']}")
            st.metric(
                label="预测概率", 
                value=f"{predicted_proba*100:.2f}%"
            )
        
        with col2:
            st.info(f"### 1年内急性加重再住院率")
            st.metric(
                label="再住院风险", 
                value=f"{subtype_info[predicted_subtype]['readmission_rate']}%"
            )
        
        with col3:
            st.warning("### 风险等级")
            # 【修复】使用subtype_info中定义的风险等级和颜色
            risk_level = subtype_info[predicted_subtype]['risk_level']
            risk_color = subtype_info[predicted_subtype]['risk_color']
            st.metric(
                label="评估", 
                value=f"{risk_color} {risk_level}"
            )
        
        # 临床建议
        st.info(f"**临床建议：** {subtype_info[predicted_subtype]['description']}")
        
        st.markdown("---")
        
        # 各亚型概率分布
        st.subheader("📈 各亚型预测概率分布")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.write("**各亚型预测概率**")
            proba_df = pd.DataFrame({
                '亚型': [subtype_info[i]['name'] for i in range(1, 5)],
                '概率(%)': prediction_proba * 100
            })
            st.bar_chart(proba_df.set_index('亚型'))
        
        with col_chart2:
            st.write("**各亚型再住院率对比**")
            readmission_df = pd.DataFrame({
                '亚型': [subtype_info[i]['name'] for i in range(1, 5)],
                '再住院率(%)': [subtype_info[i]['readmission_rate'] for i in range(1, 5)]
            })
            st.bar_chart(readmission_df.set_index('亚型'))
        
        # 详细概率表格
        st.subheader("📋 详细预测概率与再住院率")
        proba_table = pd.DataFrame({
            '亚型': [subtype_info[i]['name'] for i in range(1, 5)],
            '预测概率': [f"{p*100:.2f}%" for p in prediction_proba],
            '1年内急性加重再住院率': [f"{subtype_info[i]['readmission_rate']}%" for i in range(1, 5)],
            '风险等级': [subtype_info[i]['risk_level'] for i in range(1, 5)],
            '临床建议': [subtype_info[i]['description'] for i in range(1, 5)]
        })
        
        # 高亮显示预测亚型
        def highlight_predicted(row):
            if row['亚型'] == subtype_info[predicted_subtype]['name']:
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        styled_table = proba_table.style.apply(highlight_predicted, axis=1)
        st.dataframe(styled_table, use_container_width=True)
        
        st.markdown("---")
        
        # 特征重要性提示
        st.subheader("🔍 关键特征说明")
        
        col_feature1, col_feature2 = st.columns(2)
        
        with col_feature1:
            st.info("""
            **理化指标（4项）：**
            
            1. **平均血红蛋白量 (MCH)**
               - 当前值: {:.2f} pg
               - 正常范围: 27-34 pg
               - 临床意义: 反映红细胞携氧能力
            
            2. **载脂蛋白A (ApoA)**
               - 当前值: {:.2f} g/L
               - 正常范围: 1.0-1.6 g/L
               - 临床意义: 心血管保护因子
            
            3. **尿酸**
               - 当前值: {:.2f} μmol/L
               - 正常范围: 男性208-428, 女性155-357
               - 临床意义: 代谢指标，炎症标志物
            
            4. **FVC占预计值的百分比**
               - 当前值: {:.2f}%
               - 正常范围: ≥80%
               - 临床意义: 肺功能通气储备能力
            """.format(MCH, apoA, uric_acid, FVC))
        
        with col_feature2:
            st.info("""
            **中医证候信息（2项）：**
            
            1. **发热**
               - 当前状态: {}
               - 临床意义: 急性感染标志
               - 亚型关联: 亚型1和亚型3的关键特征
            
            2. **痰热壅肺证**
               - 当前状态: {}
               - 临床意义: 中医辨证分型
               - 亚型关联: 亚型1和亚型4的关键特征
            
            ---
            
            **亚型快速识别：**
            - 发热(+) + 痰热壅肺证(+) → 倾向亚型1 (高风险)
            - 发热(-) + 痰热壅肺证(-) → 倾向亚型2 (中风险)
            - 发热(+) + 痰热壅肺证(-) → 倾向亚型3 (中风险)
            - 发热(-) + 痰热壅肺证(+) → 倾向亚型4 (低风险)
            """.format(
                "有" if fever == 1 else "无",
                "有" if tan_re == 1 else "无"
            ))
        
    except Exception as e:
        st.error(f"❌ 预测时发生错误：{e}")
        st.exception(e)

# 输入特征汇总
with st.expander("📝 查看当前输入的特征值"):
    input_summary = pd.DataFrame({
        '特征名称': [
            '平均血红蛋白量', 
            '载脂蛋白A', 
            '尿酸', 
            'FVC占预计值的百分比',
            '发热', 
            '痰热壅肺证'
        ],
        '输入值': [
            f"{MCH:.2f}",
            f"{apoA:.2f}",
            f"{uric_acid:.2f}",
            f"{FVC:.2f}",
            "有" if fever == 1 else "无",
            "有" if tan_re == 1 else "无"
        ],
        '单位': [
            'pg',
            'g/L',
            'μmol/L',
            '%',
            '-',
            '-'
        ],
        '正常范围': [
            '27-34',
            '1.0-1.6',
            '男208-428/女155-357',
            '≥80',
            '-',
            '-'
        ]
    })
    st.dataframe(input_summary, use_container_width=True)

# 亚型特征说明
with st.expander("ℹ️ 各亚型特征及临床意义"):
    st.markdown("""
    ### 各亚型1年内急性加重再住院率
    
    | 亚型 | 再住院率 | 风险等级 | 主要特征 | 临床建议 |
    |------|----------|----------|----------|----------|
    | 亚型1 | **19.2%** | 🔴 高风险 | 痰热壅肺证(+) + 发热(+) | 需要密切随访和积极干预，建议2-4周复诊 |
    | 亚型2 | 14.5% | 🟡 中风险 | 痰热壅肺证(-) + 发热(-) | 定期随访和预防性治疗，建议1-2月复诊 |
    | 亚型3 | 14.0% | 🟡 中风险 | 痰热壅肺证(-) + 发热(+) | 定期随访和预防性治疗，建议1-2月复诊 |
    | 亚型4 | **10.1%** | 🟢 低风险 | 痰热壅肺证(+) + 发热(-) | 常规随访，建议3-6月复诊 |
    
    ### 各亚型详细特征
    
    #### 🔴 亚型1（痰热壅肺证合并发热型）- 高风险
    **主要特征：**
    - 出现发热及痰热壅肺证的概率高达98%以上
    - 平均血红蛋白量水平偏低
    - 血小板水平偏高
    - 存在限制性通气功能障碍
    
    ---
    
    #### 🟡 亚型2（无痰热壅肺证及发热型）- 中风险
    **主要特征：**
    - 无发热及痰热壅肺证
    - 年龄及收缩压水平偏高
    - 存在限制性通气功能障碍
    
    ---
    
    #### 🟡 亚型3（发热无痰热壅肺证型）- 中风险
    **主要特征：**
    - 出现发热但未诊断为痰热壅肺证
    - 年龄偏高
    - 体重、载脂蛋白A及尿酸水平偏低
    - 存在限制性通气功能障碍
    
    ---
    
    #### 🟢 亚型4（痰热壅肺证无发热型）- 低风险
    **主要特征：**
    - 诊断为痰热壅肺证但无发热
    - 体重及尿酸水平偏高
    - 无限制性通气功能障碍
    
    ---
    
    ### 注意事项
    
    ⚠️ **重要提示：**
    - 本预测结果仅供临床参考，不能替代医生的专业判断
    - 建议结合患者的病史、体格检查、影像学检查等综合评估
    - 对于高风险患者，应制定个性化的随访和治疗方案
    
    📋 **所有患者的通用建议：**
    - 戒烟，避免二手烟
    - 避免接触有害气体和粉尘
    - 接种流感疫苗和肺炎疫苗
    - 规律使用吸入药物
    - 进行肺康复训练
    - 保持良好的营养状态
    - 及时就医，预防急性加重
    """)

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 如何使用本系统
    
    #### 第一步：输入特征值
    在左侧边栏依次输入患者的**6项关键临床特征**：
    
    **理化指标（4项）：**
    1. 平均血红蛋白量 (pg) - 输入具体数值
    2. 载脂蛋白A (g/L) - 输入具体数值
    3. 尿酸 (μmol/L) - 输入具体数值
    4. FVC占预计值的百分比 (%) - 输入具体数值
    
    **中医证候信息（2项）：**
    5. 发热 - 选择"无"或"有"
    6. 痰热壅肺证 - 选择"无"或"有"
    
    #### 第二步：开始预测
    点击左侧边栏底部的 **"🔮 开始预测"** 按钮
    
    #### 第三步：查看结果
    系统将显示以下内容：
    - ✅ 预测亚型及概率
    - ✅ 1年内再住院风险评估
    - ✅ 各亚型概率分布图
    - ✅ 详细的临床建议
    - ✅ 关键特征解读
    
    #### 第四步：临床决策
    根据预测结果，结合患者实际情况：
    - 制定个性化治疗方案
    - 安排合理的随访计划
    - 进行针对性的健康教育
    - 实施预防性干预措施
    
    ---
    
    ### 模型性能指标
    
    本模型在数据集上进行了严格验证：
    
    | 数据集 | Micro-AUC | Macro-AUC | 样本量 |
    |--------|-----------|-----------|--------|
    | 训练集 | 1.000 | 1.000 | 908例 |
    | 测试集 | 0.974 | 0.972 | 389例 |
    | 时段验证集 | 0.958 | 0.940 | 239例 |
    
    **性能评价：**
    - 训练集表现完美，说明模型拟合良好
    - 测试集和时段验证集AUC均>0.94，表明模型泛化能力强
    - Micro-AUC和Macro-AUC接近，表明模型对各亚型预测均衡
    
    ---
    
    ### 特征获取指南
    
    #### 理化指标如何获取：
    
    1. **平均血红蛋白量 (MCH)**
       - 来源：血常规检查
       - 英文名：Mean Corpuscular Hemoglobin
       - 单位：pg
    
    2. **载脂蛋白A (ApoA)**
       - 来源：血脂检查
       - 英文名：Apolipoprotein A
       - 单位：g/L
    
    3. **尿酸**
       - 来源：生化检查
       - 英文名：Uric Acid (UA)
       - 单位：μmol/L
    
    4. **FVC占预计值的百分比**
       - 来源：肺功能检查
       - 英文名：FVC % predicted
       - 单位：%
    
    #### 中医证候如何判断：
    
    1. **发热**
       - 体温升高
       - 患者自述发热感
       - 伴有恶寒、出汗等症状
    
    2. **痰热壅肺证**
       - 需要由中医师进行辨证
       - 主症：咳嗽，喘息，胸闷，痰多、色黄，咯痰不爽，舌质红，舌苔黄，舌苔腻，脉数，脉滑
       - 次症：痰黏，胸痛，发热，口渴喜冷饮，大便干结，舌苔厚
       - 诊断：①咳嗽或喘息；②痰黏、色黄，咯痰不爽；③发热或口渴喜冷饮；④大便干结；⑤舌质红，舌苔黄或黄腻或黄厚腻，脉数或滑数。具备①、②2项，加③、④、⑤中的2项
       - 如无中医师，可参考上述标准初步判断[引用来源：李建生等. 中华中医药杂志, 2010.25(7):971-975]
    
    ---
    
    ### 技术支持
    
    如有任何问题或建议，请联系：
    - 📧 邮箱：[wangj1758@163.com]
    - 📱 电话：[1573196323]
    - 🏥 科室：[呼吸科]
    """)

# 模型信息
st.sidebar.markdown("---")
st.sidebar.info("""
**模型信息**
- **模型类型**：Stacking集成学习
- **基学习器**：RF, XGB, LGBM, GBM, AdaBoost, CatBoost
- **元学习器**：Logistic Regression
- **特征数量**：6个关键特征
  - 理化指标：4项
  - 中医证候：2项
- **预测类别**：4个亚型
- **性能指标**：
  - 测试集AUC: 0.974
  - 时段验证集AUC: 0.958
""")

# 快速参考卡片
st.sidebar.markdown("---")
st.sidebar.success("""
**快速参考**

🔴 **高风险 (亚型1)**
- 再住院率：19.2%
- 特征：发热(+) + 痰热壅肺证(+)

🟡 **中风险 (亚型2、3)**
- 再住院率：14-14.5%

🟢 **低风险 (亚型4)**
- 再住院率：10.1%
- 特征：发热(-) + 痰热壅肺证(+)
""")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚕️ AECOPD亚型预测系统 (6特征精简版)</strong></p>
    <p>基于机器学习的临床决策支持工具</p>
    <p style='font-size: 12px; color: gray; margin-top: 10px;'>
        <strong>免责声明：</strong>本系统仅供研究和辅助决策使用，不能替代专业医疗建议。<br>
        所有预测结果应由专业医生结合临床实际情况进行综合判断。
    </p>
    <p style='font-size: 10px; color: gray; margin-top: 5px;'>
        版本: 1.0 | 更新日期: 2025-01-20 | 特征数: 6个关键特征
    </p>
</div>
""", unsafe_allow_html=True)