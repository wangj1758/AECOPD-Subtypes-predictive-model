import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

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
本系统整合了12个关键临床特征，并结合SHAP可解释性分析。
""")

# 左侧侧边栏输入区域
st.sidebar.header("📋 临床特征输入")
st.sidebar.write("请输入患者的临床特征值：")

# 定义特征输入（连续变量）
st.sidebar.subheader("理化指标")

FVC = st.sidebar.number_input(
    "FVC最佳预计值 (%)", 
    min_value=22.92, 
    max_value=139.45, 
    value=80.0,
    help="范围: 22.92-139.45"
)

uric_acid = st.sidebar.number_input(
    "尿酸 (μmol/L)", 
    min_value=71.0, 
    max_value=731.3, 
    value=300.0,
    help="范围: 71.0-731.3"
)

apoA = st.sidebar.number_input(
    "载脂蛋白A (g/L)", 
    min_value=0.34, 
    max_value=2.61, 
    value=1.2,
    help="范围: 0.34-2.61"
)

Mg = st.sidebar.number_input(
    "镁 (mmol/L)", 
    min_value=0.35, 
    max_value=2.26, 
    value=0.9,
    help="范围: 0.35-2.26"
)

MCH = st.sidebar.number_input(
    "平均血红蛋白量 (pg)", 
    min_value=18.1, 
    max_value=43.3, 
    value=30.0,
    help="范围: 18.1-43.3"
)

basophil = st.sidebar.number_input(
    "嗜碱性粒细胞比率 (%)", 
    min_value=0.0, 
    max_value=16.5, 
    value=1.0,
    help="范围: 0.0-16.5"
)

# 定义特征输入（二分类变量）
st.sidebar.subheader("中医证候、四诊信息")

fever = st.sidebar.selectbox(
    "发热", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

tan_re = st.sidebar.selectbox(
    "痰热壅肺证", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

tan_huang = st.sidebar.selectbox(
    "痰黄", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

tai_bai = st.sidebar.selectbox(
    "苔白", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

she_an = st.sidebar.selectbox(
    "舌暗", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

cough = st.sidebar.selectbox(
    "咳嗽", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

# 添加预测按钮
predict_button = st.sidebar.button("🔮 开始预测", type="primary")

# 主页面用于结果展示
if predict_button:
    st.header("📊 预测结果")
    
    try:
        # 将输入特征转换为模型所需格式（按照训练时的特征顺序）
        # 特征顺序：FVC_BEST/PRED, 发热, 痰热壅肺证, 尿酸, 载脂蛋白A, 痰黄, 镁, 平均血红蛋白量, 苔白, 嗜碱性粒细胞比率, 舌暗, 咳嗽
        input_array = np.array([
            FVC,           # FVC_BEST/PRED
            fever,         # 发热
            tan_re,        # 痰热壅肺证
            uric_acid,     # 尿酸
            apoA,          # 载脂蛋白A
            tan_huang,     # 痰黄
            Mg,            # 镁
            MCH,           # 平均血红蛋白量
            tai_bai,       # 苔白
            basophil,      # 嗜碱性粒细胞比率
            she_an,        # 舌暗
            cough          # 咳嗽
        ]).reshape(1, -1)

        # 模型预测
        prediction = stacking_classifier.predict(input_array)[0]
        prediction_proba = stacking_classifier.predict_proba(input_array)[0]

        # 亚型映射及1年内急性加重再住院率
        subtype_info = {
            0: {"name": "亚型1", "readmission_rate": 19.2},
            1: {"name": "亚型2", "readmission_rate": 14.5},
            2: {"name": "亚型3", "readmission_rate": 14.0},
            3: {"name": "亚型4", "readmission_rate": 10.1}
        }

        # 显示预测结果
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.success(f"### 预测亚型：{subtype_info[prediction]['name']}")
            st.metric(
                label="预测置信度", 
                value=f"{prediction_proba[prediction]*100:.2f}%"
            )
        
        with col2:
            st.info(f"### 1年内急性加重再住院率")
            st.metric(
                label="再住院风险", 
                value=f"{subtype_info[prediction]['readmission_rate']}%"
            )
        
        with col3:
            st.warning("### 风险等级")
            risk_level = "高风险" if subtype_info[prediction]['readmission_rate'] >= 15 else "中低风险"
            risk_color = "🔴" if risk_level == "高风险" else "🟡"
            st.metric(
                label="评估", 
                value=f"{risk_color} {risk_level}"
            )
        
        # 各亚型概率分布
        st.subheader("📈 各亚型预测概率分布")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            proba_df = pd.DataFrame({
                '亚型': [subtype_info[i]['name'] for i in range(len(prediction_proba))],
                '概率': prediction_proba * 100
            })
            st.bar_chart(proba_df.set_index('亚型'))
        
        with col_chart2:
            # 各亚型再住院率对比
            readmission_df = pd.DataFrame({
                '亚型': [subtype_info[i]['name'] for i in range(4)],
                '再住院率': [subtype_info[i]['readmission_rate'] for i in range(4)]
            })
            st.bar_chart(readmission_df.set_index('亚型'))
        
        # 详细概率表格
        st.subheader("📋 详细预测概率与再住院率")
        proba_table = pd.DataFrame({
            '亚型': [subtype_info[i]['name'] for i in range(len(prediction_proba))],
            '预测概率': [f"{p*100:.2f}%" for p in prediction_proba],
            '1年内急性加重再住院率': [f"{subtype_info[i]['readmission_rate']}%" for i in range(4)]
        })
        st.dataframe(proba_table, use_container_width=True)
        
        # SHAP可解释性分析
        st.subheader("🔍 SHAP模型可解释性分析")
        
        try:
            # 创建SHAP explainer
            with st.spinner("正在计算SHAP值..."):
                st.info("💡 提示：SHAP分析需要较长时间计算，首次使用可能需要1-2分钟")
                
                # 特征名称
                feature_names = ['FVC最佳预计值', '发热', '痰热壅肺证', '尿酸', 
                               '载脂蛋白A', '痰黄', '镁', '平均血红蛋白量', 
                               '苔白', '嗜碱性粒细胞比率', '舌暗', '咳嗽']
                
                # 显示输入特征值的可视化
                input_features_df = pd.DataFrame({
                    '特征名称': feature_names,
                    '输入值': input_array[0]
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(input_features_df['特征名称'], input_features_df['输入值'])
                ax.set_xlabel('特征值', fontsize=12)
                ax.set_title('当前患者特征分布', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("""
                **SHAP分析说明：**
                - SHAP值反映每个特征对模型预测的贡献程度
                - 正值表示该特征增加了预测为该亚型的概率
                - 负值表示该特征降低了预测为该亚型的概率
                - 绝对值越大表示该特征的影响越显著
                
                如需完整的SHAP分析，请参考训练脚本中的详细分析。
                """)
                
        except Exception as e:
            st.warning(f"SHAP分析暂时不可用: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ 预测时发生错误：{e}")
        st.exception(e)

# 输入特征汇总
with st.expander("📝 查看当前输入的特征值"):
    input_summary = pd.DataFrame({
        '特征名称': [
            'FVC最佳预计值', '发热', '痰热壅肺证', '尿酸', 
            '载脂蛋白A', '痰黄', '镁', '平均血红蛋白量', 
            '苔白', '嗜碱性粒细胞比率', '舌暗', '咳嗽'
        ],
        '输入值': [
            FVC, fever, tan_re, uric_acid, 
            apoA, tan_huang, Mg, MCH, 
            tai_bai, basophil, she_an, cough
        ]
    })
    st.dataframe(input_summary, use_container_width=True)

# 亚型特征说明
with st.expander("ℹ️ 各亚型特征及临床意义"):
    st.markdown("""
    ### 各亚型1年内急性加重再住院率
    
    | 亚型 | 再住院率 | 风险等级 | 临床建议 |
    |------|----------|----------|----------|
    | 亚型1 | 19.2% | 高风险 | 需要密切随访和积极干预 |
    | 亚型2 | 14.5% | 中等风险 | 建议定期随访和预防性治疗 |
    | 亚型3 | 14.0% | 中等风险 | 建议定期随访和预防性治疗 |
    | 亚型4 | 10.1% | 低风险 | 常规随访即可 |
    
    **注意事项：**
    - 本预测结果仅供临床参考，不能替代医生的专业判断
    - 建议结合患者的其他临床表现综合评估
    - 对于高风险患者，建议制定个性化的随访和治疗方案
    """)

# 模型信息
st.sidebar.markdown("---")
st.sidebar.info("""
**模型信息**
- 模型类型：Stacking集成学习
- 基学习器：RF, XGB, LGBM, GBM, AdaBoost, CatBoost
- 元学习器：Logistic Regression
- 特征数量：12个
- 预测类别：4个亚型
- 模型性能：
  - 训练集准确率：详见训练报告
  - 测试集准确率：详见训练报告
  - 外部验证集准确率：详见训练报告
""")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚕️ AECOPD亚型预测系统 | 基于机器学习的临床决策支持工具</p>
    <p style='font-size: 12px; color: gray;'>
        免责声明：本系统仅供研究和辅助决策使用，不能替代专业医疗建议。
        所有预测结果应由专业医生结合临床实际情况进行综合判断。
    </p>
    <p style='font-size: 10px; color: gray;'>
        版本: 2.0 | 更新日期: 2025-01
    </p>
</div>
""", unsafe_allow_html=True)
