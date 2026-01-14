import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 加载模型
model_path = "stacking_Classifier_model.pkl"
stacking_classifier = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(
    layout="wide", 
    page_title="AECOPD亚型预测系统", 
    page_icon="🏥"
)

st.title("🏥 AECOPD出院后1年内急性加重再住院亚型预测系统")
st.write("""
基于Stacking集成学习模型，预测AECOPD患者出院后1年内急性加重再住院的6个亚型。
本系统整合了13个关键临床特征，并结合SHAP可解释性分析。
""")

# 左侧侧边栏输入区域
st.sidebar.header("📋 临床特征输入")
st.sidebar.write("请输入患者的临床特征值：")

# 定义特征输入（连续变量）
st.sidebar.subheader("连续型变量")
FEV = st.sidebar.number_input(
    "FEV1%/FVC最佳预计值 (%)", 
    min_value=26.81, 
    max_value=102.87, 
    value=65.0,
    help="范围: 26.81-102.87"
)

BMI = st.sidebar.number_input(
    "体重指数 BMI (kg/m²)", 
    min_value=8.65, 
    max_value=40.53, 
    value=22.0,
    help="范围: 8.65-40.53"
)

HDL = st.sidebar.number_input(
    "高密度脂蛋白胆固醇 (mmol/L)", 
    min_value=0.35, 
    max_value=2.61, 
    value=1.2,
    help="范围: 0.35-2.61"
)

Mg = st.sidebar.number_input(
    "镁 (mmol/L)", 
    min_value=0.35, 
    max_value=2.26, 
    value=0.9,
    help="范围: 0.35-2.26"
)

RBC = st.sidebar.number_input(
    "红细胞计数 (×10¹²/L)", 
    min_value=0.0, 
    max_value=249.19, 
    value=4.5,
    help="范围: 0-249.19"
)

SBP = st.sidebar.number_input(
    "收缩压 (mmHg)", 
    min_value=0.0, 
    max_value=222.0, 
    value=120.0,
    help="范围: 0-222"
)

# 定义特征输入（二分类变量）
st.sidebar.subheader("二分类变量 (0=无，1=有)")

mai_shu = st.sidebar.selectbox(
    "脉数", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

jingshen_weimi = st.sidebar.selectbox(
    "精神萎靡", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

mai_hua = st.sidebar.selectbox(
    "脉滑", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

tai_bai = st.sidebar.selectbox(
    "苔白", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

xiong_men = st.sidebar.selectbox(
    "胸闷", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

chuan_xi = st.sidebar.selectbox(
    "喘息", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

mai_chen = st.sidebar.selectbox(
    "脉沉", 
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
        # 注意：这里的顺序需要与训练模型时final_selected_features的顺序一致
        input_array = np.array([
            FEV,           # FEV.1.%.FVC_BEST/PRED
            BMI,           # BMI
            HDL,           # 高密度脂蛋白胆固醇
            Mg,            # 镁
            RBC,           # 红细胞计数
            SBP,           # 收缩压
            mai_shu,       # 脉数
            jingshen_weimi,# 精神萎靡
            mai_hua,       # 脉滑
            tai_bai,       # 苔白
            xiong_men,     # 胸闷
            chuan_xi,      # 喘息
            mai_chen       # 脉沉
        ]).reshape(1, -1)

        # 模型预测
        prediction = stacking_classifier.predict(input_array)[0]
        prediction_proba = stacking_classifier.predict_proba(input_array)[0]

        # 亚型映射
        subtype_names = {
            0: "亚型1",
            1: "亚型2", 
            2: "亚型3",
            3: "亚型4",
            4: "亚型5",
            5: "亚型6"
        }

        # 显示预测结果
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success(f"### 预测亚型：{subtype_names.get(prediction, '未知')}")
            st.metric(
                label="预测置信度", 
                value=f"{prediction_proba[prediction]*100:.2f}%"
            )
        
        with col2:
            st.subheader("各亚型预测概率分布")
            proba_df = pd.DataFrame({
                '亚型': [subtype_names[i] for i in range(len(prediction_proba))],
                '概率': prediction_proba * 100
            })
            st.bar_chart(proba_df.set_index('亚型'))
        
        # 详细概率表格
        st.subheader("详细预测概率")
        proba_table = pd.DataFrame({
            '亚型': [subtype_names[i] for i in range(len(prediction_proba))],
            '预测概率 (%)': [f"{p*100:.2f}%" for p in prediction_proba]
        })
        st.dataframe(proba_table, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 预测时发生错误：{e}")
        st.exception(e)

# 输入特征汇总
with st.expander("📝 查看当前输入的特征值"):
    input_summary = pd.DataFrame({
        '特征名称': [
            'FEV1%/FVC最佳预计值', 'BMI', '高密度脂蛋白胆固醇', '镁', 
            '红细胞计数', '收缩压', '脉数', '精神萎靡', '脉滑', 
            '苔白', '胸闷', '喘息', '脉沉'
        ],
        '输入值': [
            FEV, BMI, HDL, Mg, RBC, SBP, 
            mai_shu, jingshen_weimi, mai_hua, tai_bai, 
            xiong_men, chuan_xi, mai_chen
        ]
    })
    st.dataframe(input_summary, use_container_width=True)

# 模型信息
st.sidebar.markdown("---")
st.sidebar.info("""
**模型信息**
- 模型类型：Stacking集成学习
- 基学习器：RF, XGB, LGBM, GBM, AdaBoost, CatBoost
- 元学习器：Logistic Regression
- 特征数量：13个
- 预测类别：6个亚型
""")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚕️ AECOPD亚型预测系统 | 基于机器学习的临床决策支持工具</p>
    <p style='font-size: 12px; color: gray;'>
        免责声明：本系统仅供研究和辅助决策使用，不能替代专业医疗建议
    </p>
</div>
""", unsafe_allow_html=True)