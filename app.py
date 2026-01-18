import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
@st.cache_resource
def load_model():
    try:
        model_path = "stacking_Classifier_model.pkl"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

stacking_classifier = load_model()

# 设置页面配置和标题
st.set_page_config(
    layout="wide", 
    page_title="AECOPD亚型预测系统", 
    page_icon="🏥"
)

st.title("🏥 AECOPD出院后1年内急性加重再住院亚型预测系统")
st.write("""
基于Stacking集成学习模型,预测AECOPD患者出院后1年内急性加重再住院的4个亚型。
本系统整合了12个关键临床特征,并结合SHAP可解释性分析。
""")

# 左侧侧边栏输入区域
st.sidebar.header("📋 临床特征输入")
st.sidebar.write("请输入患者的临床特征值：")

# 定义特征输入（连续变量）
st.sidebar.subheader("连续型变量")

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
    value=0.5,
    help="范围: 0.0-16.5"
)

# 定义特征输入（二分类变量）
st.sidebar.subheader("二分类变量 (0=无，1=有)")

fever = st.sidebar.selectbox(
    "发热", 
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

tan_re_yong_fei = st.sidebar.selectbox(
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

# 特征名称列表（用于SHAP展示）
feature_names = [
    'FVC最佳预计值', '发热', '痰热壅肺证', '尿酸', 
    '载脂蛋白A', '痰黄', '镁', '平均血红蛋白量', 
    '苔白', '嗜碱性粒细胞比率', '舌暗', '咳嗽'
]

# 主页面用于结果展示
if predict_button:
    if stacking_classifier is None:
        st.error("模型未能成功加载，请检查模型文件是否存在。")
    else:
        st.header("📊 预测结果")
        
        try:
            # 将输入特征转换为模型所需格式
            input_array = np.array([
                FVC,           # FVC_BEST/PRED
                fever,         # 发热
                tan_re_yong_fei,  # 痰热壅肺证
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
            
            # 亚型映射及再住院率
            subtype_info = {
                0: {"name": "亚型1", "readmission_rate": 19.2},
                1: {"name": "亚型2", "readmission_rate": 14.5},
                2: {"name": "亚型3", "readmission_rate": 14.0},
                3: {"name": "亚型4", "readmission_rate": 10.1}
            }
            
            # 显示预测结果
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.success(f"### 预测亚型：{subtype_info[prediction]['name']}")
                st.metric(
                    label="预测置信度", 
                    value=f"{prediction_proba[prediction]*100:.2f}%"
                )
            
            with col2:
                st.info(f"### 1年再住院率")
                st.metric(
                    label="历史统计数据", 
                    value=f"{subtype_info[prediction]['readmission_rate']}%"
                )
            
            with col3:
                st.subheader("各亚型预测概率分布")
                proba_df = pd.DataFrame({
                    '亚型': [subtype_info[i]['name'] for i in range(len(prediction_proba))],
                    '概率': prediction_proba * 100
                })
                st.bar_chart(proba_df.set_index('亚型'))
            
            # 详细概率表格（包含再住院率）
            st.subheader("详细预测概率及再住院率信息")
            proba_table = pd.DataFrame({
                '亚型': [subtype_info[i]['name'] for i in range(len(prediction_proba))],
                '预测概率': [f"{p*100:.2f}%" for p in prediction_proba],
                '1年再住院率': [f"{subtype_info[i]['readmission_rate']}%" for i in range(len(prediction_proba))]
            })
            
            # 高亮显示预测的亚型
            def highlight_prediction(row):
                if row['亚型'] == subtype_info[prediction]['name']:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            styled_table = proba_table.style.apply(highlight_prediction, axis=1)
            st.dataframe(styled_table, use_container_width=True)
            
            # SHAP可解释性分析
            st.header("🔍 SHAP可解释性分析")
            st.write("以下分析展示了各特征对预测结果的影响程度：")
            
            try:
                with st.spinner('正在生成SHAP分析图...'):
                    # 创建SHAP解释器
                    explainer = shap.TreeExplainer(stacking_classifier)
                    shap_values = explainer.shap_values(input_array)
                    
                    # 如果是多分类，shap_values是列表
                    if isinstance(shap_values, list):
                        shap_values_for_prediction = shap_values[prediction]
                    else:
                        shap_values_for_prediction = shap_values
                    
                    # 创建SHAP force plot
                    st.subheader(f"对{subtype_info[prediction]['name']}预测的特征贡献")
                    
                    # Waterfall plot (更清晰的可视化)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=shap_values_for_prediction[0],
                            base_values=explainer.expected_value[prediction] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                            data=input_array[0],
                            feature_names=feature_names
                        ),
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    # 特征重要性条形图
                    st.subheader("特征重要性排序")
                    feature_importance = pd.DataFrame({
                        '特征': feature_names,
                        '贡献值': np.abs(shap_values_for_prediction[0])
                    }).sort_values('贡献值', ascending=False)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.barh(feature_importance['特征'], feature_importance['贡献值'])
                    ax2.set_xlabel('SHAP值绝对值')
                    ax2.set_title('特征对预测结果的影响程度')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                    
                    # 显示数值表格
                    st.dataframe(feature_importance, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"SHAP分析生成失败: {e}")
                st.info("注意：SHAP分析需要模型支持TreeExplainer，请确保使用的是树模型。")
        
        except Exception as e:
            st.error(f"❌ 预测时发生错误：{e}")
            st.exception(e)

# 输入特征汇总
with st.expander("📝 查看当前输入的特征值"):
    input_summary = pd.DataFrame({
        '特征名称': feature_names,
        '输入值': [
            FVC, fever, tan_re_yong_fei, uric_acid, 
            apoA, tan_huang, Mg, MCH, 
            tai_bai, basophil, she_an, cough
        ]
    })
    st.dataframe(input_summary, use_container_width=True)

# 亚型再住院率信息展示
with st.expander("📈 各亚型1年再住院率统计"):
    readmission_df = pd.DataFrame({
        '亚型': ['亚型1', '亚型2', '亚型3', '亚型4'],
        '1年再住院率 (%)': [19.2, 14.5, 14.0, 10.1]
    })
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(readmission_df, use_container_width=True)
    with col2:
        st.bar_chart(readmission_df.set_index('亚型'))

# 模型信息
st.sidebar.markdown("---")
st.sidebar.info("""
**模型信息**
- 模型类型：Stacking集成学习
- 基学习器：RF, XGB, LGBM, GBM, AdaBoost, CatBoost
- 元学习器：Logistic Regression
- 特征数量：12个
- 预测类别：4个亚型

**亚型再住院率**
- 亚型1: 19.2%
- 亚型2: 14.5%
- 亚型3: 14.0%
- 亚型4: 10.1%
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