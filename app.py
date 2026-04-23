import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ============================================================================
# 1. 模型加载与页面配置
# ============================================================================

model_path = "stacking_Classifier_model.pkl"
try:
    stacking_classifier = joblib.load(model_path)
    st.success("✓ 模型加载成功")
except FileNotFoundError:
    st.error("⚠️ 模型文件未找到，请确保 stacking_Classifier_model.pkl 在同一目录下")
    st.stop()
except Exception as e:
    st.error(f"⚠️ 模型加载失败: {e}")
    st.stop()

st.set_page_config(
    layout="wide",
    page_title="AECOPD亚型预测系统",
    page_icon="🏥"
)

# ============================================================================
# 2. 亚型信息定义
# ============================================================================

SUBTYPE_INFO = {
    1: {
        "name": "亚型1",
        "readmission_rate": 17.9,
        "risk_level": "高风险",
        "risk_color": "🔴",
        "main_features": "以老年患者为主；中医证候特异性表现为痰湿阻肺证、血瘀证及痰瘀阻肺证",
        "secondary_features": (
            "症状涉及神疲、胸痛、痛苦面容、尿频及失眠，舌脉以舌紫、舌暗、苔白、脉弦、脉沉及脉细为主。"
            "特异性合并肺癌、肺纤维化、脑血管疾病及胸腔积液。"
            "生理生化方面D-二聚体、纤维蛋白原、单核细胞比率及凝血酶原时间偏高，"
            "血红蛋白及血小板偏低。"
            "CT影像LAA_950_pct低至中水平，Normal_pct极高，Con_pct高至极高。"
        ),
        "followup": "2-4周复诊"
    },
    2: {
        "name": "亚型2",
        "readmission_rate": 11.5,
        "risk_level": "低风险",
        "risk_color": "🟢",
        "main_features": "以肥胖、超重为主，肺功能分级为GOLD 2级；中医证候特异性表现为痰热壅肺证",
        "secondary_features": (
            "症状以咳嗽、咳痰、咯血、痰色黄、发热及头晕为主，舌脉以舌红、苔黄、苔腻、脉滑及脉数为主。"
            "特异性合并支气管扩张及糖尿病。"
            "生理生化方面C反应蛋白及乳酸脱氢酶偏高，高密度脂蛋白胆固醇、血红蛋白及凝血酶原时间偏低。"
            "CT影像Con_pct极高，LAA_950_pct低至中水平。"
        ),
        "followup": "3-6月复诊"
    },
    3: {
        "name": "亚型3",
        "readmission_rate": 13.0,
        "risk_level": "中风险",
        "risk_color": "🟡",
        "main_features": "以男性、吸烟、饮酒及体重过低为主，肺功能分级为GOLD 3～4级；无特异性证候富集",
        "secondary_features": (
            "症状涉及痰粘难咯、面色黄、神疲、汗多及心悸，舌脉以苔厚为特异性表现。"
            "特异性合并肺气肿。"
            "生理生化方面红细胞压积、血红蛋白及高密度脂蛋白胆固醇偏高，球蛋白偏低。"
            "CT影像LAA_950_pct高至极高，Normal_pct及Con_pct均处于低至中水平。"
        ),
        "followup": "1-2月复诊"
    }
}

# ============================================================================
# 3. 页面标题
# ============================================================================

st.title("🏥 AECOPD出院后1年内急性加重再住院亚型预测系统")
st.write("""
基于Stacking集成学习模型，预测AECOPD患者出院后1年内急性加重再住院的**3个亚型**。
本系统整合了**15项关键临床特征**（一般信息、影像特征、肺功能、生理生化、中医证候及舌脉），为临床决策提供辅助支持。
""")

# ============================================================================
# 4. 侧边栏输入区域
# ============================================================================

st.sidebar.header("📋 临床特征输入")
st.sidebar.write("请输入患者的15项关键临床特征：")

# --- 一般信息 ---
st.sidebar.subheader("一般信息 (2项)")

gender = st.sidebar.selectbox(
    "性别",
    options=[0, 1],
    format_func=lambda x: "女" if x == 0 else "男",
    help="患者性别：男=1，女=0"
)

age = st.sidebar.number_input(
    "年龄（岁）",
    min_value=26,
    max_value=91,
    value=65,
    step=1,
    help="患者年龄，范围26～91岁"
)

# --- 影像特征 ---
st.sidebar.subheader("CT影像特征 (2项)")

con_pct = st.sidebar.number_input(
    "实变百分比 Con_pct（%）",
    min_value=0.42,
    max_value=30.48,
    value=5.0,
    step=0.01,
    format="%.4f",
    help="CT定量分析：实变区域占肺总体积的百分比，范围0.42～30.48%"
)

laa_950_pct = st.sidebar.number_input(
    "肺气肿指数 LAA_950_pct（%）",
    min_value=0.32,
    max_value=62.50,
    value=15.0,
    step=0.01,
    format="%.4f",
    help="CT定量分析：CT值低于-950 HU的低衰减区域百分比，反映肺气肿程度，范围0.32～62.50%"
)

# --- 肺功能 ---
st.sidebar.subheader("肺功能 (2项)")

fev1_pred = st.sidebar.number_input(
    "FEV1占预计值百分比 FEV1%Pred（%）",
    min_value=12.50,
    max_value=115.61,
    value=50.0,
    step=0.01,
    format="%.2f",
    help="第一秒用力呼气容积占预计值百分比，范围12.50～115.61%"
)

fev1_fvc = st.sidebar.number_input(
    "FEV1/FVC比值（%）",
    min_value=19.66,
    max_value=69.99,
    value=45.0,
    step=0.01,
    format="%.2f",
    help="第一秒用力呼气容积与用力肺活量之比，范围19.66～69.99%"
)

# --- 生理生化 ---
st.sidebar.subheader("生理生化指标 (2项)")

co2 = st.sidebar.number_input(
    "静脉血二氧化碳（mmol/L）",
    min_value=16.9,
    max_value=49.6,
    value=30.0,
    step=0.1,
    format="%.1f",
    help="静脉血二氧化碳浓度，范围16.9～49.6 mmol/L"
)

fibrinogen = st.sidebar.number_input(
    "纤维蛋白原（g/L）",
    min_value=2.01,
    max_value=9.12,
    value=3.5,
    step=0.01,
    format="%.2f",
    help="静脉血纤维蛋白原浓度，范围2.01～9.12 g/L"
)

# --- 中医证候 ---
st.sidebar.subheader("中医证候 (2项)")

tan_shi = st.sidebar.selectbox(
    "痰湿阻肺证",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="中医辨证是否为痰湿阻肺证"
)

tan_re = st.sidebar.selectbox(
    "痰热壅肺证",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help=(
        "中医辨证是否为痰热壅肺证。\n\n"
        "诊断标准（李建生等. 中华中医药杂志, 2010）：\n"
        "主症：咳嗽或喘息；痰黏色黄，咯痰不爽\n"
        "次症：发热或口渴喜冷饮；大便干结；舌质红，舌苔黄或黄腻，脉数或滑数\n"
        "具备主症2项 + 次症中2项可诊断"
    )
)

# --- 舌脉 ---
st.sidebar.subheader("舌脉 (4项)")

she_hong = st.sidebar.selectbox(
    "舌红",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="舌质是否偏红（中医舌诊）"
)

tai_huang = st.sidebar.selectbox(
    "苔黄",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="舌苔是否呈黄色（中医舌诊）"
)

mai_xian = st.sidebar.selectbox(
    "脉弦",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="脉象是否为弦脉（中医脉诊）"
)

mai_shu = st.sidebar.selectbox(
    "脉数",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有",
    help="脉象是否为数脉（中医脉诊）"
)

# 预测按钮
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 开始预测", type="primary", use_container_width=True)

# ============================================================================
# 5. 主页面结果展示
# ============================================================================

if predict_button:
    st.header("📊 预测结果")

    try:
        # 特征顺序：Con_pct, LAA_950_pct, FEV1%Pred, 纤维蛋白原, FEV1/FVC,
        #          二氧化碳, 年龄, 痰湿阻肺证, 苔黄, 脉弦, 舌红, 痰热壅肺证, 脉数, 性别
        input_array = np.array([
            con_pct,
            laa_950_pct,
            fev1_pred,
            fibrinogen,
            fev1_fvc,
            co2,
            age,
            tan_shi,
            tai_huang,
            mai_xian,
            she_hong,
            tan_re,
            mai_shu,
            gender
        ]).reshape(1, -1)

        prediction = stacking_classifier.predict(input_array)[0]
        prediction_proba = stacking_classifier.predict_proba(input_array)[0]

        predicted_subtype = int(prediction)
        predicted_proba = prediction_proba[predicted_subtype - 1]

        # ——— 5.1 核心结果 ———
        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("### 预测亚型")
            st.metric(
                label=SUBTYPE_INFO[predicted_subtype]["name"],
                value=f"预测概率: {predicted_proba * 100:.2f}%"
            )

        with col2:
            st.info("### 1年内急性加重再住院率")
            st.metric(
                label="再住院风险评估",
                value=f"{SUBTYPE_INFO[predicted_subtype]['readmission_rate']}%"
            )

        with col3:
            risk_level = SUBTYPE_INFO[predicted_subtype]["risk_level"]
            risk_color = SUBTYPE_INFO[predicted_subtype]["risk_color"]
            st.warning("### 风险等级")
            st.metric(label="综合评估", value=f"{risk_color} {risk_level}")

        # ——— 5.2 临床特征 ———
        st.markdown("---")
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.info(f"""
**📌 主要特征：**

{SUBTYPE_INFO[predicted_subtype]['main_features']}

**📋 次要特征：**

{SUBTYPE_INFO[predicted_subtype]['secondary_features']}
""")

        with col_f2:
            st.warning(f"""
**📅 随访建议：**

建议 **{SUBTYPE_INFO[predicted_subtype]['followup']}**，结合患者症状变化及时调整诊疗方案。

**通用建议：**
- 戒烟，避免二手烟及有害气体暴露
- 规律使用吸入药物（支气管扩张剂、糖皮质激素等）
- 接种流感疫苗和肺炎疫苗
- 进行肺康复训练（呼吸训练、运动训练、营养支持）
- 中医证候明显者可考虑中西医结合治疗
""")

        st.markdown("---")

        # ——— 5.3 概率分布与再住院率对比 ———
        st.subheader("📈 各亚型预测概率分布与再住院率对比")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.write("**各亚型预测概率**")
            proba_df = pd.DataFrame({
                "亚型": [SUBTYPE_INFO[i]["name"] for i in range(1, 4)],
                "概率(%)": prediction_proba * 100
            })
            st.bar_chart(proba_df.set_index("亚型"))

        with col_c2:
            st.write("**各亚型再住院率对比**")
            readmission_df = pd.DataFrame({
                "亚型": [SUBTYPE_INFO[i]["name"] for i in range(1, 4)],
                "再住院率(%)": [SUBTYPE_INFO[i]["readmission_rate"] for i in range(1, 4)]
            })
            st.bar_chart(readmission_df.set_index("亚型"))

        # ——— 5.4 详细概率表格 ———
        st.subheader("📋 详细预测概率与再住院率")

        proba_table = pd.DataFrame({
            "亚型": [SUBTYPE_INFO[i]["name"] for i in range(1, 4)],
            "预测概率": [f"{p * 100:.2f}%" for p in prediction_proba],
            "再住院率": [f"{SUBTYPE_INFO[i]['readmission_rate']}%" for i in range(1, 4)],
            "风险等级": [
                f"{SUBTYPE_INFO[i]['risk_color']} {SUBTYPE_INFO[i]['risk_level']}"
                for i in range(1, 4)
            ]
        })

        def highlight_predicted(row):
            if row["亚型"] == SUBTYPE_INFO[predicted_subtype]["name"]:
                return ["background-color: #90EE90"] * len(row)
            return [""] * len(row)

        st.dataframe(
            proba_table.style.apply(highlight_predicted, axis=1),
            use_container_width=True
        )

        st.markdown("---")

        # ——— 5.5 输入特征解读 ———
        st.subheader("🔍 当前输入特征详细解读")

        col_i1, col_i2 = st.columns(2)

        with col_i1:
            st.info(f"""
**一般信息：**
- 性别：**{"男" if gender == 1 else "女"}**
- 年龄：**{age} 岁**

**CT影像特征：**
- 实变百分比 (Con_pct)：**{con_pct:.4f}%**
- 肺气肿指数 (LAA_950_pct)：**{laa_950_pct:.4f}%**

**肺功能：**
- FEV1%Pred：**{fev1_pred:.2f}%**
- FEV1/FVC：**{fev1_fvc:.2f}%**

**生理生化：**
- 静脉血二氧化碳：**{co2:.1f} mmol/L**
- 纤维蛋白原：**{fibrinogen:.2f} g/L**
""")

        with col_i2:
            st.info(f"""
**中医证候：**
- 痰湿阻肺证：**{"有" if tan_shi == 1 else "无"}**
- 痰热壅肺证：**{"有" if tan_re == 1 else "无"}**

**舌脉：**
- 舌红：**{"有" if she_hong == 1 else "无"}**
- 苔黄：**{"有" if tai_huang == 1 else "无"}**
- 脉弦：**{"有" if mai_xian == 1 else "无"}**
- 脉数：**{"有" if mai_shu == 1 else "无"}**
""")

    except Exception as e:
        st.error(f"❌ 预测时发生错误：{e}")
        st.exception(e)

# ============================================================================
# 6. 可折叠信息面板
# ============================================================================

with st.expander("📝 查看当前输入的特征值"):
    input_summary = pd.DataFrame({
        "特征类别": [
            "一般信息", "一般信息",
            "CT影像特征", "CT影像特征",
            "肺功能", "肺功能",
            "生理生化", "生理生化",
            "中医证候", "中医证候",
            "舌脉", "舌脉", "舌脉", "舌脉"
        ],
        "特征名称": [
            "性别", "年龄",
            "实变百分比 (Con_pct)", "肺气肿指数 (LAA_950_pct)",
            "FEV1%Pred", "FEV1/FVC",
            "静脉血二氧化碳", "纤维蛋白原",
            "痰湿阻肺证", "痰热壅肺证",
            "舌红", "苔黄", "脉弦", "脉数"
        ],
        "输入值": [
            "男" if gender == 1 else "女",
            str(age),
            f"{con_pct:.4f}%",
            f"{laa_950_pct:.4f}%",
            f"{fev1_pred:.2f}%",
            f"{fev1_fvc:.2f}%",
            f"{co2:.1f} mmol/L",
            f"{fibrinogen:.2f} g/L",
            "有" if tan_shi == 1 else "无",
            "有" if tan_re == 1 else "无",
            "有" if she_hong == 1 else "无",
            "有" if tai_huang == 1 else "无",
            "有" if mai_xian == 1 else "无",
            "有" if mai_shu == 1 else "无"
        ],
        "正常范围/标准": [
            "-", "26～91岁",
            "0.42～30.48%", "0.32～62.50%",
            "12.50～115.61%", "19.66～69.99%",
            "16.9～49.6 mmol/L", "2.01～9.12 g/L",
            "见中医辨证标准", "见中医辨证标准",
            "-", "-", "-", "-"
        ]
    })
    st.dataframe(input_summary, use_container_width=True)

with st.expander("ℹ️ 各亚型详细特征及临床意义"):
    st.markdown("""
### 各亚型1年内急性加重再住院率

| 亚型 | 再住院率 | 风险等级 | 建议随访频率 |
|------|----------|----------|--------------|
| 亚型1 | **17.9%** | 🔴 高风险 | 2-4周复诊 |
| 亚型2 | 11.5%    | 🟢 低风险 | 3-6月复诊 |
| 亚型3 | 13.0%    | 🟡 中风险 | 1-2月复诊 |

---

### 亚型1（高风险）🔴

**主要人群特征：** 以老年患者为主。

**中医证候：** 痰湿阻肺证、血瘀证及痰瘀阻肺证为特异性表现，症状涉及神疲、胸痛、痛苦面容、尿频及失眠，舌脉以舌紫、舌暗、苔白、脉弦、脉沉及脉细为主。

**合并疾病：** 肺癌、肺纤维化、脑血管疾病及胸腔积液。

**生理生化：** D-二聚体、直接胆红素、谷草转氨酶、纤维蛋白原、单核细胞比率、凝血酶原时间偏高；血红蛋白及血小板偏低。

**CT影像：** LAA_950_pct低至中水平，Normal_pct处于极高水平，Con_pct高至极高水平。

---

### 亚型2（低风险）🟢

**主要人群特征：** 以肥胖、超重为主，肺功能分级为GOLD 2级。

**中医证候：** 痰热壅肺证为特异性表现，症状以咳嗽、咳痰、咯血、痰色黄、发热及头晕为主，舌脉以舌红、苔黄、苔腻、脉滑及脉数为主。

**合并疾病：** 支气管扩张及糖尿病。

**生理生化：** C反应蛋白、血小板压积及乳酸脱氢酶偏高；高密度脂蛋白胆固醇、血红蛋白及凝血酶原时间偏低。

**CT影像：** LAA_950_pct低至中水平，Con_pct处于极高水平。

---

### 亚型3（中风险）🟡

**主要人群特征：** 以男性、吸烟、饮酒及体重过低为主，肺功能分级为GOLD 3～4级，无特异性证候富集。

**中医证候：** 症状涉及痰粘难咯、面色黄、神疲、汗多及心悸，舌脉以苔厚为特异性表现。

**合并疾病：** 肺气肿。

**生理生化：** 红细胞压积、血红蛋白及高密度脂蛋白胆固醇偏高；球蛋白偏低。

**CT影像：** LAA_950_pct高至极高，Normal_pct及Con_pct均处于低至中水平。

---

### ⚠️ 注意事项

- 本预测结果仅供临床参考，不能替代医生的专业判断
- 建议结合患者病史、体格检查、影像学检查等综合评估
- 对高风险患者（亚型1），应制定个性化随访和治疗方案
- 中医证候的判断建议由专业中医师进行
""")

with st.expander("📖 使用说明"):
    st.markdown("""
### 如何使用本系统

#### 第一步：输入特征值

在左侧边栏依次输入患者的 **15项关键临床特征**：

**一般信息（2项）：**
- **性别**：男/女
- **年龄**：26～91岁

**CT影像特征（2项）：**
- **实变百分比 (Con_pct)**：CT定量，单位%，范围0.42～30.48
- **肺气肿指数 (LAA_950_pct)**：CT定量，单位%，范围0.32～62.50

**肺功能（2项）：**
- **FEV1%Pred**：FEV1占预计值百分比，范围12.50～115.61%
- **FEV1/FVC**：气道阻塞程度指标，范围19.66～69.99%

**生理生化（2项）：**
- **静脉血二氧化碳**：范围16.9～49.6 mmol/L
- **纤维蛋白原**：范围2.01～9.12 g/L

**中医证候（2项）：** 痰湿阻肺证、痰热壅肺证（有/无）

**舌脉（4项）：** 舌红、苔黄、脉弦、脉数（有/无）

#### 第二步：开始预测

点击左侧底部 **"🔮 开始预测"** 按钮。

#### 第三步：查看结果

系统将展示预测亚型、1年内再住院率、风险等级、临床特征描述、随访建议及各亚型概率分布图。

---

### 模型性能指标

**内部验证集：**

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
|--------|--------|--------|--------|-----------|
| 0.859  | 0.848  | 0.847  | 0.848  | 0.782     |

- 各亚型正确分类率：亚型1（81%）、亚型2（83%）、亚型3（91%）
- 微平均AUC = 宏平均AUC = **0.958**

**时段验证集：**

| 准确率 | 精确率 | 召回率 | F1分数 | Kappa系数 |
|--------|--------|--------|--------|-----------|
| 0.741  | 0.736  | 0.794  | 0.728  | 0.603     |

- 各亚型正确分类率：亚型1（90%）、亚型2（54%）、亚型3（77%）
- 微平均AUC = **0.907**，宏平均AUC = **0.929**

---

### 技术支持

- 📧 邮箱：wangj1758@163.com
- 📱 电话：1573196323
- 🏥 科室：呼吸科

---

### 参考文献

1. 李建生, 等. 慢性阻塞性肺疾病中医证候诊断标准（2008年版）[J]. 中华中医药杂志, 2010, 25(7): 971-975.
2. 中华医学会呼吸病学分会慢性阻塞性肺疾病学组. 慢性阻塞性肺疾病诊治指南（2021年修订版）[J]. 中华结核和呼吸杂志, 2021, 44(3): 170-205.
""")

# ============================================================================
# 7. 侧边栏信息面板
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.info("""
**模型信息**

- **模型类型**：Stacking集成学习
- **特征数量**：15项
  - 一般信息：2项
  - CT影像特征：2项
  - 肺功能：2项
  - 生理生化：2项
  - 中医证候：2项
  - 舌脉：4项
  - 性别（二分类）：1项
- **预测类别**：3个亚型
- **性能指标（内部验证）**：
  - 准确率: 0.859 | AUC: 0.958
- **性能指标（时段验证）**：
  - 准确率: 0.741 | AUC: 0.907
""")

st.sidebar.markdown("---")
st.sidebar.success("""
**风险等级快速参考**

🔴 **高风险（亚型1）**
- 再住院率：**17.9%**
- 老年，痰湿/血瘀证为主
- 建议：2-4周复诊

🟡 **中风险（亚型3）**
- 再住院率：13.0%
- 男性吸烟，GOLD 3-4级
- 建议：1-2月复诊

🟢 **低风险（亚型2）**
- 再住院率：11.5%
- 肥胖，痰热壅肺证为主
- 建议：3-6月复诊
""")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚕️ AECOPD亚型预测系统</strong></p>
    <p>基于Stacking集成学习的临床决策支持工具</p>
    <p style='font-size: 12px; color: gray; margin-top: 10px;'>
        <strong>免责声明：</strong>本系统仅供研究和辅助决策使用，不能替代专业医疗建议。<br>
        所有预测结果应由专业医生结合临床实际情况进行综合判断。<br>
        中医证候的判断建议由专业中医师进行。
    </p>
    <p style='font-size: 10px; color: gray; margin-top: 5px;'>
        版本: V1.0 | 特征: 15个 | 亚型: 3个
    </p>
</div>
""", unsafe_allow_html=True)