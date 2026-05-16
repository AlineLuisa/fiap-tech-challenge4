# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# 1. CONFIGURAÇÃO INICIAL DO STREAMLIT
# ==================================================
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="⚖️",
    layout="centered"
)

st.title("🍽️ Predição de Nível de Obesidade")
st.markdown(
    """
    Esta aplicação utiliza um modelo de Machine Learning para classificar o nível de obesidade
    com base em dados demográficos, hábitos alimentares e estilo de vida.
    """
)

# ==================================================
# 2. CARREGAR O MODELO TREINADO
# ==================================================
@st.cache_resource
def load_model():
    """Carrega o pipeline treinado salvo em disco."""
    return joblib.load("logreg_best_pipeline.joblib")

model = load_model()
st.success("✅ Modelo carregado com sucesso!")

# ==================================================
# 3. FUNÇÕES DE FEATURE ENGINEERING
# ==================================================
def compute_imc(weight, height):
    """Calcula o Índice de Massa Corporal (IMC)."""
    return weight / (height ** 2)

def compute_healthy_score(fcvc, faf, ch2o):
    """Calcula o Healthy Score: soma do consumo de vegetais, atividade física e consumo de água."""
    return fcvc + faf + ch2o

def compute_sedentary_index(tue, faf):
    """Calcula o índice de sedentarismo: tempo de uso de tecnologia / (atividade física + 1)."""
    return tue / (faf + 1)

def map_mtrans(mtrans):
    """Mapeia o meio de transporte para código numérico."""
    mapping = {
        "Walking": 3,
        "Bike": 3,
        "Public_Transportation": 2,
        "Automobile": 1,
        "Motorbike": 1
    }
    return mapping.get(mtrans, 2)  # default = 2

def map_caec(caec):
    """Mapeia consumo de comida entre refeições."""
    mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    return mapping.get(caec, 1)

def map_calc(calc):
    """Mapeia consumo de álcool."""
    mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    return mapping.get(calc, 0)

def binary_encode(value):
    """Retorna [0,1] para 'yes' e [1,0] para 'no'."""
    if value == "yes":
        return [0, 1]
    else:
        return [1, 0]

# ==================================================
# 4. FORMULÁRIO DE ENTRADA DO USUÁRIO
# ==================================================
with st.form("form_predicao"):
    st.subheader("📋 Dados Pessoais")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gênero", ["Female", "Male"])
        age = st.slider("Idade (anos)", min_value=14, max_value=80, value=25, step=1)
        height = st.number_input("Altura (m)", min_value=1.0, max_value=2.2, value=1.70, step=0.01, format="%.2f")
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5, format="%.1f")
    with col2:
        family_history = st.selectbox("Histórico familiar de sobrepeso", ["yes", "no"])
        favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
        fcvc = st.slider("Consumo de vegetais (1 = raramente, 2 = às vezes, 3 = sempre)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)

    st.subheader("🏃‍♂️ Hábitos e Estilo de Vida")
    col3, col4 = st.columns(2)
    with col3:
        caec = st.selectbox("Consome comida entre refeições?", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Fumante?", ["yes", "no"])
        ch2o = st.slider("Consumo de água (1 = <1L, 2 = 1-2L, 3 = >2L)", 1, 3, 2)
        scc = st.selectbox("Monitora consumo calórico?", ["yes", "no"])
    with col4:
        faf = st.slider("Atividade física semanal (0 = nenhuma, 1 = 1-2x, 2 = 3-4x, 3 = 5x+)", 0, 3, 1)
        tue = st.slider("Tempo em dispositivos eletrônicos (0 = 0-2h, 1 = 3-5h, 2 = >5h)", 0, 2, 1)
        calc = st.selectbox("Consome álcool?", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Meio de transporte principal", ["Walking", "Bike", "Public_Transportation", "Automobile", "Motorbike"])

    st.subheader("📊 Informações Adicionais (calculadas automaticamente)")
    st.caption("IMC, Healthy Score e Índice de Sedentarismo serão calculados com base nos dados acima.")

    submitted = st.form_submit_button("🔍 Prever Nível de Obesidade")

# ==================================================
# 5. PROCESSAMENTO E PREDIÇÃO
# ==================================================
if submitted:
    with st.spinner("Processando dados e realizando predição..."):
        # --- Calcula features derivadas ---
        imc = compute_imc(weight, height)
        healthy_score = compute_healthy_score(fcvc, faf, ch2o)
        sedentary_index = compute_sedentary_index(tue, faf)

        # --- Mapeamentos e codificações ---
        mtrans_code = map_mtrans(mtrans)
        caec_code = map_caec(caec)
        calc_code = map_calc(calc)
        gender_binary = 1 if gender == "Female" else 0

        # --- Binary encoding para variáveis binárias ---
        family_hist_encoded = binary_encode(family_history)
        favc_encoded = binary_encode(favc)
        smoke_encoded = binary_encode(smoke)
        scc_encoded = binary_encode(scc)

        # --- Montar Dicionário com absolutamente todas as colunas possíveis ---
        # Inclui as 18 colunas finais E as 5 colunas fantasmas que o preprocessor exige na entrada
        raw_input = {
            # As 18 colunas que você quer no modelo final:
            'Age': age,
            'Height': height,
            'family_history_0': family_hist_encoded[0],
            'family_history_1': family_hist_encoded[1],
            'FAVC_0': favc_encoded[0],
            'FAVC_1': favc_encoded[1],
            'NCP': ncp,
            'SMOKE_0': smoke_encoded[0],
            'SMOKE_1': smoke_encoded[1],
            'SCC_0': scc_encoded[0],
            'SCC_1': scc_encoded[1],
            'imc': imc,
            'Healthy_Score': healthy_score,
            'Sedentary_Index': sedentary_index,
            'MTRANS_Code': mtrans_code,
            'caec_code': caec_code,
            'calc_code': calc_code,
            'gender_binary': gender_binary,
            
            # As 5 colunas que o preprocessor exige para não dar erro (alimentadas com os inputs atuais)
            'Weight': weight,
            'FCVC': fcvc,
            'CH2O': ch2o,
            'FAF': faf,
            'TUE': tue
        }

        # Criar o DataFrame inicial
        input_df = pd.DataFrame([raw_input])

        # --- Reindexar dinamicamente conforme o que o Pipeline espera na ENTRADA ---
        # Isso vai ordenar as colunas na ordem exata exigida pelo seu arquivo logreg_best_pipeline.joblib
        expected_columns = model.named_steps['preprocessor'].feature_names_in_
        X_input = input_df.reindex(columns=expected_columns)

        # --- Predição ---
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]

        # Mapeamento das classes (conforme seu classification_report)
        target_names = [
            "Insufficient_Weight",
            "Normal_Weight",
            "Obesity_Type_I",
            "Obesity_Type_II",
            "Obesity_Type_III",
            "Overweight_Level_I",
            "Overweight_Level_II"
        ]
        proba_dict = {cls: round(prob * 100, 2) for cls, prob in zip(target_names, prediction_proba)}

    # ==================================================
    # 6. EXIBIÇÃO DOS RESULTADOS
    # ==================================================
    st.subheader("📈 Resultado da Predição")
    st.success(f"### 🧬 Nível de Obesidade Previsto: **{prediction}**")

    with st.expander("🔍 Ver detalhes das probabilidades"):
        st.write("Probabilidades por classe (%):")
        st.dataframe(pd.DataFrame(proba_dict.items(), columns=["Classe", "Probabilidade (%)"]))

    st.subheader("📊 Resumo das Informações Inseridas")
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Idade", f"{age} anos")
        st.metric("Altura", f"{height:.2f} m")
        st.metric("Peso", f"{weight:.1f} kg")
        st.metric("IMC", f"{imc:.2f}")
    with col6:
        st.metric("Healthy Score", healthy_score)
        st.metric("Índice de Sedentarismo", f"{sedentary_index:.2f}")
        st.metric("Refeições principais/dia", ncp)
        st.metric("Consumo de vegetais (escala 1-3)", fcvc)

    st.caption("Modelo treinado com Regressão Logística (acurácia ~94%) utilizando dados de estilo de vida e hábitos alimentares.")