import streamlit as st
import pandas as pd
import pickle
import joblib

# Configuración
st.set_page_config(
    page_title="Sistema de Predicción",
    layout="centered"
)

st.title("Sistema de Predicción de Fallo en Vehículos Eléctricos")
st.write("Aplicación desarrollada como entorno productivo controlado")

# Carga del modelo
modelo = pickle.load(open("modelo_final_xgboost.sav", "rb"))
scaler = joblib.load("scaler_robust.sav")

# Entradas
st.sidebar.header("Ingrese los datos del vehículo")

#charging_voltage = st.sidebar.number_input("Voltaje de carga (V)", 0.0, 600.0, 220.0)
SOC = st.sidebar.slider("Estado de carga - SOC (%)", 0.0, 100.0, 80.0)
SOH = st.sidebar.slider("Estado de salud - SOH (%)", 0.0, 100.0, 90.0)
charging_cycles = st.sidebar.number_input("Ciclos de carga", 0, 10000, 500)
battery_temp = st.sidebar.number_input("Temperatura de la batería (°C)", 0.0, 200.0, 30.0)
motor_rpm = st.sidebar.number_input("RPM del motor", 0, 20000, 1500)
motor_torque = st.sidebar.number_input("Torque del motor (Nm)", 0.0, 500.0, 200.0)
motor_temp = st.sidebar.number_input("Temperatura del motor (°C)", 0.0, 300.0, 60.0)
brake_pad_wear = st.sidebar.slider("Desgaste de pastillas de freno (%)", 0.0, 100.0, 20.0)
tire_pressure = st.sidebar.number_input("Presión de llantas (PSI)", 0.0, 60.0, 32.0)

#Dataframe
data = {
    #"charging_voltage": charging_voltage,
    "SOC": SOC,
    "SOH": SOH,
    "charging_cycles": charging_cycles,
    "battery_temp": battery_temp,
    "motor_rpm": motor_rpm,
    "motor_torque": motor_torque,
    "motor_temp": motor_temp,
    "brake_pad_wear": brake_pad_wear,
    "tire_pressure": tire_pressure
}

df = pd.DataFrame([data])

orden_columnas = [
    "SOC",
    "SOH",
    "charging_cycles",
    "battery_temp",
    "motor_rpm",
    "motor_torque",
    "motor_temp",
    "brake_pad_wear",
    "tire_pressure"
]

st.subheader("Datos ingresados")
st.write(df)

#Escalador
df_scaled_values = scaler.transform(df) 
df_scaled = pd.DataFrame(df_scaled_values, columns = orden_columnas)

# Botón
if st.button("Realizar predicción"):

    pred = modelo.predict(df_scaled)
    prob = modelo.predict_proba(df_scaled)

    st.subheader("Resultado")

    if pred[0] == 1:
        st.error("ALERTA: Riesgo de fallo en el vehículo eléctrico")
    else:
        st.success("Estado NORMAL del vehículo eléctrico")

    st.subheader("Probabilidad")
    st.write(prob)