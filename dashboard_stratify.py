import streamlit as st
import pandas as pd
import joblib
import calendar
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
import base64
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ========== CONFIGURACIÓN ==========

st.set_page_config(page_title="Ximple Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* === FONDO Y ESTILO GENERAL === */
body, .stApp {
    font-family: 'Segoe UI', sans-serif;
    background-color: white;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background-color: #1e1e1e !important;
}

/* === BOTONES GENERALES === */
.metric-label, .metric-value, .stButton>button, .stDownloadButton>button {
    color: white !important;
    background-color: #d4b14c !important;
    border-radius: 6px;
    border: none;
    font-weight: 600;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
    background-color: #c19b3a !important;
    color: white !important;
    transition: background-color 0.3s ease-in-out;
}

/* === RADIO BUTTONS === */

/* Color del texto */
[data-testid="stSidebar"] .stRadio label {
    color: white !important;
}

/* Cambia el círculo del botón activo a dorado */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] svg {
    stroke: #d4b14c !important;
    fill: #d4b14c !important;
}

/* Opcion activa: texto blanco y negrita */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] > div:nth-child(2) {
    font-weight: bold;
    color: white !important;
}

/* Hover opcional para svg */
[data-testid="stSidebar"] .stRadio label:hover svg {
    stroke: #d4b14c !important;
}

/* Radio button borde extra para visibilidad */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] div[role="presentation"] {
    border: 2px solid #d4b14c !important;
    background: #d4b14c !important;
}

/* Selectbox borde blanco */
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stRadio {
    border-radius: 8px;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ========== CARGA DE DATOS ==========
@st.cache_data
def load_model():
    return joblib.load("modelo_final.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("df_merged_clusters.csv")

modelo = load_model()
df = load_data()

# ========== SIDEBAR CLUSTER (Texto blanco, grande y en negritas) ==========
st.sidebar.markdown("<h2 style='color:white; font-size:24px; font-weight:bold;'>Filtros</h2>", unsafe_allow_html=True)

cluster_names = {
    0: "Pagos Frecuentes y Pocos Atrasos",
    1: "Alto Volumen y Alta Morosidad",
    2: "Baja Actividad Crediticia"
}
cluster_display = [f"{k} – {v}" for k, v in cluster_names.items()]

# Etiqueta del selectbox personalizada
st.sidebar.markdown("<p style='color:white; font-size:18px; font-weight:bold;'>Selecciona el tipo de Aliada:", unsafe_allow_html=True)
cluster_str = st.sidebar.selectbox("", cluster_display)

cluster_sel = int(cluster_str.split(" – ")[0])
df_cluster = df[df['cluster_kmeans'] == cluster_sel]
df_cluster = df_cluster.copy()
df_cluster["intensive_use"] = df_cluster["intensive_use"].astype(str)

# ========== MENÚ DE NAVEGACIÓN (Texto grande, fuerte y elegante) ==========
st.markdown(
    """
    <style>
        /* Cambia color del texto de las opciones de radio */
        .stRadio > div div {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <div style='margin-top: 30px; margin-bottom: 10px;'>
        <p style='color: white; font-size: 18px; font-weight: bold;'>Secciones del Dashboard:</p>
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    label="",
    options=[
        "Inicio",
        "Resumen de Clusters",
        "Comportamiento de Demanda",
        "Calidad de Pagos",
        "Distribución Geográfica",
        "Predicción de Intensivas"
    ],
    key="menu_radio"
)
# Logo en la esquina superior izquierda
st.sidebar.markdown(
    """
    <style>
        /* Aseguramos que el sidebar sea relativo para posicionamiento absoluto interno */
        section[data-testid="stSidebar"] {
            position: relative;
        }

        /* Estilo del contenedor del logo */
        .logo-container {
            position: fixed;
            bottom: 0;
            left: 0;
            padding: 10px;
            z-index: 100;
        }

        .logo-container img {
            height: 80px;
        }
    </style>

    <div class="logo-container">
        <img src="https://raw.githubusercontent.com/Juca8/DashboardEdit/main/Logotipo%20STRATIFY.png">
    </div>
    """,
    unsafe_allow_html=True
)
# ========== TÍTULO CON LOGO CENTRADO Y ELEGANTE ==========
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_base64_image("StratifyLogo.png")

st.markdown(
    f"""
    <div style='text-align: center; padding: 30px 0 40px 0;'>
        <img src='data:image/png;base64,{image_base64}' width='200'
             style='margin-bottom: 15px; border-radius: 16px;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);'/>
        <h1 style='font-family: "Segoe UI", sans-serif; margin-bottom: 0;'>Ximple – Dashboard de Comportamiento Crediticio</h1>
        <h3 style='color: #bfa14c; font-weight: normal; margin-top: 0;'>{cluster_names[cluster_sel]}</h3>
    </div>
    """,
    unsafe_allow_html=True
)
  #========== NUEVA SECCIÓN: INICIO ==========
if menu == "Inicio":
    st.markdown("<h2 style='color:#d4b14c;'>📌 Introduction</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:black; font-size:16px;'>
     En Ximple, creemos que comprender el comportamiento de consumo de nuestras <strong>Aliadas</strong> —nuestras aliadas comerciales— es esencial para un crecimiento sostenible. 
    Este proyecto representa un esfuerzo por aprovechar el poder del análisis de datos, la visualización y el aprendizaje automático para transformar la forma en que se entiende y gestiona la demanda de préstamos. 
    Hemos desarrollado un dashboard interactivo que permite a los tomadores de decisiones pasar de respuestas reactivas a estrategias proactivas, personalizando el soporte y optimizando la asignación de recursos.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color:#d4b14c;'>🔍 Situación / Problema</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:black; font-size:16px;'>
    ¿Cómo podemos usar los datos disponibles para anticipar y entender el uso de préstamos entre nuestras Aliadas?<br><br>
     Aunque Ximple cuenta con registros históricos de préstamos, aún no se había aprovechado plenamente su valor para predecir patrones de uso, identificar usuarias estratégicas o adaptar las operaciones según la estacionalidad o el tipo de cliente. 
    Esta falta de enfoque predictivo limitaba la capacidad de asignar recursos de manera eficiente, personalizar estrategias de contacto o desarrollar productos financieros ajustados al comportamiento real de las usuarias.
     </p>
     """, unsafe_allow_html=True)
    
      st.markdown("<hr style='border:1px solid #d4b14c;'>", unsafe_allow_html=True)
      st.markdown("<h2 style='color:#d4b14c;'> Conclusions & Recommendations</h2>", unsafe_allow_html=True)
      st.markdown("""
     <p style='color:black; font-size:16px;'>
     Nuestros hallazgos revelaron que aproximadamente el 20% de las Aliadas generan la mayor parte de la actividad crediticia. Estas usuarias de alto uso suelen solicitar préstamos en intervalos cortos y muestran un comportamiento estratégico. Recomendamos:
    
    • Enfocar los esfuerzos de segmentación en las usuarias intensivas para aumentar la retención y maximizar el valor a largo plazo.
    • Integrar variables predictivas como el intervalo entre préstamos, el tipo de producto y la región en los dashboards para respaldar decisiones operativas.
    • Utilizar XIMPLE FIJO como producto de entrada y XIMPLE OPTIMIZA para clientas más maduras.
    • Preparar las operaciones para los picos de demanda de fin de año.
    
    Al implementar internamente el modelo predictivo y las herramientas visuales, Ximple puede pasar de una analítica descriptiva a una estrategia proactiva, mejorando la experiencia del cliente y la eficiencia operativa.
     </p>
    """, unsafe_allow_html=True)
    
     st.markdown("<hr style='border:1px solid #d4b14c;'>", unsafe_allow_html=True)
    
# ========== NUEVA SECCIÓN: RESUMEN DE CLUSTERS ==========
if menu == "Resumen de Clusters":
    st.subheader("¿Qué significan los Clusters?")
    st.markdown("""
    Los clusters agrupan a las Aliadas según su comportamiento crediticio, no por contacto ni comunicación.

    - **Cluster 1** – *Aliadas con pagos frecuentes y pocos atrasos*.
        - Realizan muchos préstamos con tiempos cortos entre cada uno.
        - Aunque tienen mora su comportamiento es muy activo y estable.
    - **Cluster 2** – *Aliadas con alto volumen y alta morosidad*.
        - Muchas llamadas con mora alta y pagos tardíos.
        - Son clientes con comportamiento más riesgoso.
    - **Cluster 3** – *Aliadas de baja actividad crediticia*.
        - Tienen pocas cuotas pagadas y un bajo volumen de préstamos.
        - Uso esporádico o poco intensivo del sistema.

    Estas agrupaciones fueron generadas usando un modelo KMeans sobre variables como:
    - `cuotas_pagadas`
    - `cuotas_tarde`
    - `cuotas_mora`
    - `dias_promedio` 
    - `Total_llamadas`

    Además, se validó el modelo con PCA y el método del codo para asegurar que las 3 agrupaciones fueran las más representativas del comportamiento natural.
    """, unsafe_allow_html=True)

    # Mostrar la tabla con el resumen por cluster
    st.markdown("### Resumen General por Cluster")
    df_summary = pd.DataFrame({
        'Cluster': [0, 1, 2],
        'cuotas_pagadas': [44.94, 16.18, 131.45],
        'cuotas_tarde': [5.23, 2.57, 18.56],
        'cuotas_mora': [16.04, 2.16, 2.69],
        'dias_promedio': [-1.71, -0.80, -0.70],
        'Total_llamadas': [4.29, 2.08, 5.06],
        'Región Centro (%)': [48.0, 68.0, 23.0],
        'Región Norte (%)': [3.0, 2.0, 0.0],
        'Región Other (%)': [31.0, 16.0, 66.0],
        'Región Sur (%)': [19.0, 14.0, 11.0]
    })
    st.dataframe(df_summary, use_container_width=True, height=150)


# ========== MÉTRICAS ==========
if menu != "Inicio":
    st.subheader("Métricas del Cluster")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Aliadas", len(df_cluster))
    col2.metric("% Intensivas", f"{100 * df_cluster['intensive_use'].astype(int).mean():.1f}%")
    col3.metric("Prom. días entre préstamos", f"{df_cluster['dias_promedio'].mean():.1f} días")
# ========== GRÁFICAS SEGMENTADAS ==========
if menu == "Comportamiento de Demanda":
    st.subheader("Comportamiento de Demanda")

    tipo_prestamo = df_cluster.groupby(["LoanType", "RecipientType"]).size().reset_index(name="count")
    src1 = ColumnDataSource(tipo_prestamo)
    fig1 = figure(x_range=tipo_prestamo["LoanType"].unique(), width=800, height=400, title="Tipo de Préstamo")
    fig1.title.text_font_size = "16pt"  # Tamaño
    fig1.title.text_font_style = "bold"  # Negritas
    fig1.vbar(x="LoanType", top="count", width=0.7, source=src1,
              legend_field="RecipientType",
              color=factor_cmap("RecipientType", palette=["#bfa14c", "#000000"], factors=["Ally", "Client"]))
    fig1.xgrid.grid_line_color = None
    fig1.legend.title = "Tipo de Cliente"
    fig1.legend.label_text_font_size = "10pt"
    fig1.legend.label_text_font = "Segoe UI"
    fig1.legend.location = "top_right"
    st.bokeh_chart(fig1, use_container_width=True)

    if "IssueMonth" in df_cluster.columns:
        prestamos_mes = df_cluster["IssueMonth"].value_counts().sort_index().reset_index()
        prestamos_mes.columns = ["Mes", "Total"]
        prestamos_mes["Mes"] = prestamos_mes["Mes"].apply(lambda x: calendar.month_abbr[int(x)])
        src5 = ColumnDataSource(prestamos_mes)
        fig5 = figure(x_range=prestamos_mes["Mes"], width=800, height=400, title="Estacionalidad de Préstamos")
        fig5.title.text_font_size = "16pt"  # Tamaño
        fig5.title.text_font_style = "bold"  # Negritas
        fig5.vbar(x="Mes", top="Total", width=0.6, source=src5, color="#bfa14c")
        st.bokeh_chart(fig5, use_container_width=True)

    if "payment_frequency" in df_cluster.columns:
        freq = df_cluster["payment_frequency"].value_counts().reset_index()
        freq.columns = ["Frecuencia", "Total"]
        src6 = ColumnDataSource(freq)
        fig6 = figure(x_range=freq["Frecuencia"], width=800, height=400, title="Frecuencia de Pago")
        fig6.title.text_font_size = "16pt"  # Tamaño
        fig6.title.text_font_style = "bold"  # Negritas
        fig6.vbar(x="Frecuencia", top="Total", width=0.6, source=src6, color="#bfa14c")
        st.bokeh_chart(fig6, use_container_width=True)

elif menu == "Calidad de Pagos":
    st.subheader("Calidad de Pagos")

    src2 = ColumnDataSource(df_cluster)
    fig2 = figure(width=800, height=400, title="Frecuencia vs. Simultaneidad")
    fig2.title.text_font_size = "16pt"  # Tamaño
    fig2.title.text_font_style = "bold"  # Negritas
    fig2.circle(x="dias_promedio", y="prestamos_outstanding", size=6, source=src2,
                color=factor_cmap('intensive_use', palette=['#bfa14c', '#000000'], factors=['0', '1']),
                legend_field="intensive_use")
    st.bokeh_chart(fig2, use_container_width=True)

    tipo_mora = df_cluster.groupby("RecipientType")["cuotas_mora"].mean().reset_index()
    src3 = ColumnDataSource(tipo_mora)
    fig3 = figure(x_range=tipo_mora["RecipientType"], width=800, height=400, title="Mora por Tipo de Cliente")
    fig3.title.text_font_size = "16pt"  # Tamaño
    fig3.title.text_font_style = "bold"  # Negritas
    fig3.vbar(x="RecipientType", top="cuotas_mora", width=0.5, source=src3, color="#bfa14c")
    st.bokeh_chart(fig3, use_container_width=True)

    if "contactability_level" in df_cluster.columns:
        mora_contacto = df_cluster.groupby("contactability_level")["cuotas_mora"].mean().reset_index()
        src7 = ColumnDataSource(mora_contacto)
        fig7 = figure(x_range=mora_contacto["contactability_level"], width=800, height=400, title="Mora por Contactabilidad")
        fig7.title.text_font_size = "16pt"  # Tamaño
        fig7.title.text_font_style = "bold"  # Negritas
        fig7.vbar(x="contactability_level", top="cuotas_mora", width=0.5, source=src7, color="#bfa14c")
        st.bokeh_chart(fig7, use_container_width=True)

    if "effective_payer" in df_cluster.columns:
        mora_pago = df_cluster.groupby("effective_payer")["cuotas_mora"].mean().reset_index()
        mora_pago["effective_payer"] = mora_pago["effective_payer"].astype(str)
        src8 = ColumnDataSource(mora_pago)
        fig8 = figure(x_range=mora_pago["effective_payer"], width=800, height=400, title="Mora por cumplimiento")
        fig8.title.text_font_size = "16pt"  # Tamaño
        fig8.title.text_font_style = "bold"  # Negritas
        fig8.vbar(x="effective_payer", top="cuotas_mora", width=0.5, source=src8, color="#bfa14c")
        st.bokeh_chart(fig8, use_container_width=True)

elif menu == "Distribución Geográfica":
    st.subheader("Distribución Geográfica")

    region = df_cluster["customer_region"].value_counts().reset_index()
    region.columns = ["Región", "Total"]
    src4 = ColumnDataSource(region)
    fig4 = figure(x_range=region["Región"], width=800, height=400, title="Clientes por Región")
    fig4.title.text_font_size = "16pt"  # Tamaño
    fig4.title.text_font_style = "bold"  # Negritas
    fig4.vbar(x="Región", top="Total", width=0.6, source=src4, color="#bfa14c")
    st.bokeh_chart(fig4, use_container_width=True)

    # Tabla simplificada de ciudades por región
    st.markdown("### Estados por Región")
    region_map = {
        "North": "NUEVO LEÓN, CHIHUAHUA",
        "South": "VERACRUZ, OAXACA, TABASCO, GUERRERO, CHIAPAS",
        "Other": "DURANGO, SAN LUIS POTOSÍ, JALISCO, TAMAULIPAS, AGUASCALIENTES",
        "Center": "ESTADO DE MÉXICO, CDMX, PUEBLA, QUERÉTARO, HIDALGO"
    }
    df_region_pretty = pd.DataFrame(region_map.items(), columns=["Región", "Ciudades"])
    st.dataframe(df_region_pretty, use_container_width=True, height=180)

# ========== PREDICCIÓN ==========

elif menu == "Predicción de Intensivas":
    st.subheader("Predicción de Aliadas Intensivas")

    st.markdown("""
    ### ¿Qué hace esta sección?
    Esta sección permite **predecir si una Aliada será intensiva en el uso de préstamos** a partir de características clave de su comportamiento.

    ### ¿Qué significa "intensiva"?
    Una aliada es intensiva si cumple estas 3 condiciones:
    1. **Más de 2 préstamos activos simultáneamente**
    2. **Tiempo promedio entre préstamos menor a 15 días**
    3. **Más de 3 préstamos entregados en menos de 60 días**

    ### ¿Cómo usarlo?
    1. Descarga la plantilla con las columnas correctas.
    2. Llena los datos siguiendo los ejemplos.
    3. Súbela para obtener la predicción.

    ### Variables utilizadas por el modelo:
    - `dias_promedio`
    - `RecipientType` (Client o Ally)
    - `LoanType` (XIMPLE FIJO, XIMPLE OPTIMIZA)
    - `DisbursementMeans` (STP, MERCANCIA, etc.)
    - `customer_region` (North, South, Center, Other)
    """, unsafe_allow_html=True)

    # Descargar plantilla
    try:
        with open("plantilla_prediccion_aliada.csv", "rb") as file:
            st.download_button("📄 Descargar Plantilla CSV", file, file_name="plantilla_prediccion_aliada.csv")
    except FileNotFoundError:
        st.warning("❌ No se encontró la plantilla.")

    # Uploader
    file = st.file_uploader("Carga un archivo CSV para predecir", type="csv")
    if file:
        try:
            # Leer input y corregir valores
            new_data = pd.read_csv(file)
            new_data["LoanType"] = new_data["LoanType"].replace("XIMPLE FLUJO", "XIMPLE FIJO")

            # Preprocesamiento
            top_5_features = ['dias_promedio', 'RecipientType', 'LoanType', 'DisbursementMeans', 'customer_region']
            num_cols = ['dias_promedio']
            cat_cols = ['RecipientType', 'LoanType', 'DisbursementMeans', 'customer_region']

            df_original = pd.read_csv("df_merged_clusters.csv")
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
            ])
            preprocessor.fit(df_original[top_5_features])

            X_input = preprocessor.transform(new_data[top_5_features])
            predictions = modelo.predict(X_input)
            new_data["Prediccion_Intensive"] = predictions

            st.success("✅ Predicciones realizadas")
            st.dataframe(new_data)
            st.download_button("Descargar resultados", new_data.to_csv(index=False), file_name="predicciones_resultado.csv")
        except Exception as e:
            st.error(f"❌ Error al predecir: {e}")
# ========== FOOTER ==========
st.markdown("---")
st.caption("Ximple Dashboard · Proyecto Final de Consultoría")
