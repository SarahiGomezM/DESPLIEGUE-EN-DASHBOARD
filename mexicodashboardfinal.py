
#Librerias 
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from funpymodeling.exploratory import freq_tbl


#Estilos personalizados
custom_font = """
<style>
body {
  font-family: 'Arial', sans-serif;
}
</style>
"""

background_color = """
<style>
.stApp {
  background-color: #399999;
}

.sidebar .sidebar-content {
  background-color: #e0e0e0;
}
</style>
"""

st.markdown(custom_font, unsafe_allow_html=True)
st.markdown(background_color, unsafe_allow_html=True)



#################################################################

# Cargar la base de datos limpia
@st.cache_data
def load_data():
    data = pd.read_csv('Base_Mexico.csv')
    # Convertir 'host_is_superhost' a binario si no lo está
    if data['host_is_superhost'].dtype == object:
        data['host_is_superhost'] = data['host_is_superhost'].map({'t':"Is Superhost", 'f':"Not Superhost"})
    return data

df= load_data()

# --- PORTADA ---
st.title("Análisis de Superhosts Airbnb - Ciudad de México")  # Título principal
st.image("istockphoto-1175975898-612x612.jpg")  # Reemplaza "portada.jpg" con la ruta de tu imagen de portada
st.markdown("---")  # Separador visual

# 1. CREACIÓN DE LA SIDEBAR
# Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("Análisis de Superhosts Airbnb")
st.sidebar.header("Filtros")
st.sidebar.subheader("Panel de Selección")

# Ordenar las opciones de tipo de propiedad por popularidad
property_type_counts = df['property_type'].value_counts()
property_types_ordered = property_type_counts.index.tolist()

# Widgets en la sidebar para seleccionar tipo de propiedad, barrios y rango de precios
property_type = st.sidebar.selectbox("Selecciona el tipo de propiedad", df['property_type'].unique())
neighbourhood = st.sidebar.multiselect("Selecciona barrios", df['neighbourhood_cleansed'].unique())
price_range = st.sidebar.slider("Selecciona el rango de precios", 
                                int(df['price'].min()), int(df['price'].max()), (100, 500))



# 2. CREACIÓN DE LOS FRAMES
# Menú desplegable para seleccionar entre diferentes frames
Frames = st.selectbox(label="Frames", options=["Frame 1: Distribución", "Frame 2: Análisis Bivariado", "Frame 3: Modelado Predictivo"])

# Aplicar los filtros seleccionados
filtered_df = df[df['property_type'] == property_type]
if len(neighbourhood) > 0:
    filtered_df = filtered_df[filtered_df['neighbourhood_cleansed'].isin(neighbourhood)]
filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]



#################################################################

# 3. CONTENIDO DEL FRAME 1: Distribución de Superhosts


if Frames == "Frame 1: Distribución":
    st.title("Análisis de Superhosts Airbnb - Ciudad de México")
    
    # Verificar si hay datos suficientes para generar los gráficos
    if filtered_df.empty:
        st.warning("No hay datos disponibles para el tipo de propiedad, barrios y rango de precios seleccionados. Ajusta los filtros para visualizar los gráficos.")
    else:
        # Distribución de Superhosts por tipo de propiedad
        st.subheader("Distribución de Superhosts por Tipo de Propiedad")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=filtered_df, x='property_type', hue='host_is_superhost', ax=ax1)
        st.pyplot(fig1)

        # Calcular la proporción de Superhosts y No Superhosts por tipo de propiedad
        superhost_percentage_by_type = filtered_df.groupby(['property_type', 'host_is_superhost'])['host_is_superhost'].count().reset_index(name='count')
        superhost_percentage_by_type['percentage'] = superhost_percentage_by_type.groupby('property_type')['count'].transform(lambda x: (x / x.sum()) * 100)

        # Pie Chart: Proporción de Superhosts y No Superhosts por Tipo de Propiedad
        st.subheader("Proporción de Superhosts y No Superhosts por Tipo de Propiedad")
        fig2 = px.pie(superhost_percentage_by_type, values='percentage', names='host_is_superhost', color='host_is_superhost',
                      facet_col='property_type', title='Proporción de Superhosts y No Superhosts por Tipo de Propiedad')
        st.plotly_chart(fig2)

        
        # Mostrar el porcentaje de Superhosts por tipo de propiedad
        st.subheader("Porcentaje de Superhosts por Tipo de Propiedad")
        st.write(superhost_percentage_by_type)
        

    

#################################################################

# 4. CONTENIDO DEL FRAME 2: Análisis Bivariado (correlaciones y precios)
if Frames == "Frame 2: Análisis Bivariado":
    st.title("Análisis Bivariado entre Superhosts y Variables Numéricas")

    try:
        # Distribución de Precios entre Superhosts y No Superhosts
        st.subheader("Distribución de Precios entre Superhosts y No Superhosts")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=filtered_df, x='host_is_superhost', y='price', ax=ax3)
        st.pyplot(fig3)

        # Matriz de Correlaciones
        st.subheader("Matriz de Correlaciones")
        numeric_cols = ['price', 'number_of_reviews', 'review_scores_rating']
        corr = filtered_df[numeric_cols].corr()
        fig4, ax4 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)
        # Calcular y mostrar los coeficientes de correlación
        st.subheader("Coeficientes de Correlación:")
        numeric_cols = ['price', 'number_of_reviews', 'review_scores_rating']
        corr_matrix = filtered_df[numeric_cols].corr()
        st.write(corr_matrix)



        # Calcular y mostrar estadísticas descriptivas para Superhosts
        st.subheader("Estadísticas Descriptivas para Superhosts:")
        superhosts_df = df[df['host_is_superhost'] == 'Is Superhost']  # Filtrar solo Superhosts del DataFrame original
        st.write(f"**Superhosts:**")
        st.write(f"Media: {superhosts_df['price'].mean():.2f}")
        st.write(f"Mediana: {superhosts_df['price'].median():.2f}")
        st.write(f"Desviación Estándar: {superhosts_df['price'].std():.2f}")



        # Convertir 'host_is_superhost' a numérico (0 o 1)
        filtered_df['host_is_superhost'] = filtered_df['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

        # Calcular y mostrar estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas para No Superhost:")

        # Definir nonsuperhost_prices fuera del bloque if
        nonsuperhost_prices = filtered_df.loc[filtered_df['host_is_superhost'] == 0, 'price']

      
        # Mostrar estadísticas de No Superhosts
        st.write(f"**No Superhosts:**")
        st.write(f"Media: {nonsuperhost_prices.mean():.2f}")
        st.write(f"Mediana: {nonsuperhost_prices.median():.2f}")
        st.write(f"Desviación Estándar: {nonsuperhost_prices.std():.2f}")

        # Descripción de los valores de 'host_is_superhost'
        st.write("**Nota:** 1 : Superhosts   |   0 : No Superhosts.")

        
    except ValueError:
        st.warning("No hay Distribución de Precios entre Superhosts y No Superhosts dentro de los rangos seleccionados.")


#################################################################

# 5. CONTENIDO DEL FRAME 3: Modelado Predictivo (Regresión Logística)
if Frames == "Frame 3: Modelado Predictivo":
    st.title("Predicción de Precios con Regresión Lineal")

    # --- Feature Engineering and Preprocessing ---
    features = ['number_of_reviews', 'review_scores_rating', 'accommodates', 'bedrooms', 'bathrooms', 'property_type', 'neighbourhood_cleansed']
    modeling_df = df[features + ['price']]
    
    # Aplicar los filtros al modeling_df (si es necesario)
    modeling_df = modeling_df[modeling_df['property_type'] == property_type]
    if len(neighbourhood) > 0:
        modeling_df = modeling_df[modeling_df['neighbourhood_cleansed'].isin(neighbourhood)]
    modeling_df = modeling_df[(modeling_df['price'] >= price_range[0]) & (modeling_df['price'] <= price_range[1])]
    
        
    # One-hot encoding para 'property_type' y 'neighbourhood_cleansed'
    modeling_df = pd.get_dummies(modeling_df, columns=['property_type', 'neighbourhood_cleansed'], prefix=['property_type', 'neighbourhood'])

    # --- Model Training and Prediction ---
    try:
        X = modeling_df.drop('price', axis=1)
        y = modeling_df['price']
        
        # Escalar las variables numéricas
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Mostrar las predicciones
        st.subheader("Predicciones de Precios:")
        predictions_df = pd.DataFrame({'Precio Real': y_test, 'Precio Predicho': y_pred})  
        predictions_df['Recomendación'] = predictions_df.apply(lambda row: 'Subir Precio' if row['Precio Predicho'] > row['Precio Real'] else 'Bajar Precio', axis=1)
        st.write(predictions_df)

        # ... (Otras métricas de evaluación)

    except ValueError:
        st.write("No hay suficientes datos para entrenar el modelo. Ajusta los filtros.")


import streamlit.components.v1 as components
components.html("<script>const elements = window.parent.document.querySelectorAll('.stButton button');  elements.forEach(el => el.style.visibility = 'hidden'); </script>", height=0, width=0)
