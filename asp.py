import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import statsmodels.api as sm

# Configuración de la página
st.set_page_config(page_title="Sistema de Análisis de Estudiantes - SIATECH ORAH", layout="wide")
st.title("Sistema de Análisis de Estudiantes")
st.header("Bienvenido al sistema de análisis de estudiantes")

# Sección de carga de datos
st.sidebar.header("Carga de datos")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo CSV con los datos de los estudiantes")

if uploaded_file is not None:
    # Carga de datos
    df = pd.read_csv(uploaded_file)

    # Configuración de pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Análisis Descriptivo", "Visualización de Datos", "Correlaciones", "Modelos Multivariantes", "Proyecciones", "Análisis de Diferencias entre Grupos"])

    with tab1:
        st.header("Análisis Descriptivo")
        st.write("""
        ### Descripción de los Datos
        En esta sección, puedes ver un resumen estadístico de los datos cargados. Esto incluye medidas como la media, mediana, desviación estándar, valores mínimos y máximos, entre otros.
        """)
        st.write(df.describe(include='all'))

        # Análisis de variables cualitativas
        st.subheader("Análisis de Variables Cualitativas")
        qual_vars = df.select_dtypes(include=['object']).columns.tolist()
        if qual_vars:
            qual_var = st.selectbox("Seleccione una variable cualitativa para analizar", qual_vars)
            st.write(df[qual_var].value_counts())

    with tab2:
        st.header("Visualización de Datos")
        st.write("""
        ### Visualización de los Datos
        Utiliza las opciones a continuación para generar visualizaciones de los datos. Esto puede ayudarte a entender mejor la distribución y las características de las variables.
        """)
        st.subheader("Histograma")
        hist_variable = st.selectbox("Seleccione la variable para el histograma", df.columns.tolist())
        fig_hist = go.Figure(data=[go.Histogram(x=df[hist_variable])])
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Diagrama de Caja")
        box_variable = st.selectbox("Seleccione la variable para el diagrama de caja", df.columns.tolist())
        fig_box = go.Figure(data=[go.Box(y=df[box_variable])])
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Gráfico de Barras para Variables Cualitativas")
        bar_variable = st.selectbox("Seleccione la variable cualitativa para el gráfico de barras", qual_vars)
        if bar_variable:
            fig_bar = go.Figure(data=[go.Bar(x=df[bar_variable].value_counts().index, y=df[bar_variable].value_counts().values)])
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.header("Matriz de Correlación")
        st.write("""
        ### Análisis de Correlaciones
        Selecciona las variables para generar una matriz de correlación que muestre la relación entre ellas. Los valores de correlación varían entre -1 y 1, donde 1 indica una correlación positiva perfecta, -1 una correlación negativa perfecta y 0 indica ninguna correlación.
        """)
        corr_variables = st.multiselect("Seleccione las variables para la matriz de correlación", df.columns.tolist(), default=df.columns.tolist())
        if corr_variables:
            correlation_matrix = df[corr_variables].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='Viridis',
                zmin=-1, zmax=1))
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.header("Modelos Multivariantes")
        st.write("""
        ### Modelos Multivariantes
        En esta sección puedes aplicar diferentes modelos multivariantes a tus datos. Selecciona las variables independientes y dependientes y ajusta los parámetros del modelo según sea necesario.
        """)

        st.subheader("Regresión Lineal")
        x_vars = st.multiselect("Seleccione las variables independientes (X)", df.columns.tolist())
        y_var = st.selectbox("Seleccione la variable dependiente (y)", df.columns.tolist())
        if x_vars and y_var:
            X = df[x_vars]
            y = df[y_var]
            X = sm.add_constant(X)  # Adding a constant
            model = sm.OLS(y, X).fit()
            st.write(model.summary())

            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Gráfico de regresión
            st.subheader("Gráfico de Regresión")
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=y, y=y_pred, mode='markers', name='Datos'))
            fig_reg.add_trace(go.Scatter(x=y, y=y, mode='lines', name='Ajuste'))
            fig_reg.update_layout(title="Gráfico de Regresión", xaxis_title="Valor Real", yaxis_title="Valor Predicho")
            st.plotly_chart(fig_reg, use_container_width=True)

            st.write("""
            #### Interpretación del Modelo de Regresión Lineal:
            - **Coeficientes**: Indican la magnitud y dirección del efecto de cada variable independiente sobre la variable dependiente.
            - **Intercepto**: Valor predicho cuando todas las variables independientes son cero.
            - **MSE**: Indica el promedio del cuadrado de los errores, es decir, la diferencia entre los valores reales y los predichos.
            - **R^2**: Mide qué tan bien el modelo explica la variabilidad de la variable dependiente.
            - **p-valor**: Indica la significancia estadística de cada coeficiente.
            """)

        st.subheader("Modelo de Árbol de Decisión")
        tree_vars = st.multiselect("Seleccione las variables independientes (X) para el árbol", df.columns.tolist())
        tree_y_var = st.selectbox("Seleccione la variable dependiente (y) para el árbol", df.columns.tolist())
        if tree_vars and tree_y_var:
            X_tree = df[tree_vars]
            y_tree = df[tree_y_var]
            tree_depth = st.slider("Profundidad del árbol de decisión", 1, 10, 3)
            model_tree = DecisionTreeClassifier(max_depth=tree_depth)
            model_tree.fit(X_tree, y_tree)
            st.write(f"Importancias de características con profundidad {tree_depth}: {model_tree.feature_importances_}")

            # Gráfico del árbol de decisión
            from sklearn.tree import plot_tree
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(model_tree, feature_names=tree_vars, class_names=True, filled=True, ax=ax)
            st.pyplot(fig)

            st.write("""
            #### Interpretación del Modelo de Árbol de Decisión:
            - **Profundidad del Árbol**: Controla la cantidad de divisiones en el árbol. Una mayor profundidad puede llevar a un modelo más ajustado pero también a sobreajuste.
            - **Importancias de Características**: Indican la importancia relativa de cada variable independiente en la predicción de la variable dependiente.
            """)

        st.subheader("Bosques Aleatorios")
        rf_vars = st.multiselect("Seleccione las variables independientes (X) para el bosque aleatorio", df.columns.tolist())
        rf_y_var = st.selectbox("Seleccione la variable dependiente (y) para el bosque aleatorio", df.columns.tolist())
        if rf_vars and rf_y_var:
            X_rf = df[rf_vars]
            y_rf = df[rf_y_var]
            n_estimators = st.slider("Número de árboles en el bosque", 10, 100, 50)
            model_rf = RandomForestClassifier(n_estimators=n_estimators)
            model_rf.fit(X_rf, y_rf)
            st.write(f"Importancias de características: {model_rf.feature_importances_}")

            # Gráfico de Importancia de Características
            st.subheader("Importancia de Características")
            fig_rf = go.Figure(data=[go.Bar(x=rf_vars, y=model_rf.feature_importances_)])
            fig_rf.update_layout(title="Importancia de Características en el Bosque Aleatorio", xaxis_title="Características", yaxis_title="Importancia")
            st.plotly_chart(fig_rf, use_container_width=True)

            st.write("""
            #### Interpretación del Modelo de Bosque Aleatorio:
            - **Número de Árboles**: Más árboles pueden llevar a una mejor generalización, pero también incrementan el tiempo de computación.
            - **Importancias de Características**: Indican la importancia relativa de cada variable independiente en la predicción de la variable dependiente.
            """)

    with tab5:
        st.header("Proyecciones")
        st.write("""
        ### Análisis de Componentes Principales (PCA)
        Utiliza PCA para reducir la dimensionalidad de los datos y visualizarlos en un espacio de menor dimensión.
        """)
        pca_vars = st.multiselect("Seleccione las variables para PCA", df.columns.tolist())
        if pca_vars:
            X_pca_input = df[pca_vars]
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_pca_input)
            st.write(f"Componentes principales: {pca.components_}")

            # Gráfico de proyección
            fig_pca = go.Figure(
                data=[go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode="markers", marker=dict(color=df[df.columns[0]]))])
            fig_pca.update_layout(title="Gráfico de proyección", xaxis_title="Componente 1", yaxis_title="Componente 2")
            st.plotly_chart(fig_pca, use_container_width=True)

            st.write("""
            #### Interpretación del PCA:
            - **Componentes Principales**: Son combinaciones lineales de las variables originales que capturan la mayor parte de la variabilidad de los datos.
            - **Componente 1 y 2**: Representan las nuevas dimensiones en las cuales los datos son proyectados.
            """)

    with tab6:
        st.header("Análisis de Diferencias entre Grupos")
        st.write("""
        ### Análisis de Diferencias entre Grupos
        Utiliza este análisis para comparar las medias de dos grupos y determinar si son significativamente diferentes.
        """)
        category = st.selectbox("Seleccione la categoría para analizar", df.columns.tolist())
        quant_var = st.selectbox("Seleccione la variable cuantitativa para comparar", df.select_dtypes(include=[np.number]).columns.tolist())
        if category and quant_var:
            group1 = st.selectbox("Seleccione el grupo 1", df[category].unique())
            group2 = st.selectbox("Seleccione el grupo 2", df[category].unique())
            t_stat, p_val = ttest_ind(df[df[category] == group1][quant_var], df[df[category] == group2][quant_var])
            st.write(f"t-stat: {t_stat}, p-valor: {p_val}")

            # ANOVA
            st.subheader("Análisis de Varianza (ANOVA)")
            anova_groups = [df[df[category] == group][quant_var] for group in df[category].unique()]
            f_stat, p_val_anova = f_oneway(*anova_groups)
            st.write(f"F-stat: {f_stat}, p-valor: {p_val_anova}")
# Pie de página
st.markdown("""
---
Elaborado por **Alexander Fernando Haro**, Carrera Administración de Empresas e Inteligencia de Negocios del Instituto Superior Tecnológico España.
""")