# Observatorio Regional de Nariño

Análisis Económico, Territorial y Competitividad Municipal

Aplicación web interactiva desarrollada con **Streamlit** para el análisis de indicadores económicos, sociales y de competitividad de los municipios del Departamento de Nariño (Colombia).

## Características

- Datos oficiales / aproximaciones documentadas (DANE + CPC)
- KPIs comparativos Nariño vs Colombia
- Cartografía interactiva
- Ranking municipal y gráficos de dispersión
- Econometría espacial (Moran Global + LISA)
- Estadística descriptiva completa
- Matriz de correlaciones y regresión OLS (Competitividad ~ Educación)

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub
2. Ve a https://share.streamlit.io
3. Selecciona el repositorio y el archivo `app.py`
4. Streamlit instalará automáticamente las dependencias de `requirements.txt`

## Autor

Economista Investigador Senior – Econometría Espacial y Desarrollo Regional
