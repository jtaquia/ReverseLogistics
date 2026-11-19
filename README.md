# Streamlit + VRPy: flota heterogénea por depósito

## Entradas
- Depósitos: `ID_Deposito,latitud,longitud,NumCamiones`. Una fila por depósito.
- Centros de salud: `ID_CentroSalud,latitud,longitud,PesoDesperdicio` (kg).

La aplicación crea una fila por cada camión, con identificador automático y selector de capacidad **0.5, 1 o 1.5 tn**. Cada depósito puede tener una mezcla distinta.

## Instalación
```bash
conda create -n vrpy_med python=3.11 -y
conda activate vrpy_med
pip install -r requirements.txt
streamlit run app.py
```

## Configuración Google Maps
La aplicación usa Google Distance Matrix API (Legacy), si tu proyecto tiene acceso. Requiere facturación habilitada. Puedes introducir la clave en la interfaz o configurar `GOOGLE_MAPS_API_KEY` en Streamlit Secrets. No publiques la clave en GitHub.

## Modelo
Se asignan centros al depósito con menor distancia de ida (heurística), y VRPy optimiza un problema de flota mixta por depósito mediante `mixed_fleet=True`, con capacidades y número de camiones por tipo. Cada vehículo realiza como máximo una ruta. No se divide la demanda de un centro entre camiones.

## Indicadores
Viajes, distancia, utilización, km/kg, costos parametrizados, emisiones parametrizadas y tiempo VRPy. Exportación a Excel y CSV, con identificador de camión en cada ruta.

## Limitaciones
La versión mantiene una sola configuración editable de flota por ejecución; para comparar escenarios diferentes se deben realizar ejecuciones sucesivas. El mapa presenta puntos, no trazados viales. No se modelan ventanas horarias ni compatibilidad sanitaria. La factibilidad y las rutas deben validarse antes de uso operativo. La instalación de `cspy` y la interfaz de `best_routes_type` deben verificarse con la versión de VRPy utilizada.
