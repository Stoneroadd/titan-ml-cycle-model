# Titan ML Cycle Model

MVP de machine learning para entrenar una regresión lineal desde cero y estimar tiempo de ciclo con un dataset sintético de acarreo.

## Ejecutar

Requiere Python 3.10 o superior, sin dependencias externas.

```powershell
python train.py data/cycles.csv
```

El script usa gradiente descendente, divide los datos en entrenamiento y prueba por orden temporal, y reporta MAE, RMSE y R². No usa datos de una faena real y no sirve para planificar operaciones.

## Variables

`distance_km`, `payload_t`, `queue_min` son características; `cycle_time_min` es el objetivo. El modelo escala las características con estadísticas del conjunto de entrenamiento.

## Alcance

Es una demostración didáctica de regresión y evaluación. El pequeño dataset sintético no respalda conclusiones de producción ni recomendaciones de despacho.

## Licencia

MIT para código y dataset.
