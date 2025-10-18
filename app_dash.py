fig_diff_temp = px.scatter(df_diff_temp, x='TempMedia', y='Diferencia', color='Modelo',
                           title=f'Diferencias vs Temp Media - {code}', opacity=0.7)  # Quita trendline='lowess'

fig_diff_rs = px.scatter(df_diff_rs, x='Radiacion', y='Diferencia', color='Modelo',
                         title=f'Diferencias vs Radiación - {code}', opacity=0.7)  # Quita trendline='lowess'