# %%
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def detect_structure_points(df, window):
    """
    Détecte les points de structure avec validation de cassure et retournement
    """
    df = df.copy()
    df['isPivotHigh'] = False
    df['isPivotLow'] = False
    
    last_valid_high = None
    last_valid_low = None
    last_high_price = None
    last_low_price = None
    
    for i in range(window, len(df) - window):
        current_candle = df.iloc[i]
        next_candle = df.iloc[i + 1] if i + 1 < len(df) else None
        
        # Mise à jour des derniers points valides si None
        if last_valid_high is None:
            if is_local_high(df, i, window):
                last_valid_high = i
                last_high_price = current_candle['high']
        if last_valid_low is None:
            if is_local_low(df, i, window):
                last_valid_low = i
                last_low_price = current_candle['low']
                
        # Vérification de cassure de point haut
        if last_valid_high is not None and current_candle['high'] > last_high_price:
            # Vérification du retournement sur la bougie suivante
            if next_candle is not None and next_candle['close'] < next_candle['open']:
                # Marquer le nouveau point haut
                df.loc[i, 'isPivotHigh'] = True
                
                # Trouver le point bas entre les deux points hauts
                low_section = df.iloc[last_valid_high:i+1]
                min_idx = low_section['low'].idxmin()
                df.loc[min_idx, 'isPivotLow'] = True
                
                # Mettre à jour les références
                last_valid_high = i
                last_high_price = current_candle['high']
                last_valid_low = min_idx
                last_low_price = df.iloc[min_idx]['low']
        
        # Vérification de cassure de point bas
        if last_valid_low is not None and current_candle['low'] < last_low_price:
            # Vérification du retournement sur la bougie suivante
            if next_candle is not None and next_candle['close'] > next_candle['open']:
                # Marquer le nouveau point bas
                df.loc[i, 'isPivotLow'] = True
                
                # Trouver le point haut entre les deux points bas
                high_section = df.iloc[last_valid_low:i+1]
                max_idx = high_section['high'].idxmax()
                df.loc[max_idx, 'isPivotHigh'] = True
                
                # Mettre à jour les références
                last_valid_low = i
                last_low_price = current_candle['low']
                last_valid_high = max_idx
                last_high_price = df.iloc[max_idx]['high']
    
    return df

def is_local_high(df, index, window):
    """
    Vérifie si un point est un sommet local
    """
    if index - window < 0 or index + window >= len(df):
        return False
    current_high = df.iloc[index]['high']
    return all(current_high >= df.iloc[i]['high'] for i in range(index-window, index+window+1))

def is_local_low(df, index, window):
    """
    Vérifie si un point est un creux local
    """
    if index - window < 0 or index + window >= len(df):
        return False
    current_low = df.iloc[index]['low']
    return all(current_low <= df.iloc[i]['low'] for i in range(index-window, index+window+1))

def create_chart(df, start_idx=4000, end_idx=5000):
    """
    Crée le graphique avec les points de structure
    """
    dfpl = df[start_idx:end_idx]
    
    fig = go.Figure(data=[go.Candlestick(
        x=dfpl.index,
        open=dfpl['open'],
        high=dfpl['high'],
        low=dfpl['low'],
        close=dfpl['close'],
        name='EURUSD'
    )])
    
    # Ajout des points de pivot hauts
    fig.add_trace(go.Scatter(
        x=dfpl[dfpl['isPivotHigh']].index,
        y=dfpl[dfpl['isPivotHigh']]['high'] + 1e-3,
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='Pivot Haut'
    ))
    
    # Ajout des points de pivot bas
    fig.add_trace(go.Scatter(
        x=dfpl[dfpl['isPivotLow']].index,
        y=dfpl[dfpl['isPivotLow']]['low'] - 1e-3,
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name='Pivot Bas'
    ))
    
    fig.update_layout(
        title='Analyse technique EURUSD avec points de structure',
        yaxis_title='Prix',
        xaxis_title='Période',
        template='plotly_dark'
    )
    
    return fig

def main():
    # Chargement des données
    df = pd.read_csv("EURUSD_Candlestick_1_Hour_BID_04.05.2003-15.04.2023.csv")
    df = df[df['volume'] != 0].reset_index(drop=True)
    df = df.head(5000)
    
    # Détection des points de structure
    df = detect_structure_points(df, window=1)
    
    # Création et affichage du graphique
    fig = create_chart(df)
    fig.show()
    
    # Statistiques
    print(f"Nombre de points hauts de structure : {df['isPivotHigh'].sum()}")
    print(f"Nombre de points bas de structure : {df['isPivotLow'].sum()}")
    
    return df

if __name__ == "__main__":
    df = main()
    


