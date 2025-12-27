import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Conv1D

# ==========================================
# 1. 核心邏輯區 (資料與模型)
# ==========================================

@st.cache_data(ttl=3600)
def get_data_with_macro(stock_code):
    ticker = f"{stock_code}.TW"
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close', 'Volume']].rename(columns={'Close': 'Close', 'Volume': 'Volume'})
        
        # 宏觀數據
        macro_tickers = {'^TWII': 'Market_Index', 'TWD=X': 'USD_TWD', '^VIX': 'VIX'}
        for t, name in macro_tickers.items():
            try:
                macro_df = yf.download(t, period="5y", progress=False)
                if isinstance(macro_df.columns, pd.MultiIndex):
                    macro_df.columns = macro_df.columns.get_level_values(0)
                df[name] = macro_df['Close']
            except:
                pass
        
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

def add_indicators(df):
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    df.dropna(inplace=True)
    return df

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    att = MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.1)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(inputs + att)
    ffn = Conv1D(filters=64, kernel_size=1, activation="relu")(x)
    ffn = Dropout(0.1)(ffn)
    ffn = Conv1D(filters=input_shape[-1], kernel_size=1)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 關鍵：遞迴預測函式 (可指定天數)
def predict_recursive(model, data, scaler, window_size, future_days, last_price, feat_count):
    curr_input = data[-window_size:].copy()
    preds = []
    
    for _ in range(future_days):
        inp = curr_input.reshape(1, window_size, feat_count)
        pred_val = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred_val)
        
        new_row = curr_input[-1].copy()
        new_row[0] = pred_val 
        curr_input = np.vstack([curr_input[1:], new_row])
        
    dummy = np.zeros((len(preds), feat_count))
    dummy[:, 0] = np.array(preds)
    real_returns = scaler.inverse_transform(dummy)[:, 0]
    
    prices = []
    curr = last_price
    for r in real_returns:
        curr = curr * np.exp(r)
        prices.append(curr)
    return prices

# ==========================================
# 2. Streamlit 介面設計
# ==========================================
st.set_page_config(page_title="AI 雙模式預測", page_icon="📈")

st.title("📈 Transformer AI 雙模式預測")
st.markdown("此 App 同時提供 **「明日極速預測」** 與 **「未來5日波段預測」**。")

with st.sidebar:
    st.header("⚙️ 設定")
    stock_code = st.text_input("股票代號", value="2330")
    epochs = st.slider("訓練強度 (Epochs)", 10, 60, 30)
    st.info("點擊按鈕後請稍候，AI 正在現場訓練模型...")

if st.button("🚀 執行雙重預測"):
    status = st.empty()
    bar = st.progress(0)
    
    # 1. 數據
    status.text("正在下載與處理數據...")
    df = get_data_with_macro(stock_code)
    
    if df is None or len(df) < 200:
        st.error("數據不足或下載失敗")
    else:
        bar.progress(20)
        df = add_indicators(df)
        
        # 特徵準備
        cols = ['Log_Return', 'RSI', 'MACD', 'Volatility', 'Volume']
        if 'Market_Index' in df.columns: 
            df['Market_Return'] = np.log(df['Market_Index'] / df['Market_Index'].shift(1)).fillna(0)
            cols.append('Market_Return')
        if 'USD_TWD' in df.columns: cols.append('USD_TWD')
        if 'VIX' in df.columns: cols.append('VIX')
        
        data = df[cols].values
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        ws = 60
        X, y = [], []
        for i in range(ws, len(scaled)):
            X.append(scaled[i-ws:i])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        
        split = int(len(X) * 0.9)
        X_train, y_train = X[:split], y[:split]
        
        # 2. 訓練
        bar.progress(40)
        status.text(f"正在訓練 Transformer 模型 (Epochs: {epochs})...")
        model = build_model((ws, len(cols)))
        cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[cb], verbose=0)
        
        # 3. 預測
        bar.progress(80)
        status.text("正在計算雙模式預測結果...")
        
        last_price = df['Close'].iloc[-1]
        
        # 預測未來 5 天 (包含了第 1 天)
        future_prices = predict_recursive(model, scaled, scaler, ws, 5, last_price, len(cols))
        
        price_1day = future_prices[0]  # 第1天
        price_5day = future_prices[-1] # 第5天
        
        bar.progress(100)
        status.success("計算完成！請切換下方分頁查看結果。")
        
        # === 雙模式分頁顯示 ===
        tab1, tab2 = st.tabs(["🚀 明日預測 (1 Day)", "🌊 波段預測 (5 Days)"])
        
        # --- TAB 1: 1天預測 ---
        with tab1:
            st.subheader(f"📅 明日股價預測")
            
            d1_diff = price_1day - last_price
            d1_pct = (d1_diff / last_price) * 100
            d1_color = "green" if d1_diff > 0 else "red"
            
            col1, col2 = st.columns(2)
            col1.metric("目前股價", f"{last_price:.2f}")
            col2.metric("明日預測價", f"{price_1day:.2f}", f"{d1_diff:+.2f} ({d1_pct:+.2f}%)")
            
            if abs(d1_pct) < 0.5:
                advice = "➡️ 盤整機率高"
            elif d1_pct > 0:
                advice = "🔥 看漲"
            else:
                advice = "❄️ 看跌"
            st.info(f"AI 短線建議: {advice}")

        # --- TAB 2: 5天預測 ---
        with tab2:
            st.subheader(f"🌊 未來 5 日趨勢")
            
            d5_diff = price_5day - last_price
            d5_pct = (d5_diff / last_price) * 100
            
            st.metric("5日後目標價", f"{price_5day:.2f}", f"{d5_diff:+.2f} ({d5_pct:+.2f}%)")
            
            # 畫圖
            fig, ax = plt.subplots(figsize=(10, 5))
            # 真實 (近30天)
            ax.plot(df.index[-30:], df['Close'].iloc[-30:], label="歷史股價", color='blue')
            # 預測 (未來5天)
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
            ax.plot(future_dates, future_prices, label="AI 預測路徑", color='green', marker='o', linestyle='--')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 表格
            res_df = pd.DataFrame({
                "日期": future_dates.strftime('%Y-%m-%d'),
                "預測股價": [f"{p:.2f}" for p in future_prices]
            })
            st.table(res_df)
