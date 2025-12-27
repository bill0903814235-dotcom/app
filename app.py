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
# 1. 數據獲取與特徵工程 (含快取機制)
# ==========================================

# 使用 @st.cache_data 快取函式結果，避免每次操作都重新下載和計算
@st.cache_data(ttl=3600) # 快取有效時間為 1 小時
def get_and_process_data(stock_code):
    # --- 1.1 數據獲取 ---
    ticker = f"{stock_code}.TW"
    try:
        # 下載個股數據
        df = yf.download(ticker, period="5y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close', 'Volume']].rename(columns={'Close': 'Close', 'Volume': 'Volume'})
        
        # 下載宏觀數據
        macro_tickers = {'^TWII': 'Market_Index', 'TWD=X': 'USD_TWD', '^VIX': 'VIX'}
        for t, name in macro_tickers.items():
            try:
                macro_df = yf.download(t, period="5y", progress=False)
                if isinstance(macro_df.columns, pd.MultiIndex):
                    macro_df.columns = macro_df.columns.get_level_values(0)
                df[name] = macro_df['Close']
            except:
                # 若某項宏觀數據下載失敗，則略過
                pass
        
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        
        if len(df) < 200:
            return None, "數據不足，無法訓練模型。"

    except Exception as e:
        return None, f"數據下載失敗: {e}"

    # --- 1.2 特徵工程 ---
    # A. 轉換目標：對數收益率
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # B. 技術指標
    # 波動率
    df['Volatility'] = df['Log_Return'].rolling(window=20).std()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    df.dropna(inplace=True)

    return df, None

# ==========================================
# 2. 模型相關函式
# ==========================================

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    # Attention Block
    attention_output = MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.1)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    # Feed Forward Block
    ffn_output = Conv1D(filters=64, kernel_size=1, activation="relu")(x)
    ffn_output = Dropout(0.1)(ffn_output)
    ffn_output = Conv1D(filters=input_shape[-1], kernel_size=1)(ffn_output)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    # Output Head
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_recursive_future(model, data_scaled, scaler, window_size, future_days, last_close_price, feature_count):
    curr_input = data_scaled[-window_size:].copy()
    predicted_returns = []
    
    for _ in range(future_days):
        pred_input = curr_input.reshape(1, window_size, feature_count)
        pred_scaled = model.predict(pred_input, verbose=0)
        pred_val = pred_scaled[0, 0]
        predicted_returns.append(pred_val)
        
        new_row = curr_input[-1].copy()
        new_row[0] = pred_val 
        curr_input = np.vstack([curr_input[1:], new_row])
        
    dummy_array = np.zeros((len(predicted_returns), feature_count))
    dummy_array[:, 0] = np.array(predicted_returns)
    real_returns = scaler.inverse_transform(dummy_array)[:, 0]
    
    predicted_prices = []
    curr_price = last_close_price
    for ret in real_returns:
        next_price = curr_price * np.exp(ret)
        predicted_prices.append(next_price)
        curr_price = next_price
    return predicted_prices

# ==========================================
# 3. Streamlit App 介面
# ==========================================

# 設定頁面標題和圖示
st.set_page_config(page_title="AI 股價預言家", page_icon="📈")

st.title("📈 Transformer AI 台股預測 App")
st.markdown("輸入台股代號，AI 將自動整合個股、大盤、匯率及 VIX 數據，利用 **Transformer 模型** 進行分析與預測。")

# 側邊欄：使用者輸入與設定
with st.sidebar:
    st.header("⚙️ 設定參數")
    stock_code = st.text_input("股票代號 (例如: 2330)", value="2330")
    epochs = st.slider("訓練次數 (Epochs)", min_value=10, max_value=100, value=30, step=10, help="訓練次數越多，模型學得越久，但也可能過擬合。建議 30-50。")
    st.markdown("---")
    st.info("💡 **提示**：首次執行或更換股票代號時，因需重新下載數據與訓練模型，請耐心等待約 1-3 分鐘。")

# 主畫面：開始預測按鈕
if st.button("🚀 開始 AI 分析與預測"):
    # 用一個進度條和狀態文字來提示使用者
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- 階段 1: 數據準備 ---
    status_text.text("1/4 正在下載與處理數據...")
    df, error_msg = get_and_process_data(stock_code)
    progress_bar.progress(25)
    
    if error_msg:
        st.error(error_msg)
    else:
        # --- 階段 2: 特徵工程與資料分割 ---
        status_text.text("2/4 正在建立特徵並準備訓練資料...")
        
        # 定義特徵欄位
        feature_cols = ['Log_Return', 'RSI', 'MACD', 'Volatility', 'Volume']
        if 'Market_Index' in df.columns: 
            df['Market_Return'] = np.log(df['Market_Index'] / df['Market_Index'].shift(1)).fillna(0)
            feature_cols.append('Market_Return')
        if 'USD_TWD' in df.columns: feature_cols.append('USD_TWD')
        if 'VIX' in df.columns: feature_cols.append('VIX')
        
        # 資料標準化
        data = df[feature_cols].values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 製作時間序列資料
        prediction_days = 60
        X, y = [], []
        for i in range(prediction_days, len(scaled_data)):
            X.append(scaled_data[i-prediction_days:i])
            y.append(scaled_data[i, 0]) # 預測 Log_Return
        X, y = np.array(X), np.array(y)
        
        # 分割訓練集 (90%)
        split_idx = int(len(X) * 0.9)
        X_train, y_train = X[:split_idx], y[:split_idx]
        progress_bar.progress(50)
        
        # --- 階段 3: 模型訓練 ---
        status_text.text(f"3/4 正在訓練 Transformer 模型 (Epochs: {epochs})...")
        
        # 建立模型
        model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
        
        # 設定早停機制
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # 開始訓練 (verbose=0 不顯示詳細訓練過程)
        with st.spinner("模型訓練中，這可能需要一點時間..."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[callback], verbose=0)
        progress_bar.progress(75)
            
        # --- 階段 4: 進行預測 ---
        status_text.text("4/4 正在進行盲測驗證與未來預測...")
        
        # A. 10日盲測驗證 (Backtest)
        test_days = 10
        start_test_idx = len(scaled_data) - test_days
        base_price_backtest = df['Close'].iloc[start_test_idx - 1]
        input_data_backtest = scaled_data[start_test_idx - prediction_days : start_test_idx]
        
        blind_predicted_prices = predict_recursive_future(
            model, input_data_backtest, scaler, prediction_days, test_days, base_price_backtest, len(feature_cols)
        )
        
        # B. 未來 5 日預測 (Forecast)
        future_days = 5
        last_close = df['Close'].iloc[-1]
        future_prices = predict_recursive_future(
            model, scaled_data, scaler, prediction_days, future_days, last_close, len(feature_cols)
        )
        
        progress_bar.progress(100)
        status_text.success("✅ 分析完成！請查看下方結果。")
        
        # --- 結果展示 ---
        st.markdown("---")
        st.header(f"📊 {stock_code} 分析報告")
        
        # 計算日期
        last_date = df.index[-1]
        future_dates = []
        temp_date = last_date
        for _ in range(future_days):
            temp_date += pd.Timedelta(days=1)
            while temp_date.weekday() >= 5: # 跳過週末
                temp_date += pd.Timedelta(days=1)
            future_dates.append(temp_date)
            
        # 1. 關鍵指標卡片
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最新收盤價", f"{last_close:.2f}", f"日期: {last_date.strftime('%Y-%m-%d')}")
        with col2:
            target_price = future_prices[-1]
            diff = target_price - last_close
            pct = (diff / last_close) * 100
            st.metric("5日後目標價", f"{target_price:.2f}", f"{diff:+.2f} ({pct:+.2f}%)")
        with col3:
            trend = "🔥 強力看漲" if pct > 3 else "📈 看漲" if pct > 0.5 else "📉 看跌" if pct < -0.5 else "❄️ 強力看跌" if pct < -3 else "➡️ 盤整"
            st.metric("AI 趨勢建議", trend)
            
        # 2. 互動式走勢圖
        st.subheader("📈 走勢圖：真實 vs 盲測 vs 未來預測")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 繪製真實股價 (最近 60 天)
        plot_lookback = 60
        recent_dates = df.index[-plot_lookback:]
        recent_prices = df['Close'].iloc[-plot_lookback:]
        ax.plot(recent_dates, recent_prices, label="真實股價", color='blue', linewidth=2, alpha=0.7)
        
        # 繪製盲測驗證 (紅線)
        blind_dates = df.index[-test_days:]
        ax.plot(blind_dates, blind_predicted_prices, label="AI 盲測 (過去10天)", color='red', linestyle='--', marker='o', markersize=4)
        
        # 繪製未來預測 (綠線)
        ax.plot(future_dates, future_prices, label="AI 預測 (未來5天)", color='green', linestyle='-', marker='x', markersize=6, linewidth=2)
        
        # 圖表設定
        ax.set_title(f"{stock_code} Price Prediction Analysis", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (TWD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # 3. 未來價格數據表
        st.subheader("📋 未來 5 日預測數據明細")
        forecast_df = pd.DataFrame({
            "日期": [d.strftime('%Y-%m-%d') for d in future_dates],
            "星期": [d.strftime('%A') for d in future_dates],
            "預測股價": [f"{p:.2f}" for p in future_prices]
        })
        # 將星期轉換為中文
        weekdays_map = {'Monday': '一', 'Tuesday': '二', 'Wednesday': '三', 'Thursday': '四', 'Friday': '五'}
        forecast_df['星期'] = forecast_df['星期'].map(weekdays_map)
        
        st.table(forecast_df)

else:
    # 初始畫面提示
    st.info("👈 請在左側輸入股票代號，並點擊「開始 AI 分析與預測」按鈕。")