import talib as ta
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame, num_bars: int = 1):
    df = df.copy()
    
    # Artificially create open value
    df['open'] = df['close'].shift(num_bars)

    # Overlap studies
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['close'],
                                                           timeperiod=5,
                                                           nbdevup=2,
                                                           nbdevdn=2,
                                                           matype=0)
    df = df.copy()
    df['dema'] = ta.DEMA(df['close'], timeperiod=30)
    df = df.copy()
    df['ema4'] = ta.EMA(df['close'], timeperiod=4)
    df = df.copy()
    df['ema10'] = ta.EMA(df['close'], timeperiod=10)
    df = df.copy()
    df['ema20'] = ta.EMA(df['close'], timeperiod=20) 
    df = df.copy()
    df['ema30'] = ta.EMA(df['close'], timeperiod=30) 
    df = df.copy()
    df['ht_trendline'] = ta.HT_TRENDLINE(df['close']) 
    df = df.copy()
    df['kama'] = ta.KAMA(df['close'], timeperiod=30) 
    df = df.copy()
    df['ma5'] = ta.MA(df['close'], timeperiod=5) 
    df = df.copy()
    df['ma10'] = ta.MA(df['close'], timeperiod=10) 
    df = df.copy()
    df['ma20'] = ta.MA(df['close'], timeperiod=20) 
    df = df.copy()
    df['ma50'] = ta.MA(df['close'], timeperiod=50) 
    df = df.copy()
    df['ma100'] = ta.MA(df['close'], timeperiod=100) 
    df = df.copy()
    df['ma200'] = ta.MA(df['close'], timeperiod=200) 
    df = df.copy()
    df['mama'], df['fama'] = ta.MAMA(df['close'], fastlimit=0.02, slowlimit=0.20)
    df = df.copy()
    df['midpoint'] = ta.MIDPOINT(df['close'], timeperiod=14) 
    df = df.copy()
    df['midprice'] = ta.MIDPRICE(df['high'], df['low'], timeperiod=14) 
    df = df.copy()
    df['sar'] = ta.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2) 
    df = df.copy()
    df['tema3'] = ta.TEMA(df['close'], timeperiod=3) 
    df = df.copy()
    df['tema5'] = ta.TEMA(df['close'], timeperiod=5) 
    df = df.copy()
    df['tema10'] = ta.TEMA(df['close'], timeperiod=10) 
    df = df.copy()
    df['tema20'] = ta.TEMA(df['close'], timeperiod=20) 
    df = df.copy()
    df['tema30'] = ta.TEMA(df['close'], timeperiod=30) 
    df = df.copy()
    df['trima30'] = ta.TRIMA(df['close'], timeperiod=30) 
    df = df.copy()
    df['wma30'] = ta.WMA(df['close'], timeperiod=30) 
    df = df.copy()
    
    # Momentum indicators
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['adxr'] = ta.ADXR(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['apo'] = ta.APO(df['close'], fastperiod=12, slowperiod=26, matype=0) 
    df = df.copy()
    df['aroondown'], df['aroonup'] = ta.AROON(df['high'], df['low'], timeperiod=14)
    df = df.copy()
    df['aroonosc'] = ta.AROONOSC(df['high'], df['low'], timeperiod=14) 
    df = df.copy()
    df['bop'] = ta.BOP(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['cmo'] = ta.CMO(df['close'], timeperiod=14) 
    df = df.copy()
    df['dx'] = ta.DX(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'],
                                                           fastperiod=12,
                                                           slowperiod=26,
                                                           signalperiod=9)
    df = df.copy()
    df['mfi'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df = df.copy()
    df['minus_di'] = ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['minus_dm'] = ta.MINUS_DM(df['high'], df['low'], timeperiod=14) 
    df = df.copy()
    df['mom'] = ta.MOM(df['close'], timeperiod=10) 
    df = df.copy()
    df['plus_di'] = ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['plus_dm'] = ta.PLUS_DM(df['high'], df['low'], timeperiod=14) 
    df = df.copy()
    df['ppo'] = ta.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0) 
    df = df.copy()
    df['roc'] = ta.ROC(df['close'], timeperiod=10) 
    df = df.copy()
    df['rocp'] = ta.ROCP(df['close'], timeperiod=10) 
    df = df.copy()
    df['rocr'] = ta.ROCR(df['close'], timeperiod=10) 
    df = df.copy()
    df['rsi'] = ta.RSI(df['close'], timeperiod=14 * num_bars) 
    df = df.copy()
    df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df = df.copy()
    df['fastk'], df['fastd'] = ta.STOCHF(df['high'], df['low'], df['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    df = df.copy()
    df['fastk_rsi'], df['fastd_rsi'] = ta.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df = df.copy()
    df['trix'] = ta.TRIX(df['close'], timeperiod=30) 
    df = df.copy()
    df['ultosc'] = ta.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28) 
    df = df.copy()
    df['willr'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()

    # Volume indicators
    df['ad'] = ta.AD(df['high'], df['low'], df['close'], df['volume']) 
    df = df.copy()
    df['adosc'] = ta.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10) 
    df = df.copy()
    df['obv'] = ta.OBV(df['close'], df['volume']) 
    df = df.copy()
    
    # Cycle indicators
    df['ht_dcperiod'] = ta.HT_DCPERIOD(df['close']) 
    df = df.copy()
    df['ht_dcphase'] = ta.HT_DCPHASE(df['close']) 
    df = df.copy()
    df['inphase'], df['quadrature'] = ta.HT_PHASOR(df['close'])
    df = df.copy()
    df['sine'], df['leadsine'] = ta.HT_SINE(df['close'])
    df = df.copy()
    df['ht_trendmode'] = ta.HT_TRENDMODE(df['close']) 
    df = df.copy()
    
    # Volatility indicators
    df['trange'] = ta.TRANGE(df['high'], df['low'], df['close']) 
    df = df.copy()
    df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()
    df['natr'] = ta.NATR(df['high'], df['low'], df['close'], timeperiod=14) 
    df = df.copy()

    # Statistic Functions
    df['beta'] = ta.BETA(df['high'], df['low'], timeperiod=5) 
    df = df.copy()
    df['correl'] = ta.CORREL(df['high'], df['low'], timeperiod=30) 
    df = df.copy()
    df['linearreg'] = ta.LINEARREG(df['close'], timeperiod=14) 
    df = df.copy()
    df['linearreg_angle'] = ta.LINEARREG_ANGLE(df['close'], timeperiod=14) 
    df = df.copy()
    df['linearreg_intercep'] = ta.LINEARREG_INTERCEPT(df['close'], timeperiod=14) 
    df = df.copy()
    df['linearreg_slope'] = ta.LINEARREG_SLOPE(df['close'], timeperiod=14) 
    df = df.copy()
    df['stddev'] = ta.STDDEV(df['close'], timeperiod=5, nbdev=1) 
    df = df.copy()
    df['tsf'] = ta.TSF(df['close'], timeperiod=14) 
    df = df.copy()
    
    # Pattern recognition
    df['CDL2CROWS'] = ta.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
    df = df.copy()
    df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDL3INSIDE'] = ta.CDL3INSIDE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLBELTHOLD'] = ta.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLDOJI'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLDOJISTAR'] = ta.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLENGULFING'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHAMMER'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHARAMI'] = ta.CDLHARAMI(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHIKKAKE'] = ta.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLINNECK'] = ta.CDLINNECK(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLKICKING'] = ta.CDLKICKING(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLLONGLINE'] = ta.CDLLONGLINE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLMATHOLD'] = ta.CDLMATHOLD(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLONNECK'] = ta.CDLONNECK(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLPIERCING'] = ta.CDLPIERCING(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLSHORTLINE'] = ta.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLTAKURI'] = ta.CDLTAKURI(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLTHRUSTING'] = ta.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLTRISTAR'] = ta.CDLTRISTAR(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()
    df['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close']) 
    df = df.copy()

    df.reset_index(inplace=True, drop=True)
    df.dropna(inplace=True)

    return df

def window_series(tensor_slice, step, enc_steps_in, dec_steps_in, dec_steps_out, n_features, labels_dimension=0):
    def mask_labels(tensor, dim=labels_dimension):
        mask = np.zeros((dec_steps_out, n_features))
        mask[:, dim] = 1
        mask = tf.convert_to_tensor(mask, dtype=tf.float64)
        
        return tf.math.multiply(mask, tensor)

    dataset = tf.data.Dataset.from_tensor_slices(tensor_slice)
    dataset = dataset.window(enc_steps_in + dec_steps_in + dec_steps_out, shift=step, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(enc_steps_in + dec_steps_in + dec_steps_out))
    dataset = dataset.map(lambda window: (window[:enc_steps_in], 
                                          window[enc_steps_in:enc_steps_in+dec_steps_in], 
                                          mask_labels(window[enc_steps_in+dec_steps_in:])))

    return dataset

def make_dataset(filepath, **kwargs):
    step = kwargs['step']
    val_date = kwargs['val_date']
    test_date = kwargs['test_date']
    enc_steps_in = kwargs['enc_steps_in']
    dec_steps_in = kwargs['dec_steps_in']
    dec_steps_out = kwargs['dec_steps_out']
    excluded_features = kwargs['excluded_features']    
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = [x.lower() for x in df.columns]
    df.drop(columns=['open', 'adj close'], inplace=True)
    
    df = preprocess(df)
    
    features_list = ['close'] + [f for f in df.columns if f not in excluded_features and f != 'close']
    n_features = len(features_list)
    
    if val_date and test_date:
        train_df = df[df['date'] < val_date]
        val_df = df[(df['date'] >= val_date) & (df['date'] < test_date)]
        test_df = df[df['date'] >= test_date]
        
        scaler = StandardScaler()
        train_dataset = window_series(scaler.fit_transform(train_df[features_list].values.tolist()), 
                                      step, 
                                      enc_steps_in, 
                                      dec_steps_in, 
                                      dec_steps_out,
                                      n_features)
        
        val_dataset = window_series(scaler.transform(val_df[features_list].values.tolist()), 
                            step, 
                            enc_steps_in, 
                            dec_steps_in, 
                            dec_steps_out,
                            n_features)

        test_dataset = window_series(scaler.transform(test_df[features_list].values.tolist()), 
                                     step, 
                                     enc_steps_in, 
                                     dec_steps_in, 
                                     dec_steps_out,
                                     n_features)
        
        X_train_enc = np.array([enc_x.numpy() for enc_x, _, _ in train_dataset])
        X_train_dec = np.array([dec_x.numpy() for _, dec_x, _ in train_dataset])
        y_train = np.array([dec_y.numpy() for _, _, dec_y in train_dataset])

        X_val_enc = np.array([enc_x.numpy() for enc_x, _, _ in val_dataset])
        X_val_dec = np.array([dec_x.numpy() for _, dec_x, _ in val_dataset])
        y_val = np.array([dec_y.numpy() for _, _, dec_y in val_dataset])

        X_test_enc = np.array([enc_x.numpy() for enc_x, _, _ in test_dataset])
        X_test_dec = np.array([dec_x.numpy() for _, dec_x, _ in test_dataset])
        y_test = np.array([dec_y.numpy() for _, _, dec_y in test_dataset])
    else:
        train_df = df
        scaler = StandardScaler()
        train_dataset = window_series(scaler.fit_transform(train_df[features_list].values.tolist()), 
                                      step, 
                                      enc_steps_in, 
                                      dec_steps_in, 
                                      dec_steps_out,
                                      n_features)
        
        X_train_enc = np.array([enc_x.numpy() for enc_x, _, _ in train_dataset])
        X_train_dec = np.array([dec_x.numpy() for _, dec_x, _ in train_dataset])
        y_train = np.array([dec_y.numpy() for _, _, dec_y in train_dataset])
        
        X_val_enc, X_val_dec, y_val, X_test_enc, X_test_dec, y_test, scaler = None, None, None, None, None, None, None
        
    return X_train_enc, X_train_dec, y_train, X_val_enc, X_val_dec, y_val, X_test_enc, X_test_dec, y_test, scaler, features_list
