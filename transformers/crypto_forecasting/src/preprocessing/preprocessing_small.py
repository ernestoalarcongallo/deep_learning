import talib as ta
import pandas as pd
import numpy as np
import keras
import tensorflow as tf


def preprocess(df: pd.DataFrame, num_bars: int = 1):
    df = df.copy()
    df['open'] = df['close'].shift(num_bars)
    df['rsi'] = ta.RSI(df['close'], timeperiod=14 * num_bars)
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'],
                                                           fastperiod=12 * num_bars,
                                                           slowperiod=26 * num_bars,
                                                           signalperiod=9 * num_bars)
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14 * num_bars)
    df['ema4'] = ta.EMA(df['close'], timeperiod=4 * num_bars)
    df['ema10'] = ta.EMA(df['close'], timeperiod=10 * num_bars)
    df['ema20'] = ta.EMA(df['close'], timeperiod=20 * num_bars)
    
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(df['close'],
                                                               timeperiod=5 * num_bars,
                                                               nbdevup=2,
                                                               nbdevdn=2,
                                                               matype=0)
    
    df['aroondown'], df['aroonup'] = ta.AROON(df['high'], df['low'], timeperiod=14)
    df['money_flow'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['momentum'] = ta.MOM(df['close'], timeperiod=10)
    df['william'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['rate_of_change'] = ta.ROC(df['close'], timeperiod=10)
    
    df['CDL2CROWS'] = ta.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDL3INSIDE'] = ta.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
    df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
    df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
    df['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
    df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
    df['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'])
    df['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])
    df['CDLBELTHOLD'] = ta.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])
    df['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])
    df['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    df['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])
    df['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])
    df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
    df['CDLDOJI'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLDOJISTAR'] = ta.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLENGULFING'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])
    df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLHAMMER'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
    df['CDLHARAMI'] = ta.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
    df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
    df['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])
    df['CDLHIKKAKE'] = ta.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close'])
    df['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close'])
    df['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close'])
    df['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDLINNECK'] = ta.CDLINNECK(df['open'], df['high'], df['low'], df['close'])
    df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
    df['CDLKICKING'] = ta.CDLKICKING(df['open'], df['high'], df['low'], df['close'])
    df['CDLKICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close'])
    df['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close'])
    df['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])
    df['CDLLONGLINE'] = ta.CDLLONGLINE(df['open'], df['high'], df['low'], df['close'])
    df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    df['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close'])
    df['CDLMATHOLD'] = ta.CDLMATHOLD(df['open'], df['high'], df['low'], df['close'])
    df['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLONNECK'] = ta.CDLONNECK(df['open'], df['high'], df['low'], df['close'])
    df['CDLPIERCING'] = ta.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
    df['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close'])
    df['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])
    df['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close'])
    df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLSHORTLINE'] = ta.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close'])
    df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
    df['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close'])
    df['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close'])
    df['CDLTAKURI'] = ta.CDLTAKURI(df['open'], df['high'], df['low'], df['close'])
    df['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close'])
    df['CDLTHRUSTING'] = ta.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close'])
    df['CDLTRISTAR'] = ta.CDLTRISTAR(df['open'], df['high'], df['low'], df['close'])
    df['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close'])
    df['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close'])
    df['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close'])

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
