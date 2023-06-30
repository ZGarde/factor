import numpy as np
import pandas as pd
from numpy import fabs as npfabs
from scipy.stats import norm
from scipy import stats
"""
RMA计算指数加权移动平均线
"""
def RMA(close, N):
    rma = close.ewm(alpha=1/N, min_periods=N).mean()
    return rma

"""
MAD计算N天内平的数值
MAD = mean(abs((N天的收盘价格-N天的平均价格)))
"""
def MAD(close, N):
    def mad_(series):
        return npfabs(series - series.mean()).mean()
    
    mad = close.rolling(N, min_periods=N).apply(mad_, raw=True)
    return mad

"""
SMA(C,N)移动平均
SMA = mean(C, N)
"""
def SMA(close, n):
    return close.rolling(n, min_periods=n).mean()


"""
WMA(x,m)-加权移动平均，这个函数对于近日的权重会比其它函数敏感

逆序累加，最后一个元素会参与n遍计算，第一个元素只会参与1次计算，所以总共有n*(n + 1)/2个数的平均
"""
def WMA(close, n):
    return close.rolling(n).apply(lambda x: x[::-1].cumsum().sum() * 2 / n / (n + 1))


"""
EMA是求指数平滑移动平均EMA(X,N),求X的N日指数平滑移动平均。
若Y=EMA(X,N)，则Y=[2*X+(N-1)*Y']/(N+1),其中Y'表示上一周期Y值。
"""
def EMA(close, n):
    return close.ewm(span=n, adjust=False).mean()

"""
MAX函数求出close和close2中较大的数字
MAX = if (close > close2) close else close2
"""
def MAX(close, close2):
    return (close > close2) * close + (close <= close2) * close2

"""
MA3求简单移动平均,MA3(X),求X的3日移动平均值。
算法：(X1+X2+X3)/3
"""
class MA3:
    def __init__(self):
        self.name = 'ma3'
        self.vars = ['close']

    def run(self, d):
        return d["close"].rolling(3).mean()

"""
MA6求简单移动平均,MA6(X),求X的6日移动平均值。
算法：(X1+X2+X3+...+X6)/6
"""
class MA6:
    def __init__(self):
        self.name = 'ma6'
        self.vars = ['close']

    def run(self, d):
        return d["close"].rolling(6).mean()

    
"""
MA12求简单移动平均,MA12(X),求X的12日移动平均值。
算法：(X1+X2+X3+...+X12)/12
"""
class MA12:
    def __init__(self):
        self.name = 'ma12'
        self.vars = ['close']

    def run(self, d):
        return d["close"].rolling(12).mean()

    
"""
MA24求简单移动平均,MA24(X),求X的24日移动平均值。
算法：(X1+X2+X3+...+X24)/24
"""
class MA24:
    def __init__(self):
        self.name = 'ma24'
        self.vars = ['close']

    def run(self, d):
        return d["close"].rolling(24).mean()
  
"""
AR指标是反映市场当前情况下多空双方力量发展对比的结果。它
是以当日的开盘价为基点,与当日最高价当、最低价相比较，通过
开盘价在股价中的地位反应市场买卖的人气。

AR = SUM(HIGH - OPEN, N) / SUM(OPEN - LOW, N) * 100
"""    
    
class F20230116:
    def __init__(self, N = 20):
        self.name = 'ar'
        self.vars = ['open','high','low']
        self.N = N
    def run(self, d):
        M = (d['open'] - d['low']).rolling(self.N).sum() * 100
        M[M == 0] = 1
        AR = (d['high'] - d['open']).rolling(self.N).sum() / M
        return AR

    
"""
ASI累计振动升降指标通过比较过去一段时间股价开高低收的关系来判读股价
的长期趋势。当ASI为正，说明趋势会继续;当ASI为负，说明趋势会终结。
A = ABS(HIGH - CLOSE[1])
B = ABS(LOW - CLOSE[1])
C = ABS(HIGH - LOW[1])
D = ABS(CLOSE[1] - OPEN[1])
E = CLOSE.diff(1)
F = CLOSE - OPEN
G = CLOSE[1] - OPEN[1]
X = E + 0.5 + G
K = MAX(A,B)
R = IF(A > B AND A>C, A+0.5B+0.25D, IF(B>A AND B>C, B + 0.5A + 0.25D, C + 0.25D))
SI = 16 * X / R * K
ASI = SUM(SI, N)/6
"""    
class F20230124:
    def __init__(self):
        self.name = 'asi'
        self.vars = ['open', 'high','low','close']

    def run(self, d):
        A = (d["high"] - d["close"].shift(1)).abs()
        B = (d["low"] - d["close"].shift(1)).abs()
        C = (d["high"] - d["low"].shift(1)).abs()
        D = (d["close"].shift(1) - d["open"].shift(1)).abs()
        E = d["close"].diff(1)
        F = d["close"] - d["open"]
        G = d["close"].shift(1) - d["open"].shift(1)
        X = E + 1/2*F + G
        R1 = A + 1/2 * B + 1/4 * D
        R2 = B + 1/2 * A + 1/4 * D
        R3 = C + 1/4 * D
        R = ((A > B) & (A > C)) * R1 + ((B > A) & (B > C)) * R2 + ((C > A) & (C > B)) * R3
        R[R == 0] = 1
        K = (A > B) * A + (A <= B) * B
        SI = 16 * X / R *  K
        ASI = SI.rolling(20).sum()
        return ASI/6

    
"""
CMF指标基于这样的假设，即强势市场 (处于上升趋势的市场) 通
常都伴随着位于日最高价与最低价之间上半部分的收盘价以及放大
的成交量。与此相反，弱势市场 (处于下跌趋势的市场)通常都伴
随着位于日最高价与最低价之间的下半部分的收盘价以及放大的
成交量。如果在成交量放大的同时，价格持续收于日最高价与最
低价之间的上半部分，那么该指标将会是正值，表示该证券处于
强势之中。相反，如果在成交量放大的同时，价格持续收于日最
高价与最低价之间的下半部分，那么该指标将是负值，表示该证
券处于弱势之中。

CLV = VOL*((CLOSE - LOW) - (HIGH - CLOSE))/(HIGH - CLOSE)
CMF = SUM(CLV,N)/SUM(VOL,N) * 100
"""
class F20230129:
    def __init__(self, N = 20):
        self.name = 'cmf'
        self.N = N
        self.vars = ['vol', 'close', 'low', 'high']
    def run(self, d):
        M1 = d['high'] - d['low']
        M1[M1 == 0] = 1
        CLV = d['vol'] *( (d['close'] - d['low']) - (d['high'] - d['close']))/M1
        M = d['vol'].rolling(self.N).sum()
        M[M == 0] = 1
        CMF = CLV.rolling(self.N).sum() / M * 100
        return CMF

    
    
    
"""
CCI指标测量当前价格对近期平均价格的偏离程度。商品通道指标
数值高则当前价格高于平均价格，反之亦然。作为超买超卖指标，
商品通道指标能预测价格趋势的背离。

TP = (HIGH + LOW + CLOSE)/3
CCI = (TP - SMA(TP , N))/(0.015 * MAD(TP , N))
"""    
class F20230204:
    def __init__(self, N = 20):
        self.name = 'cci'
        self.vars = ['high','low' ,'close']
        self.N = N
        
    def run(self, d):
        TP = (d['high'] + d['low'] + d['close'])/3
        M = MAD(TP, self.N)
        M[M == 0] = 1
        CCI = (TP - SMA(TP, self.N))/(0.015 * M)
        return CCI

"""
CVI指标计算最高价与最低价的价差均值来衡量股价的波动率，与ATR不同的
是CVI指标没有考虑周期间价格的跳空。在实际使用过程中，CVI指标结合用
均线等其他趋势指标去增加趋势判断的准确率。

CVI = (EMA(HIGH - LOW, N) - EMA(HIGH - LOW, N)[N])/EMA(HIGH - LOW, N) * 100
"""    
class F20230211:
    def __init__(self, N = 20):
        self.name = 'cvi'
        self.N = N
        self.vars = ['high', 'low']
    def run(self, d):
        A = EMA(d['high'] - d['low'], self.N)
        A[A == 0] = 1
        CVI = (A.diff(self.N)/A) * 100
        return CVI
    
    
"""
BR指标反映的是当前情况下多空双方力量斗争的结果。它以当前一日的收盘价
作为基础，与当日的最高价、最低价相比较，通过昨日收盘价在股价中的地位
反应市场买卖的人气。BR最好与AR结合使用，BR、AR均急跌，表明股价以到顶，
反跌在即，投资者应尽快出货，BR比AR低，且AR<50，表明股价已经到底，投资
者可吸纳低股，BR急速高升，而AR处在盘整或小跌时，表明股价正在上升;BR>AR，
又转为BR<AR时，也可买进;BR攀至高峰，又以50的跌幅出现时，投资者也可低
价进货，待股价升高再卖出。

BR = SUM(MAX(0,HIGH - CLOSE[1]),N)/SUM(MAX(0,CLOSE[1]-LOW),N)
"""   
class F20230218:
    def __init__(self, N = 20):
        self.name = 'br'
        self.N = N
        self.vars = ['high', 'low', 'close']
    def run(self, d):
        A = d['high'] - d['close'].shift(1) + (d['high'] - d['close'].shift(1)).abs()
        B = d['close'].shift(1) - d['low'] + (d['close'].shift(1) - d['low']).abs()
        M = B.rolling(self.N).sum()
        M[M == 0] = 1
        BR = A.rolling(self.N).sum() /  M
        return BR * 100
    
    
"""
MACD称为指数平滑异同平均线，是从双指数移动平均线发展而来的，由快的指数移
动平均线 (EMA)减去慢的指数移动平均线。当MACD从负数转向正数，是买的信号;当
MACD从正数转向负数，是卖的信号。

DIF = EMA(CLOSE,N1) - EMA(CLOSE,N2)
DEA = EMA(DIF,N3)
MACD = 2*(DIF - DEA)
"""
    
class F20230225:
    def __init__(self, N1 = 12, N2 = 26, N3 = 9):
        self.name = 'macd'
        self.vars = ['close', 'open','high','low']
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        
    def run(self, d):
        DIF = EMA(d["close"], self.N1) - EMA(d["close"], self.N2)
        DEA = EMA(DIF, self.N3)
        MACD = 2 * (DIF - DEA)
        return MACD * 100
    
"""
KVO指标的目的是为了观察短期和长期股票资金的流入和流出的情况。它的主要用途
是确认股票价格趋势的方向和强度。用它来判断股价趋势方向，如果股价是上升趋
势，它的摆动范围靠上 (大于0的方向);如果是下降趋势，它的摆动范围靠下 (小于
0的方向)

TR=IF(HIGH+LOW+CLOSE>HIGH[1]+LOW[1]+CLOSE[1],1,-1)
DM = HIGH-LOW
CM =IF(TR=TR[1],CM[1]+DM,DM[1]+DM)
VF=VOL*ABS(2*(DM/CM-1))*TR*100
KVO(N1,N2) = EMA(VF,N1)-EMA(VF,N2)
"""    
    
class F20230305:
    def __init__(self, N = 34, M1 = 55,M2 = 13):
        self.name = 'kvo'
        self.vars = ['close','vol','high','low']
        self.N = N
        self.M1 = M1
        self.M2 = M2
        
    def run(self, d):
        MID = (d['high'] + d['low'] + d['close'])/3
        SV = d['vol'] * (MID.diff(1)>0) - d['vol'] * (MID.diff(1) <= 0)
        KVO = EMA(SV, self.N) - EMA(SV, self.M1)
        KVO2 = EMA(KVO,self.M2)
        return KVO2/1000/2
    
    
"""
DDI指标，即方向标准离差指数，一般用于观察一段时间内股价相对于前一天向上
波动和向下波动的比例，并对其进行移动平均分析。通过分析DDI柱状线，可以判
断是买入信号还是卖出信号.

DMZ=IF(HIGH+LOW<=HIGH[1]+LOW[1],0,MAX(ABS(HIGH-HIGH[]),ABS(LOW-LOW[1])))
DMF=IF(HIGH+LOW>HIGH[1]+LOW[1],0,MAX(ABS(HIGH-HIGH[1]),ABS(LOW-LOW[1])))
DIZ=SUM(DMZ,N)/(SUM(DMZ,N)+SUM(DMF,N))*100
DIF= SUM(DMF,N)/(SUM(DMZ,N)+SUM(DMF,N))*100
DDI= DIZ-DIF
"""    
class F20230315:
    def __init__(self, N = 20):
        self.name = 'ddi'
        self.vars = ['open','high','low']
        self.N = N
        
    def run(self, d):
        A = d['high'].diff(1).abs()
        B = d['low'].diff(1).abs()
        M1 = (A + B +(A - B).abs())/2
        DMZ = (d['high'].diff(1) + d['low'].diff(1) > 0) * M1
        DMF = (d['high'].diff(1) + d['low'].diff(1) <= 0) * M1
        M = DMZ.rolling(self.N).sum() + DMF.rolling(self.N).sum()
        M[M == 0] = 1
        DIZ = DMZ.rolling(self.N).sum() / M * 100
        DIF = DMF.rolling(self.N).sum() / M * 100
        DDI = (DIZ - DIF)
        return DDI
        
"""
找出最高点出现的日期
"""
def high_len_cal(x):
    return (np.maximum.accumulate(x) == x.max()).sum()


"""
找出最低点出现的日期
"""
def low_len_cal(x):
    return (np.minimum.accumulate(x) == x.min()).sum()
    
"""
AROON指标计算自价格达到近期最高值和最低值以来所经过的期间数，帮助投资者预
测证券价格趋势、强弱以及趋势的反转等

AROON上升数=(计算期天数-最高价后的天数)/计算期天数*100
AROON下降数 =(计算期天数-最低价后的天数)/计算期天数*100
AROON= AROON上升数 - AROON下降数
"""    
class F20230321:
    def __init__(self, N = 20):
        self.name = 'aroon'
        self.vars = ['high','low']
        self.N = N
        
    def run(self, d):
        AROONUP = d['close'].rolling(self.N).apply(high_len_cal, raw=True, engine='cython') / self.N * 100
        AROONDOWN = d['close'].rolling(self.N).apply(low_len_cal, raw=True, engine='cython') / self.N * 100
        return AROONDOWN - AROONUP
    

"""
TEMA是对单一指数移动平均、双重指数移动平均和三重指数移动平均的综合当存在趋
势时，该指标的时滞与3个组成要素中的任何一个都要短。

EMAI = EMA(REAL,N)
EMA2 = EMA(EMA1,N
EMA3 = EMA(EMA2,N)
TEMA =3*EMA1-3*EMA2+EMA3
"""      
class F20230404:
    def __init__(self, N = 5, REAL = 'vol'):
        self.name = 'tema'
        self.REAL = REAL
        self.vars = [REAL]
        self.N = N
        
    def run(self, d):
        EMA1 = EMA(d[self.REAL], self.N)
        EMA2 = EMA(EMA1, self.N)
        EMA3 = EMA(EMA2, self.N)
        TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        return TEMA/100

    
    
"""
RI区域指标计算了期间内股票最高价到最低价的区域超过了期间之间收盘价到收盘价的
区域的时间，可用于辨别趋势的开始与结束。

TR=MAX(HIGH-LOW,ABS(CLOSE[1]-HIGH),ABS(CLOSE[1]-LOW))
W=IF(CLOSE>CLOSE[1],TR/(CLOSE-CLOSE[1]),TR)
SR(N1)=IF(MAX(W,N1)-MIN(W,N1)>0(W-MIN(W,N1))/(MAX(W,N1)-MIN(W,NI))*100, (W-MIN(W,N1))*100)
RI(N1,N2) = EMA(SR(N1),N2)
"""      
class F20230413:
    def __init__(self, N1 = 20, N2 = 5):
        self.name = 'ri'
        self.vars = ['high', 'low', 'close']
        self.N1 = N1
        self.N2 = N2
        
    def run(self, d):
        A = d['high'] - d['low']
        B = (d['high'] - d['close'].shift(1)).abs()
        C = (d['low'] - d['close'].shift(1)).abs()
        TR = ((A >= B) & (A >= C)) * A + ((B > A) & (B >= C)) * B + ((C > A) & (C > B)) * C
        M = d['close'].diff(1)
        M[M == 0] = 1
        W = (d['close'] > d['close'].shift(1)) * TR / M  + (d['close'] <= d['close'].shift(1)) * TR
        M2 = W.rolling(self.N1).apply(lambda x: x.max()- x.min())
        M2[M2 == 0] = 1
        SR = (W.rolling(self.N1).max() - W.rolling(self.N1).min() > 0 ) *  (W - W.rolling(self.N1).min())\
        / M2 * 100
        + (W.rolling(self.N1).max() - W.rolling(self.N1).min()  <  0 ) * (W - W.rolling(self.N1).min()) * 100
        RIN1N2 = EMA(SR, self.N2)
        return RIN1N2
    
"""

OBV指标的理论基础是市场价格的有效变动必须有成交量配合，量是价的先行指标。利用
OBV可以验证当前股价走势的可靠性，并可以得到趋势可能反转的信号。比起单独使用成
交量来，OBV看得更清楚。

OBV= OBV[1]+IF(CLOSE>CLOSE[1],VOL,IF(CLOSE<CLOSE[1,-VOL,0)
"""
class F20230427:
    def __init__(self, N = 9, M1 = 3, M2 = 3):
        self.name = 'obv'
        self.vars = ['close','vol']
        
    def run(self, d):
        OBV = (d['close'].diff(1) > 0) * d['vol'] - (d['close'].diff(1) < 0) * d['vol']
        OBV = OBV.cumsum()
        return OBV/1000000   
    
"""
MFI指标是RSI扩展指标，MFI指标用成交金额代替的指数，是某一时间周期内上涨的成交量
之和与下跌的成交量之和的比率。MFI>80为超买，当其回头向下跌破80时，为短线卖出时机
。MFI<20为超卖，当其回头向上突破20时，为短线买进时机。

TP =(HIGH+LOW+CLOSE)/3
MF = TP*VOL
PF(N) = SUM(IF(TP>TP[1],MF,0),N)
NF(N) = SUM(IF(TP<=TP[1],MF,0),N)
MR(N) = PF(N)/NF(N)
MFI= 100-(100/(1+MR))
"""      
class F20230501:
    def __init__(self, N = 20):
        self.name = 'mfi'
        self.vars = ['high', 'low', 'close','vol']
        self.N = N
        
    def run(self, d):
        TP = (d['high'] + d['low'] + d['close'])/3
        MF = TP * d['vol']
        PF = ((TP.diff(1) > 0) * MF).rolling(self.N).sum()
        NF = ((TP.diff(1) <= 0) * MF).rolling(self.N).sum()
        #大数乘大数计算过后的累加可能会产生微小的浮点误差，下面用于消除对应误差
        NF[NF < 1e-4] = 1
        PF[PF < 1e-4] = 1
        MR = PF / NF
        MFI = 100 - (100/(1 + MR))
        return MFI
 

"""
CMO指标通过计算今收与昨收的价差来判断趋势的强弱，当CMO大于50时，处于超买状态;当CMO
小于50时处于超卖状态。

CZ1=IF(CLOSE-CLOSE[1]>0,CLOSE-CLOSE[1],0)
CZ2=IF(CLOSE-CLOSE[1]<0,ABS(CLOSE-CLOSE[1]),0)
SU(N)= SUM(CZ1,N)
SD(N)= SUM(CZ2,N)
CMO=(SU(N)-SD(N))/(SU(N)+SD(N))*100
"""      
class F20230518:
    def __init__(self, N = 20):
        self.name = 'cmo'
        self.vars = ['close']
        self.N = N
        
    def run(self, d):
        CZ1 = d['close'].diff(1).clip(lower = 0)
        CZ2 = d['close'].diff(1).clip(upper=0).abs()
        SU = CZ1.rolling(self.N).sum()
        SD = CZ2.rolling(self.N).sum()
        M = SU + SD
        M[M == 0] = 1
        CMO = (SU - SD) /M * 100
        return CMO
    

"""
PVT指标与OBV指标类似，区别是计算PVT指标时，当收盘价大于前收盘价时只有部分成交
量会累加到之前的PVT指标上。

PVT=(CLOSE-CLOSE[1])/CLOSE[1]*VOL+PVT[1]
"""   
class F20230526:
    def __init__(self):
        self.name = 'pvt'
        self.vars = ['vol', 'close']
    def run(self, d):
        M = d['close'].shift(1)
        M[M == 0] = 1
        PV = d['close'].diff(1)/M * d['vol']
        PVT = PV.cumsum()
        return PVT/100  

"""
VHF指标用来判断当前行情处于趋势阶段还是振荡阶段，类似MACD等指标也能帮助识别行情趋势，
但当行情处于盘整阶段时会经常发出错误信号，而RSI在振荡行情时能准确识别超买超卖的状态，
但在趋势行情时总是发出错误信号。VHF指标通过识别趋势的强弱，来帮助投资者选择相应的指
标(如MACD或RSI)

HCP = MAX(HIGH, N)
LCP = MIN(LOW,N)
A = ABS(HCP-LCP)
B = SUM(ABS(CLOSE-CLOSE[1]),N)
"""   
    
class F20230612:
    def __init__(self, N = 20):
        self.name = 'vhf'
        self.vars = ['high', 'low']
        self.N = N
        
    def run(self, d):
        HCP = d['high'].rolling(self.N).max()
        LCP = d['low'].rolling(self.N).min()
        A = (HCP - LCP).abs()
        B = d['close'].diff(1).abs().rolling(self.N).sum()
        B[B == 0] = 1
        VHF = A/B
        return VHF

"""
RVI指标的计算方法与RSI类似，唯一的区别是RVI基于标准差，而RSI基于股价。
RVI用于判断股价波动的方向，常与趋势型指标配合使用 (如MA移动均线)。

UM(PRICE,N1)=IF(PRICE>PRICE[1],STD(PRICE,N1),0)
DM(PRICE,N1)=IF(PRICE<PRICE[1],STD(PRICE,N1),0)
UA(N2)=(UA[1]*(N-1)+UM)/N2
DA(N2)=(DA[1]*(N-1)+DM)/N2
RS(PRICE)= 100*UA/(UA+DA)
RVI=(RS(HIGH)+RS(LOW))/2
UA初始值 =SMA(UM,N)
DA初始值 =SMA(DM,N)
"""    
    
class F20230619:
    def __init__(self,N = 14):
        self.name = 'rvi'
        self.vars = ['close']
        self.N = N
    def run(self, d):
        POS = (d['close'].diff(1) > 0)
        NEG = d['close'].diff(1) < 0
        std = d['close'].rolling(self.N).std()
        pos_std = POS * std
        neg_std = NEG * std
        pos_avg = EMA(pos_std, self.N)
        neg_avg = EMA(neg_std, self.N)
        M = pos_avg + neg_avg
        M[M == 0] = 1
        RVI = 100 * pos_avg / M
        return RVI

"""
RSI指标多在30-70之间变动，通常80甚至90时被认为市场已到达超买状态，
至此市场价格自然会回落调整。当价格低跌至30以下即被认为是超卖状态，
市价将出现反弹回升。

UM=IF(CLOSE-CLOSE[1]>0,CLOSE-CLOSE[1],0)
DM =IF(CLOSE-CLOSE[1]<0,CLOSE[1]-CLOSE,0)
UA(N)=(UA[1]*(N-1)+UM)/N
DA(N)=(DA[1]*(N-I)+DM)/N
RSI= 100*(UA/(UA+DA))
"""

class F20230629:
    def __init__(self, N = 14):
        self.name = 'rsi'
        self.vars = ['close']
        self.N = N

    def run(self, d):
        NEG = d['close'].diff(1).clip(upper = 0)
        POS = d['close'].diff(1).clip(lower = 0)
        
        pos_ave = RMA(POS, self.N)
        neg_ave = RMA(NEG, self.N)
        M = pos_ave + neg_ave.abs()
        M[M == 0] = 1
        return 100 * pos_ave / M
    
"""
EMV指标是一个动量型指标，通过股价变动与成交量之间的关系去刻画股价的动量。当EMV大于0，
说明股价有上升的动能;当EMV小于0，说明股价有下降的压力。

MM =((HIGH-LOW)-(HIGH[1]-LOW[1]))/2
BR = VOL/(HIGH-LOW)
EMV = MM/BR
"""   
    
class F20230717:
    def __init__(self, N = 14):
        self.name = 'emv'
        self.vars = ['high', 'low', 'vol']
        self.N = N
        
    def run(self, d):
        M = d['vol']
        M[M == 0] = 1e10
        VOLUME = d['vol'].rolling(self.N).mean()/ M
        M2 = d['low'] + d['high']
        M2[M2 == 0] = 1
        MID = 100 *(d['low'] + d['high'] - d['low'].shift(1) - d['high'].shift(1)) / M2
        M3 = (d['high'] - d['low']).rolling(self.N).mean()
        M3[M3 == 0] = 1
        EMV = (MID * VOLUME * (d['high'] - d['low']) / M3).rolling(self.N).mean()
        return EMV
    
"""
反映价格变动的快慢程度;趋势明显的行情中，当ROC向下跌破零，卖出信号ROC向上突破零，买入信
号;震荡行情中，当ROC向下跌破ROCMA时，卖出信号，当ROC上穿ROCMA时，买入信号;股价创新高，
ROC未配合上升，显示上涨动力减弱;股价创新低，ROC未配合下降，显示下跌动力减弱，股价与ROC从
低位同时上升，短期反弹有望，股价与ROC从高位同时下降，警惕回落。

ROC =(CLOSE-CLOSE[N])/CLOSE[N]
ROCMA= SMA(ROC,M)
"""   
    
class F20230729:
    def __init__(self, N = 12, M = 6):
        self.name = 'rocma'
        self.vars = ['close']
        self.N = N
        self.M = M
        
    def run(self, d):
        M = d['close'].shift(self.N)
        M[M == 0] = 1
        ROC = d['close'].diff(self.N)/ M
        ROCMA = SMA(ROC, self.M)
        return ROCMA * 100
    

    
"""
IMI日内动量指标，通过计算过去一段周期内收盘价与开盘价的关系，来反应股票的买卖平衡。一般当
IMI低于30，我们认为股票处于超卖状态，当IMI高于70.我们认为股票处于超买状态。

USUM(N)= SUM(IF(CLOSE>OPEN,CLOSE-OPEN,0),N)
DSUM(N)= SUM(IF(CLOSE<=OPENOPEN-CLOSE,0),N)
IMI(N)= USUM(N)/(USUM(N)+DSUM(N))*100
"""   
    
class F20230805:
    def __init__(self, N = 20):
        self.name = 'imi'
        self.vars = ['close','open']
        self.N = N
        
    def run(self, d):
        USUM = (d['close'] - d['open']).clip(lower = 0).rolling(self.N).sum()
        DSUM = (d['close'] - d['open']).clip(upper = 0).rolling(self.N).sum().abs()
        M = USUM + DSUM
        USUM[M == 0] = 1
        M[M == 0] = 1
        IMI = USUM / M * 100
        return IMI

    
"""
VMACD指标与MACD指标的唯一区别是VMACD是基于成交量计算的，用来识别成交量的。

SHORT = EMA(VOL,N1)
LONG = EMA(VOL,N2)
DIFF = SHORT-LONG
DEA = EMA(DIFF,M)
VMACD = DIFF-DEA
"""    
    
class F20230817:
    def __init__(self, N1 = 12, N2 = 26, M = 9):
        self.name = 'vmacd'
        self.vars = ['vol']
        self.N1 = N1
        self.N2 = N2
        self.M = M
        
    def run(self, d):
        SHORT = EMA(d['vol'], self.N1)
        LONG = EMA(d['vol'], self.N2)
        DIFF = SHORT - LONG
        DEA = EMA(DIFF, self.M)
        VMACD = DIFF - DEA
        return VMACD
    

    
    
"""
用于表示价格的波动程度，价格波动幅度的突破通常也预示着价格的突
破，该指标价值越高，趋势改变的可能性就越高;该指标的价值越低，趋
势的移动性就越弱。

TR = MAX(HIGH-LOW,ABS(HIGH-CLOSE[1]),ABS(LOW-CLOSE[1]))
ATR = EMA(TR,N)
"""  
class F20230827:
    def __init__(self, N = 20):
        self.name = 'atr'
        self.N = N
        self.vars = ['high', 'low', 'close']
    def run(self, d):
        A = d['high'] - d['low']
        B = (d['high'] - d['close'].shift(1)).abs()
        C = (d['low'] - d['close'].shift(1)).abs()
        TR = ((A >= B) & (A >= C)) * A + ((B > A) & (B >= C)) * B + ((C > A) & (C > B)) * C
        ATR = EMA(TR, self.N)
        return ATR * 100
    
    
"""
SMI随机动量指标基于股价变化与股价波动区间的联系，刻画出当前股价与
近期股价波动的关系，对股价是否会反转或延续当前的趋势有一个基本的判读。

C(N)=(MAX(HIGH,N)+MAX(LOW,N))/2
H = CLOSE-C(N)
SHI = EMA(H,N1)
SH2 = EMA(SH1,N2)
R = MAX(HIGH,N)-MAX(LOW,N)
SRI = EMA(R,N1)
SR2 = EMA(SR1,N2)/2
SMI=(SH2/SR2)*100
"""  
class F20230904:
    def __init__(self, N = 10, N1 = 3, N2 = 3):
        self.name = 'smi'
        self.N = N
        self.N1 = N1
        self.N2 = N2
        self.vars = ['high', 'low', 'close']
    def run(self, d):
        CN = (d['high'].rolling(self.N).max() + d['low'].rolling(self.N).max() ) / 2
        H = d['close'] - CN
        SH1 = EMA(H, self.N1)
        SH2 = EMA(SH1, self.N2)
        R = d['high'].rolling(self.N).max() - d['low'].rolling(self.N).max()
        SR1 = EMA(R, self.N1)
        SR2 = EMA(SR1, self.N2)/2
        SR2[SR2 == 0] = 1
        SMI = (SH2 / SR2) * 100
        return SMI
    

"""
VR指标通过分析股价上升日成交额 (或成交量，下同)与股价下降
日成交额比值，从而掌握市场买卖气势的中期技术指标。主要用
于个股分析，其理论基础是，以成交量的变化确认低价和高价，从
而确定买“量价同步”及“量须先于价”卖时法。

A=IF(CLOSE>CLOSE[1],VOL,0)
B=IF(CLOSE<CLOSE[1],VOL,0)
VR =SUM(A,N)/SUM(B,N)*100
"""
class F20230912:
    def __init__(self, N = 20):
        self.name = 'vr'
        self.N = N
        self.vars = ['close', 'vol']
    def run(self, d):
        A = (d['close'] > d['close'].shift(1)) * d['vol']
        B = (d['close'] < d['close'].shift(1)) * d['vol']
        B[B == 0] = 1
        VR = A.rolling(self.N).sum() / B.rolling(self.N).sum() * 100
        return VR
    
    
    
"""
该指标用开盘价的向上波动幅度和向下波动幅度的距离差值来描述人
气高低ADTM指标在+1到-1之间波动，低于-0.5时为低风险区，高于+0.5
时为高风险区;ADTM上穿ADTMMA时，买入股票;ADTM跌穿ADTMMA时，卖出
股票。

DTM =IF(OPEN<OPEN[1],0,MAX((HIGH-OPEN),(OPEN-OPEN[1J)))
DBM =IF(OPEN>OPEN[1],0,MAX ((OPEN-LOW),OPEN-OPEN[1])))
STM = SUM(DTM,N)
SBM = SUM(DBMN)
ADTM =IF(STM>SBM,(STM-SBM)/STM,IF(STM=SBM,0,(STM-SBM)/SBM))
ADTMMA = SMA(ADTM,M)
"""
class F20230924:
    def __init__(self, N = 23, M = 8):
        self.name = 'adtmma'
        self.N = N
        self.M = M
        self.vars = ['open', 'high', 'low']
    def run(self, d):
        DTM = (d['open'].diff(1) >= 0)*MAX(d['high'] - d['open'], d['open'].diff(1))
        DBM = (d['open'].diff(1) < 0)*MAX(d['open'] - d['low'], d['open'].diff(1))
        STM = DTM.rolling(self.N).sum()
        SBM = DBM.rolling(self.N).sum()
        M = MAX(SBM, STM)
        M[M == 0] = 1
        ADTM = (STM - SBM)/ M
        ADTMMA = SMA(ADTM, self.M)
        return ADTMMA

    
    
"""
MassIndex主要用于寻找飙涨股或者极度弱势股的重要趋势反转点，股价高
低点之间的价差波带，忽而宽忽而窄，不断的重复循环变动，利用这种重复
循环的波带，可以预测股价的趋势反转点。

MASSINDEX= EMA(HIGH-LOW,9)/EMA((EMA(HIGH-LOW,9),9)
"""
class F20231010:
    def __init__(self):
        self.name = 'massindex'
        self.vars = [ 'high', 'low']
    def run(self, d):
        A = EMA(d['high'] - d['low'], 9)
        B = (EMA(A, 9))
        B[B == 0] = 1
        MASSINDEX = A / B
        return MASSINDEX.rolling(25).sum()
    
 


"""
估波指标又称为“估波曲线”，通过计算月度价格的变化速率的加权平均值来测
量市场的动量，属于长线指标。该指标适合在指数的月线图表中分析，当指标向
上穿越零线向预示牛市来临，是中期买入信号，但其不适宜寻找卖出时机，需结
合其它指标来进行分析。

R(N1)=((CLOSE-CLOSE[N1])/CLOSE[N1])*100
R(N2) =((CLOSE-CLOSE[N2])/CLOSE[N2]*100
RC(N1,N2) = R(N1)+R(N2)
COPPOCK(N1,N2N3)= WMA(RC(N1,N2),N3)
"""    
    
class F20231016:
    def __init__(self, N1 = 14, N2 = 11, N3 = 10):
        self.name = 'coppock'
        self.vars = ['close', 'open','high','low']
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        
        
    def run(self, d):
        M1 = d['close'].shift(self.N1)
        M2 = d['close'].shift(self.N2)
        M1[M1 == 0] = 1
        M2[M2 == 0] = 1
        RN1 = d['close'].diff(self.N1) / M1 * 100
        RN2 = d['close'].diff(self.N2) / M2 * 100
        RCN1N2 = RN1 + RN2
        COPPOCK = WMA(RCN1N2, self.N3)
        return COPPOCK
    
    
    
    
"""
VRSI是市场成交量的相对强弱指标，通过动态分析成交量的变化，识
破庄家的盘中对敲、虚假放量，虚假买卖盘等欺骗手段，从真实的量
能变化中找出庄家的战略意图，从而达到安全跟庄、稳定获利的投资
目标。

U=IF(CLOSE>CLOSE[1],VOL,IF(CLOSE=CLOSE[1],VOL/2,0))
D =IF(CLOSE<CLOSE[1],VOL,IF(CLOSE=CLOSE[1],VOL/2,0))
UU=((N-1)U[1]+U)/N
DD=((N-1)D[1]+D)/N
VRSI= 100*UU/(UU+DD)
"""    
class F20231024:
    def __init__(self, N = 20):
        self.name = 'vris'
        self.N = N
        self.vars = ['close', 'vol']
    def run(self, d):
        U = (d['close'] > d['close'].shift(1)) * d['vol'] + (d['close'] == d['close'].shift(1)) * d['vol'] / 2
        D = (d['close'] < d['close'].shift(1)) * d['vol'] + (d['close'] == d['close'].shift(1)) * d['vol'] / 2
        UU = ((self.N - 1) * U.shift(1) + U) / self.N
        DD = ((self.N - 1) * D.shift(1) + D) / self.N
        M = UU + DD
        M[M == 0] = 1
        VRSI = UU / M
        return VRSI

"""
VRSI是市场成交量的相对强弱指标，通过动态分析成交量的变化，识
破庄家的盘中对敲、虚假放量，虚假买卖盘等欺骗手段，从真实的量
能变化中找出庄家的战略意图，从而达到安全跟庄、稳定获利的投资
目标。

S1 = (VOL - VOL[1])大于0的部分
S2 = abs(VOL - VOL[1])

"""        
class F202310242:
    def __init__(self, N = 20):
        self.name = 'vrsi'
        self.N = N
        self.vars = ['close', 'vol']
    def run(self, d):
        S1 = d['vol'].diff(1).clip(lower = 0)
        S2 = d['vol'].diff(1).abs()
        A1 = SMA(S1,self.N)
        A2 = SMA(S2,self.N)
        A2[A2 == 0] = 1
        VRSI = A1 / A2
        return A2
    
"""
BIAS通过计算股价在波动过程中与移动平均线出现的偏离程度来判断股
价在剧烈波动时因偏离移动平均趋势可能形成的回档或反弹，正的乖离
率越大，表示短期获利越大，获利回吐的可能性越高;负的乖离率越大，
则空头回补的可能性越高。

BIAS = 100*(CLOSE-SMA(CLOSE,N))/SMA(CLOSE,N)
"""    
class F20231028:
    def __init__(self, N = 20):
        self.name = 'bias'
        self.vars = ['close']
        self.N = N
        
    def run(self, d):
        SMA2 = SMA(d['close'], self.N)
        SMA2[SMA2 == 0] = 1
        BIAS = (d['close'] - SMA2)/SMA2 * 100
        return BIAS
    

  
"""
用于股价的波动情况;一个相对较短时间内的波动率上升意味着市场底部的
到来，一段相对较长时间内波动率的下降意味着市场顶部的来到，可以根
据波动率预测股票未来的趋势。

REM = EMA((HIGH-LOW),N)
CV=100*(REM-REM[M])/REM[M]
"""
class F20231108:
    def __init__(self, N = 10, M = 10):
        self.name = 'cv'
        self.N = N
        self.M = M
        self.vars = ['high', 'low']
    def run(self, d):
        REM = EMA(d['high'] - d['low'] , self.N)
        M = REM.shift(self.M)
        M[M == 0] = 1
        CV = 100 * REM.diff(self.M)/ M
        return CV

    
    
"""
佳庆(cho)指标是对累积/派发线AD的改良;CHO曲线产生急促的凸
起时，代表行情可能出现向上或向下反转;股价>90天平均线，CHO
由负转正，买进参考，股价<90天平均线，CHO由正转负时，卖出参
考

MID=SUM(VOL*(2*CLOSE-HIGH-LOW)/(HIGH+LOW),0)
CHO= EMA(MID,N1)-EMA(MID,N2)
"""
class F20231119:
    def __init__(self, N1 = 10, N2 = 3):
        self.name = 'cho'
        self.N1 = N1
        self.N2 = N2
        self.vars = ['close', 'high', 'low', 'vol']
    def run(self, d):
        MID = d['vol'] * ( 2 * d['close'] - d['high'] - d['low'])/ (d['high'] + d['low'])
        MID2 = MID.cumsum()/100
        return EMA(MID2, self.N1) - EMA(MID2, self.N2)
    
"""
DBCD的原理和构造方法与乖离率类似，用法也与乖离率相同，优点
是能够保持指标的紧密同步，线条光滑，信号明确，能有效的过滤
掉伪信号。

BIAS =(CLOSE-SMA(CLOSE,N1))/SMA(CLOSE,NI)
DIF = BIAS-BIAS[N2]
DBCD = SMA(DIF,N3,1)
"""  
class F20231126:
    def __init__(self, N1 = 5, N2 = 16, N3 = 17):
        self.name = 'dbcd'
        self.vars = ['bias', 'low', 'close']
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        
    def run(self, d):
        M = SMA(d['close'], self.N1)
        M[M == 0] = 1
        BIAS = (d['close'] - SMA(d['close'], self.N1))/ M
        DIF = BIAS.diff(self.N2)
        DBCD = SMA(DIF, self.N3)
        return DBCD
    

    
    
"""
PSY将一定时期内投资者趋向买方或卖方的心理事实转化为数值，形成
人气指数，从而判断股价的未来趋势;通常，指标小于25时关注做多机
会，大于75时关注做空机会，小于10为极度超卖，大于90为极度超买，
市场宽幅振荡时，PSY也会25~75区间反复突破，给出无效信号。

PSY= 100*COUNT(CLOSE>CLOSE[1]),N)/N
"""  
class F20231207:
    def __init__(self, N = 12, M = 6):
        self.name = 'psy'
        self.vars = ['close']
        self.N = N
        self.M = M
        
    def run(self, d):
        PSY = 100 * (d['close'].diff(1) > 0).rolling(self.N).sum() /self.N
        return PSY.rolling(self.M).mean()
    
    

"""
BBI是一种将不同日数移动平均线加权平均之后的综合指标;计算BBI时，近期数
据利用较多，远期数据利用次数较少，是一种变相的加权计算，既有短期移动
平均线的灵敏，又有明显的中期趋势特征。

BBI=(MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4
"""   
    
class F20231213:
    def __init__(self):
        self.name = 'bbi'
        self.vars = ['ma3', 'ma6','ma12','ma24']

    def run(self, d):
        BBI = (d["ma3"] + d["ma6"] + d["ma12"] +d["ma24"])/4
        return BBI
    
"""
AD指标将每日的成交量通过价格加权累计，用以计算成交量的动量；
向上的AD表明买方占优势，而向下的AD表明卖方占优，AD与价格的
背离可视为买卖信号，即底背离考虑买入，顶背离考虑卖出。

AD=SUM(VOL(2*CLOSE-HIGH-LOW)/(HIGH+LOW),0)
"""    
class F20231221:
    def __init__(self):
        self.name = 'ad'
        self.vars = ['vol','close','high','low']
    def run(self, d):
        A = (2 * d['close'] - d['high'] - d['low'] )
        M = d['high'] - d['low']
        M[M == 0] = 1
        A = A/ M 
        AD = A * d['vol']
        AD = AD.cumsum()
        return AD/1000/3
    
"""
KDJ主要是研究最高价、最低价和收盘价之间的关系，同时综合了动量
观念、强弱指标及移动平均线的优点，所以能够比较直观地研判行情，
被广泛用于股市的中短期趋势分析中;K与D值永远介于0到100之间，D大
于70时，行情呈现超买现象，D小于30时，行情呈现超卖现象。

RSV =100*(CLOSE-MIN(LOW,N))/(MAX(HIGH,N)-MIN(LOW,N))
K = SMA(RSV,M1,1)
"""    
      
class F20231230K:
    def __init__(self, N = 9, M1 = 3, M2 = 3):
        self.name = 'K'
        self.vars = ['high', 'low', 'close']
        self.N = N
        self.M1 = M1
        self.M2 = M2
        
    def run(self, d):
        M = d['high'].rolling(self.N).max() - d['low'].rolling(self.N).min()
        M[M == 0] = 1
        RSV = 100 * (d['close'] - d['low'].rolling(self.N).min())/ M
        K = SMA(RSV, self.M1)
        return K
    
    
    
"""
KDJ主要是研究最高价、最低价和收盘价之间的关系，同时综合了动量
观念、强弱指标及移动平均线的优点，所以能够比较直观地研判行情，
被广泛用于股市的中短期趋势分析中;K与D值永远介于0到100之间，D大
于70时，行情呈现超买现象，D小于30时，行情呈现超卖现象。

RSV =100*(CLOSE-MIN(LOW,N))/(MAX(HIGH,N)-MIN(LOW,N))
K = SMA(RSV,M1,1)
D = SMA(K,M2,1)
"""    
class F20231230D:
    def __init__(self, N = 9, M1 = 3, M2 = 3):
        self.name = 'D'
        self.vars = ['high', 'low', 'close']
        self.N = N
        self.M1 = M1
        self.M2 = M2
        
    def run(self, d):
        M = d['high'].rolling(self.N).max() - d['low'].rolling(self.N).min()
        M[M == 0] = 1
        RSV = 100 * (d['close'] - d['low'].rolling(self.N).min())/ M
        K = SMA(RSV, self.M1)
        D = SMA(K, self.M2)
        return D
    


    
"""
KDJ主要是研究最高价、最低价和收盘价之间的关系，同时综合了动量
观念、强弱指标及移动平均线的优点，所以能够比较直观地研判行情，
被广泛用于股市的中短期趋势分析中;K与D值永远介于0到100之间，D大
于70时，行情呈现超买现象，D小于30时，行情呈现超卖现象。

RSV =100*(CLOSE-MIN(LOW,N))/(MAX(HIGH,N)-MIN(LOW,N))
K = SMA(RSV,M1,1)
D = SMA(K,M2,1)
J=3*K-2*D
"""   
class F20231230J:
    def __init__(self, N = 9, M1 = 3, M2 = 3):
        self.name = 'J'
        self.vars = ['high', 'low', 'close']
        self.N = N
        self.M1 = M1
        self.M2 = M2
        
    def run(self, d):
        M = d['high'].rolling(self.N).max() - d['low'].rolling(self.N).min()
        M[M == 0] = 1
        RSV = 100 * (d['close'] - d['low'].rolling(self.N).min())/ M
        RSV = RSV
        K = SMA(RSV, self.M1)
        D = SMA(K, self.M2)
        J = 3 * K - 2 * D
        print (J)
        return J
        
        
        
'''
Monthly_Idiosyncratic_Volatilities: 这个指标需要市场的总体回报数据
Assets_to_Market_Equity: 这个指标需要公司的资产总额数据
Bookto_MarketEquity: 这个指标需要公司的账面价值数据
Debt_to_Market_Equity: 这个指标需要公司的总负债数据
Dividend_Yield: 这个指标需要公司的分红数据

估值因子板块需要公司的财务数据，如利润、销售、现金流、企业价值等

'''


'''		
class DownsideBetaCalculator:
    def __init__(self, window=30):
        self.window = window

    def run(self, asset_returns, market_returns):

        negative_market_returns = market_returns < 0
        asset_returns = asset_returns[negative_market_returns]
        market_returns = market_returns[negative_market_returns]


        covariance = asset_returns.rolling(window=self.window).cov(market_returns)
        variance = market_returns.rolling(window=self.window).var()
        downside_beta = covariance / variance

        return downside_beta
'''

'''
class IdiosyncraticRsquaredCalculator:
    def __init__(self, window=30):
        self.window = window

    def run(self, asset_returns, market_returns):
        idiosyncratic_rsquared = asset_returns.rolling(window=self.window).apply(
            self.calculate_idiosyncratic_rsquared, args=(market_returns,), raw=True)

        return idiosyncratic_rsquared


    def calculate_idiosyncratic_rsquared(asset_returns, market_returns):

        model = LinearRegression()
        model.fit(market_returns.values.reshape(-1, 1), asset_returns.values)

 
        residuals = asset_returns.values - model.predict(market_returns.values.reshape(-1, 1))


        idiosyncratic_rsquared = np.sum(residuals ** 2) / np.sum((asset_returns.values - np.mean(asset_returns.values)) ** 2)

        return idiosyncratic_rsquared

class IdiosyncraticSkewnessCalculator:
    def __init__(self, window=30):
        self.window = window

    def run(self, asset_returns, market_returns):
        idiosyncratic_skewness = asset_returns.rolling(window=self.window).apply(
            self.calculate_idiosyncratic_skewness, args=(market_returns,), raw=True)

        return idiosyncratic_skewness

    def calculate_idiosyncratic_skewness(asset_returns, market_returns):
        
        model = LinearRegression()
        model.fit(market_returns.values.reshape(-1, 1), asset_returns.values)

        
        residuals = asset_returns.values - model.predict(market_returns.values.reshape(-1, 1))

        
        idiosyncratic_skewness = skew(residuals)

        return idiosyncratic_skewness

class IdiosyncraticVolatilityCalculator:
    def __init__(self, window=30):
        self.window = window

    def run(self, asset_returns, market_returns):
        idiosyncratic_volatility = asset_returns.rolling(window=self.window).apply(
            self.calculate_idiosyncratic_volatility, args=(market_returns,), raw=True)

        return idiosyncratic_volatility

  
    def calculate_idiosyncratic_volatility(asset_returns, market_returns):
        
        model = LinearRegression()
        model.fit(market_returns.values.reshape(-1, 1), asset_returns.values)

        residuals = asset_returns.values - model.predict(market_returns.values.reshape(-1, 1))

        
        idiosyncratic_volatility = np.std(residuals)

        return idiosyncratic_volatility

class MarketBetaCalculator:
    def __init__(self, window=30):
        self.window = window

    def run(self, asset_returns, market_returns):
        market_beta = asset_returns.rolling(window=self.window).apply(
            self.calculate_market_beta, args=(market_returns,), raw=True)

        return market_beta

  
    def calculate_market_beta(asset_returns, market_returns):
       
        model = LinearRegression()
        model.fit(market_returns.values.reshape(-1, 1), asset_returns.values)

        
        market_beta = model.coef_[0]

        return market_beta
'''	




'''
class FFrazziniPedersenBeta:
    def __init__(self):
        self.name = 'FrazziniPedersenBeta'
        self.vars = ['Date', 'Code', 'Close', 'PreClose', 'open', 'high', 'low', 'vol', 'avgprice', 'MarketReturn']

    def run(self, df):
        
        df['Return'] = df.groupby('Code')['Close'].pct_change()


        beta = []
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code]
            model = LinearRegression().fit(stock_data[['MarketReturn']], stock_data['Return'])
            beta.append(model.coef_[0])

 
        beta = pd.Series(beta, index=df['Code'].unique())
        beta = beta.rank() / len(beta)

        return beta
'''  

        
        
'''        
#估值因子
#这个因子需要股票的总发行量
class FAssetsToMarketEquity:
    def __init__(self):
        self.name = 'assets_to_market_equity'

    def run(self, d):
        # Assuming 'assets' and 'shares_outstanding' are columns in your dataframe 'd'
        d['market_equity'] = d['Close'] * d['shares_outstanding']
        d['assets_to_market_equity'] = d['assets'] / d['market_equity']
        return d
        
class FBooktoMarketEquity:
    def __init__(self):
        self.name = 'book_to_market_equity'

    def run(self, d):
        # Assuming that 'book_value' and 'market_value' are columns in your dataframe 'd'
        d['book_to_market_equity'] = d['book_value'] / d['market_value']
        return d

#这个因子需要debt和market equity
class FDebtToMarketEquity:
    def __init__(self):
        self.name = 'debt_to_market_equity'

    def run(self, d):
        
        d['debt_to_market_equity'] = d['debt'] / d['market_equity']
        return d
        
class FDivergenceRate:
    def __init__(self):
        self.name = 'divergence_rate'

    def run(self, d):
        d['log_return'] = np.log(d['Close'] / d['Close'].shift(1))
        return d


#这个因子需要股息支付
class FDividendYield:
    def __init__(self):
        self.name = 'dividend_yield'

    def run(self, d):
        # Assuming that 'dividends' is a column in your dataframe 'd' which represents the dividends paid by the company
        d['dividend_yield'] = d['dividends'] / d['Close']
        return d

#这个因子需要税前收入
class FEBITtoEnterpriseValue:
    def __init__(self):
        self.name = 'ebit_to_ev'

    def run(self, d):
        # Assuming that 'ebit' and 'ev' are columns in your dataframe 'd' which represent the company's EBIT and enterprise value
        d['ebit_to_ev'] = d['ebit'] / d['ev']
        return d


#这个因子需要每股收益      
class FEPChange:
    def __init__(self):
        self.name = 'ep_change'

    def run(self, d):
        # Assuming that 'eps' is a column in your dataframe 'd' which represents the earnings per share
        d['ep_change'] = d['eps'].pct_change()
        return d


#这个因子需要每股收益       
class FEarningstoPriceRatio:
    def __init__(self):
        self.name = 'ep_ratio'

    def run(self, d):
        d['ep_ratio'] = d['eps'] / d['Close']
        return d
        
#这个因子需要企业价值或EBITDA        
class FEnterpriseMultiple:
    def __init__(self):
        self.name = 'enterprise_multiple'

    def run(self, d):
        
        d['enterprise_multiple'] = d['ev'] / d['ebitda']
        return d
        
        
        
#这个因子需要公司债务和现金      
class FEnterpriseValue:
    def __init__(self):
        self.name = 'enterprise_value'

    def run(self, d):
        # Assuming that 'debt' and 'cash' are columns in your dataframe 'd' which represent the debt and cash of the company respectively
        d['enterprise_value'] = d['Close'] * d['vol'] + d['debt'] - d['cash']
        return d



#这个因子需要现金流量
class FOperatingCashFlowToPriceRatio:
    def __init__(self):
        self.name = 'operating_cash_flow_to_price_ratio'

    def run(self, d):
        # Assuming that 'operating_cash_flow' is a column in your dataframe 'd' which represents the operating cash flow of the company
        d['operating_cash_flow_to_price_ratio'] = d['operating_cash_flow'] / (d['Close'] * d['vol'])
        return d
 
#这个指标需要所有者收益
 
class FOwnerEarningToPrice:
    def __init__(self):
        self.name = 'owner_earning_to_price'

    def run(self, d):
        # Assuming that 'owner_earning' is a column in your dataframe 'd' which represents the owner earning of the company
        d['owner_earning_to_price'] = d['owner_earning'] / (d['Close'] * d['vol'])
        return d

#这个指标需要每股收益
class FPEToGrowthRatio:
    def __init__(self):
        self.name = 'pe_to_growth_ratio'

    def run(self, d):
        # Assuming that 'eps_growth' is a column in your dataframe 'd' which represents the EPS growth rate of the company
        d['pe_to_growth_ratio'] = (d['Close'] / d['eps']) / d['eps_growth']
        return d
  
#这个指标需要销售额和企业价值  
class FSaleToEnterpriseValue:
    def __init__(self):
        self.name = 'sale_to_enterprise_value'

    def run(self, d):
        d['sale_to_enterprise_value'] = d['sales'] / d['enterprise_value']
        return d
        
#这个指标需要销售额
class FSalesToPriceRatio:
    def __init__(self):
        self.name = 'sales_to_price_ratio'

    def run(self, d):
        # Assuming that 'sales' is a column in your dataframe 'd' which represents the sales of the company
        d['sales_to_price_ratio'] = d['sales'] / d['Close']
        return d
 
#这个指标需要 tangible_capital
class FTangibleCapitalChange:
    def __init__(self):
        self.name = 'tangible_capital_change'

    def run(self, d):
        # Assuming that 'tangible_capital' is a column in your dataframe 'd' which represents the tangible capital of the company
        d['tangible_capital_change'] = d['tangible_capital'].diff()
        return d

#这个指标需要分红，盈利增长率。
class FPEGrowthRatioToDividendYield:
    def __init__(self):
        self.name = 'pe_growth_ratio_to_dividend_yield'

    def run(self, d):
        # Assuming 'pe_ratio', 'earnings_growth_rate', 'dividends' are columns in your dataframe 'd'
        d['pe_growth_ratio'] = d['pe_ratio'] / d['earnings_growth_rate']
        d['dividend_yield'] = d['dividends'] / d['Close']
        d['pe_growth_ratio_to_dividend_yield'] = d['pe_growth_ratio'] / d['dividend_yield']
        return d
'''        
'''
class CoSkewnessCalculator:
    def __init__(self, window=30):
        self.window = window
        self.name = 'CoSkewness'
        self.vars = ['Date', 'Code', 'Close', 'PreClose', 'open', 'high', 'low', 'vol', 'avgprice', 'MarketReturn']

    def run(self, df):
        df['Return'] = df['Close'] / df['PreClose'] - 1
        df['ExcessReturn'] = df['Return'] - df['MarketReturn']
        co_skewness = df.groupby('Code')['ExcessReturn'].rolling(window=self.window).apply(lambda x: skew(x), raw=True)
        return co_skewness.reset_index(name=self.name)
'''





class MaximumDailyReturnCalculator:
    def __init__(self, window=5):  
        self.name = 'MaximumDailyReturn'
        self.vars = ['close', 'PreClose']
        self.window = window  

    def run(self, d):
        d['Return'] = d['close'].pct_change()
        d['MaximumDailyReturn'] = d['Return'].rolling(self.window).max()  
        return d['MaximumDailyReturn']

class MinimumDailyReturnCalculator:
    def __init__(self, window=5):  
        self.name = 'MinimumDailyReturn'
        self.vars = ['close', 'PreClose']
        self.window = window  

    def run(self, d):
        d['Return'] = d['close'].pct_change()
        d['MinimumDailyReturn'] = d['Return'].rolling(self.window).min()  
        return d['MinimumDailyReturn']





'''
缺少市场指数收益率数据
class FMonthlyIdiosyncraticVolatility:
    def __init__(self):
        self.name = 'MonthlyIdiosyncraticVolatility'
        self.vars = ['Date', 'Code', 'Close', 'PreClose', 'open', 'high', 'low', 'vol', 'avgprice']

    def run(self, d):
        d['Return'] = d['Close'].pct_change().dropna()
        d['Date'] = pd.to_datetime(d['Date'])
        
        return d['Return'].resample('M').std()
'''
#算的累计偏离度，不然结构错误
class FNegativeCoefficientOfSkewness:
    def __init__(self):
        self.name = 'NegativeCoefficientOfSkewness'
        self.vars = [ 'close', 'PreClose']

    def run(self, d):
        d['Return'] = d['close'].pct_change()
        d['NegativeCoefficientOfSkewness'] = d['Return'].expanding().apply(lambda x: -stats.skew(x.dropna()))
        return d['NegativeCoefficientOfSkewness']

class FPriceDelay:
    def __init__(self, lag=1):
        self.lag = lag
        self.name = 'PriceDelay'
        self.vars = [ 'close']

    def run(self, d):
        # 对收盘价进行滞后操作
        d['DelayedClose'] = d['close'].shift(self.lag)
        
        
        d['PriceDelay'] = d['close'] - d['DelayedClose']
        print(d['PriceDelay'])
         
        return d['PriceDelay']




class FTotalVolatility:
    def __init__(self):
        self.name = 'TotalVolatility'
        self.vars = ['Date', 'Code', 'Close', 'PreClose', 'open', 'high', 'low', 'vol', 'avgprice']

    def run(self, d):
        d['Return'] = d['Close'].pct_change()
        
        return d['Return'].rolling(window=30).std()
'''
需要mkt回报，beta系数等数据
class FSystematicSkewness:
    def __init__(self):
        self.name = "SystematicSkewness"
        self.vars = ['Date', 'Code', 'Close', 'PreClose', 'open', 'high', 'low', 'vol', 'avgprice']
        
    def run(self, d):
        d['Return'] = d['Close'].pct_change()
        d['MktCap'] = d['MktCap'].shift(1)
        d['MktReturn'] = d['Index'].pct_change()
        d['ExcessReturn'] = d['Return'] - d['RiskFreeRate']
        d['ExcessMktReturn'] = d['MktReturn'] - d['RiskFreeRate']
        d['SystematicReturn'] = d['Beta'] * d['ExcessMktReturn']
        d['IdiosyncraticReturn'] = d['ExcessReturn'] - d['SystematicReturn']
        return stats.skew(d['IdiosyncraticReturn'])
'''
#先定义
def calculate_tail_risk(returns):
    VaR_95 = returns.quantile(0.05)
    return VaR_95

def calculate_total_skewness(returns):
    skewness = stats.skew(returns)
    return skewness

class FTailRisk:
    def __init__(self):
        self.name = 'TailRisk'
        self.vars = ['close', 'PreClose']

    def run(self, d):
        d['Return'] = d['close'].pct_change()
        d['TailRisk'] = d['Return'].expanding().apply(lambda x: calculate_tail_risk(x.dropna()))
        return d['TailRisk']



class FTotalSkewness:
    def __init__(self):
        self.name = 'TotalSkewness'
        self.vars = ['close', 'PreClose']

    def run(self, d):
        d['Return'] = d['close'].pct_change()
        d['TotalSkewness'] = d['Return'].expanding().apply(lambda x: calculate_total_skewness(x.dropna()))
        return d['TotalSkewness']

class DownUpVolCalculator:
    def __init__(self, window=5):
        self.window = window
        self.name = 'DownUpVolatility'
        self.vars = ['Close', 'PreClose', 'open', 'high', 'low']

    def run(self, d):
        d['range_pct'] = (d['high'] - d['low']) / d['PreClose'] * 100
        d['Down_to_UpVolatility'] = d['range_pct'].rolling(window=5).std()
        
        return d['Down_to_UpVolatility'].replace([np.inf, -np.inf], np.nan)






        
        

#所有类别的日期信息
class_name = [ 
               MaximumDailyReturnCalculator,
               MinimumDailyReturnCalculator,
               FNegativeCoefficientOfSkewness,
               FPriceDelay,
               FTotalVolatility,
               FTailRisk,
               FTotalSkewness,
               DownUpVolCalculator
               
               
               
               
               
    
    
    
    
    
    ]    
    
if __name__ == '__main__': 
    df_open = pd.read_csv("open.csv").set_index("Unnamed: 0").T
    df_high = pd.read_csv("high.csv").set_index("Unnamed: 0").T
    df_low = pd.read_csv("low.csv").set_index("Unnamed: 0").T
    df_close = pd.read_csv("close.csv").set_index("Unnamed: 0").T
    df_vol = pd.read_csv("volume.csv").set_index("Unnamed: 0").T
    print("finish Reading data")
    # """
    # SecCode     000001  000002  000003  000004  ...  301373  301358  301419  688515
    # 2010-01-04   23.71   10.60     NaN    10.0  ...     NaN     NaN     NaN     NaN
    # 2010-01-05   23.30   10.36     NaN    10.0  ...     NaN     NaN     NaN     NaN
    # 2010-01-06   22.90   10.36     NaN    10.0  ...     NaN     NaN     NaN     NaN
    # 2010-01-07   22.65   10.28     NaN    10.0  ...     NaN     NaN     NaN     NaN
    # 2010-01-08   22.60   10.35     NaN    10.0  ...     NaN     NaN     NaN     NaN
    # """
    d = {'open': df_open, 'high': df_high, 'low': df_low, 'close': df_close, 'vol': df_vol}

    for i in class_name:
        #print(i)
        try:
            self = i()
            d[self.name] = self.run(d)
            print(f"{self.name} finish")
        except:
            print(i)