import pandas as pd 
import numpy as np

class FeatureExtractor():
    def __init__(self, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name 
        self.out_list = ['rsi','bandwidth','percent-b', 'cv','x','chop','high_','low_','open_','ewm-tp_','std_'] 
        self.period = 20
        self.chop_period = 20
        self.rsi_period = 14
        self.period_list = list(range(1,20))
        self.volume_period_list =list(range(1,5))
    def standardize(self,row):
        return (row[0] - row[1]) / (row[2] - row[1])

    def extract(self):

        df = pd.read_csv(self.input_file_name)
        
        #typical price (tp) calculation
        df['tp'] = df[['close','high','low']].apply(lambda x: (x[0]+x[1]+x[2])/3,axis=1)
        df['high_']=df[['close','high']].apply(lambda x: (x[1])/x[0],axis=1)
        df['low_']=df[['close','low']].apply(lambda x: (x[1])/x[0],axis=1)
        df['open_']=df[['close','open']].apply(lambda x: (x[1])/x[0],axis=1)
        #rate of change (roc) calculation for several periods
        for i, per in enumerate(self.period_list):
            df['roc-'+str(i)] = df['tp'].pct_change(per)            
            df['roc-'+str(i)] = df['roc-'+str(i)].interpolate(method='linear').bfill()
        for i, per in enumerate(self.volume_period_list):	
            df['v_roc-'+str(i)] = df['volume'].pct_change(per)            
            df['v_roc-'+str(i)] = df['roc-'+str(i)].interpolate(method='linear').bfill()
        #Bollinger bands bandwidth and percent-b calculations
        df['std'] = df['tp'].rolling(self.period).std()
        df['min'] = df['tp'].rolling(self.period).min()
        df['max'] = df['tp'].rolling(self.period).max()
		
        df['std'] = df['std'].interpolate(method='linear').bfill()
        df['min'] = df['min'].interpolate(method='linear').bfill()
        df['max'] = df['max'].interpolate(method='linear').bfill()
        df['std_']=df['std']/df['close']
        df['ewm-tp'] = df['tp'].ewm(span=self.period,min_periods=0,adjust=False,ignore_na=False).mean()
        df['ewm-tp_']=df['ewm-tp']/df['close']
        df['lower'] = df[['ewm-tp','std']].apply(lambda x: x[0] - 2 * x[1],axis=1)
        df['upper'] = df[['ewm-tp','std']].apply(lambda x: x[0] + 2 * x[1],axis=1)
        
        df['bandwidth'] = df[['ewm-tp','lower','upper']].apply(lambda x: (x[2]-x[1])/x[0],axis=1)
        df['percent-b'] = df[['tp','lower','upper']].apply(lambda x: self.standardize(x),axis=1)
        
        # coefficient of variation (cv) calculation
        df['cv'] = df[['ewm-tp','std']].apply(lambda x: x[1]/x[0],axis=1)
        
        #average true range calculation
        df['close-p'] =  df['close'].shift(1)
        df['close-p'] = df['close-p'].interpolate(method='linear').bfill()
        
        df['tr'] = df[['close-p','high','low']].apply(lambda x: max([(x[1]-x[2]),abs(x[1]-x[0]),abs(x[2]-x[0])]), 

axis=1)
        df['atr'] = df['tr'].ewm(span=1,min_periods=0,adjust=False,ignore_na=False).mean()
        df['atr-sum'] = df['atr'].rolling(self.chop_period).sum()
        df['atr-sum'] = df['atr-sum'].interpolate(method='linear').bfill()
         
        df['max-high'] = df['high'].rolling(self.chop_period).max()
        df['min-low'] = df['low'].rolling(self.chop_period).min()

        df['max-high'] = df['max-high'].interpolate(method='linear').bfill()
        df['min-low'] = df['min-low'].interpolate(method='linear').bfill()
        
        #choppiness index calculation
        df['chop'] = df[['atr-sum','max-high','min-low']].apply(lambda x: np.log10(x[0]/(x[1]-x[2]))/np.log10

(self.chop_period), axis=1)
        df['chop'] = df['chop'].interpolate(method='linear').bfill()
 
        #x indicator calculation
        df['x'] = df[['close','open']].apply(lambda x: (2*x[0] - x[1])/x[0],axis=1)
 
        #rsi calculation
        df['diff-c'] = df[['close']].diff()
        df['diff-c'] = df['diff-c'].interpolate(method='linear').bfill()
        df['up'] = df[['diff-c']].apply(lambda x: x[0] if x[0] > 0 else 0, axis = 1)
        df['down'] = df[['diff-c']].apply(lambda x: -x[0] if x[0] < 0 else 0, axis = 1)
        df['roll-up'] = df['up'].ewm(span=self.rsi_period,min_periods=0,adjust=False,ignore_na=False).mean()
        df['roll-down'] = df['down'].ewm(span=self.rsi_period,min_periods=0,adjust=False,ignore_na=False).mean()
        eps = 1e-10
        df['rsi'] = df[['roll-up','roll-down']].apply(lambda x: 1.0 - 1.0/(1.0 + x[0]/(x[1]+eps)),axis=1)

        df[self.get_column_names()].to_csv(self.output_file_name)
        

    def get_feature_names(self):
        out = self.out_list.copy()
        for i,j in enumerate(self.period_list):
            out.append('roc-'+str(i))
        for i,j in enumerate(self.volume_period_list):
            out.append('v_roc-'+str(i))
        return out

    def get_column_names(self):
        b = self.get_feature_names()
        a = ['close']
        return a + b
    
def main():
    
    input_file = 'ltcusdt-1hour.csv'
    output_file = 'ltcusdt-1hour-out.csv'
    fe = FeatureExtractor(input_file, output_file)
    fe.extract()

if __name__ == '__main__':
    main()