import gc
from abc import ABC, abstractmethod
import time

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kaggle.competitions import twosigmanews
pd.set_option('max_columns', 50)

# Methodical sources:
# https://www.kaggle.com/jsaguiar/baseline-with-news
# https://wkirgsn.github.io/2018/02/10/auto-downsizing-dtypes/

class Company:
    """There's no business like mining business.
    Jewelry and gems for the middle class.
    Currently under investigation over legal affairs."""
    
    def __init__(self):
        self.miner = DataMiner(MineralCleaner())
        self.director = Directrice()
        self.auditor = Auditor()
    
    def grow(self):
        """Any given company must pursue the only one thing 
        in order to survive."""

        self.miner.dig_in_the_mine(*self.auditor.open_the_mine())
        self.director.interrogate(self.miner)
        
        for yield_report in self.auditor.report_yield():
          ten_day_forecast_df = self.director.forecast(*yield_report)
          self.auditor.log_forecast(ten_day_forecast_df)
        
        self.auditor.file_away()


class CompanyEmployee(ABC):
    """Once you're in, you'll never get out.
    Everyone has a weak point."""
    def __init__(self):
        self.tra_df = None
    
    @staticmethod
    def clocking_work(work):
        """Work-life balance is a foreign word."""
        def do_work(*args, **kwargs):
            """Don't ask what the company can do for you 
            but instead what you can do for the company"""
            start_time = time.time()  # clock in
            ret = work(*args, **kwargs)
            end_time = time.time()  # clock out
            print('Clocking {:.3} seconds'.format(end_time-start_time))
            return ret
        return do_work


class Auditor(CompanyEmployee):
    """Checks the company environment for its integrity.
    Maintains a long-time friendship with the directrice.
    Plays golf with the CEO every wednesday."""
    
    def __init__(self):
        self.env = twosigmanews.make_env()
    
    def open_the_mine(self):
        """The mine is to be opened by authorized staff only.
        Recent incidents required tightening admission."""
        return self.env.get_training_data()
    
    def report_yield(self):
        """Confidential."""
        print('Start reporting yields..')
        return self.env.get_prediction_days()
        
    def log_forecast(self, prediction_df):
        """Corporate Compliance is a big thing.
        Every CEO needs a trustworthy internal audit, if you know what I mean."""
        self.env.predict(prediction_df)
    
    def file_away(self):
        """At the end of the day no one really knows what happend behind closed doors"""
        self.env.write_submission_file()


class MineralCleaner(CompanyEmployee):
    """Cleaning mineral specimens since 1858.
    There ain't no kind of stone he hasn't seen before.
    Spends lunchtime with the miner, his only confidant."""
    
    memory_scale_factor = 1024**2  # memory in MB

    def __init__(self, conv_table=None):
        if conv_table is None:
            self.conversion_table = \
                {'int': [np.int8, np.int16, np.int32, np.int64],
                 'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                 'float': [np.float32, ]}
        else:
            self.conversion_table = conv_table

    def _sound(self, k):
        """Press pressure sounding comes in handy for any gem collector."""
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    @CompanyEmployee.clocking_work
    def clean_minerals(self, df, verbose=False):
        """Oxalic and muriatic acid are his favorites."""
        print("Start cleaning minerals..")
        mem_usage_orig = df.memory_usage().sum() / self.memory_scale_factor
        
        ret_list = Parallel(n_jobs=1)(delayed(self._clean)
                                                (df[c], c, verbose) for c in
                                                df.columns)
        del df
        gc.collect()
        ret = pd.concat(ret_list, axis=1)
        
        mem_usage_new = ret.memory_usage().sum() / self.memory_scale_factor
        print(f"Reduced yield from {mem_usage_orig:.4f} MB to {mem_usage_new:.4f} MB.")
        return ret

    def _clean(self, s, colname, verbose):
        """When diluting always add acid to water, not water to acid."""
        # skip NaNs
        if s.isnull().any():
            if verbose:
                print(colname, 'has NaNs - Skip..')
            return s
        # detect kind of type
        coltype = s.dtype
        if np.issubdtype(coltype, np.integer):
            conv_key = 'int' if s.min() < 0 else 'uint'
        elif np.issubdtype(coltype, np.floating):
            conv_key = 'float'
        else:
            if verbose:
                print(colname, 'is', coltype, '- Skip..')
            return s
        # find right candidate
        for cand, cand_info in self._sound(conv_key):
            if s.max() <= cand_info.max and s.min() >= cand_info.min:

                if verbose:
                    print('convert', colname, 'to', str(cand))
                return s.astype(cand)


class DataMiner(CompanyEmployee):
    """Born in the mines, living in the dark, meant to perish in the ashes.
    Hacking is his purpose, dust his friend and the daily grind his fate."""
    
    merge_col_anchors = ['assetCode', 'date']
    
    def __init__(self, cleaner):
        self.cleaner = cleaner
    
    @CompanyEmployee.clocking_work
    def dig_in_the_mine(self, market_df, news_df):
        """The first half of the day digging is to pay off the digging license"""
        print("Digging..")
        
        # chop trainset to speed up kernel
        start = datetime(2011, 6, 1, 0, 0, 0).date()
        market_df = market_df.loc[market_df['time'].dt.date >= start].reset_index(drop=True)
        #news_df = news_df.loc[news_df['time'].dt.date >= start].reset_index(drop=True) # too memory hungry
        
        # trainset-only processing
        drop_cols =['assetName']
        market_df.drop(drop_cols, axis=1, inplace=True)
        market_df.dropna(inplace=True, axis=0)
        market_df['time'] = pd.to_datetime(market_df.time, utc=False)
        
        self.full_df = self.dig(market_df, news_df)
        self.full_df = self.aggregate_total_day_feats(self.full_df)
        print(self.full_df.shape)
        self.tra_df = self.full_df
        #self.tra_df = self.full_df.drop(['universe', 'assetCode', 'time'], axis=1)
        #self.tra_df = self.ask_for_cleansing(self.tra_df)
    
    def force_dig(self, market_df, news_df):
        """History repeats itself."""
        
        drop_cols =['assetName']
        market_df.drop(drop_cols, axis=1, inplace=True)
        market_df.dropna(inplace=True, axis=0)
        market_df['time'] = pd.to_datetime(market_df.time, utc=False)
        
        # get day
        day = market_df.time.dt.date[0]
        appended_market_df = self.full_df.append(market_df, sort=False).fillna(0)
        digged = self.dig(appended_market_df, news_df)
        # return day
        digged = digged.loc[digged.time.dt.date == day, [c for c in digged.columns if '_daily_' not in c]].reset_index(drop=True)
        digged = self.aggregate_total_day_feats(digged)
        return digged
    
    def aggregate_total_day_feats(self, df_):
        # total day aggregations
        cols_to_agg = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 
                     'returnsClosePrevMktres1', 'returnsOpenPrevMktres1','returnsClosePrevRaw10',
                     'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
                    'price_volume', 'bartrend_quot', 'bartrend_diff']
        day_aggs = {c: ['min', 'max', 'mean', 'std'] for c in cols_to_agg}
        daily_aggs = df_[['time']+ cols_to_agg].groupby('time').agg(day_aggs).astype(np.float32)
        daily_aggs.columns = ['{}_daily_{}'.format(a, b) for a, b in daily_aggs.columns]

        return df_.set_index('time').join(daily_aggs).reset_index()
    
    def dig(self, market_df, news_df):
        """Rock for rock, hack for hack, day for day"""
        #news_df = self.hack_for_news(news_df)
        #market_df = market_df.merge(news_df, how='left', on=self.merge_col_anchors)
        if news_df is not None:
            del news_df  # too memory hungry
        
        # sort by asset first, in order to not mix them up
        market_df.sort_values(by=['assetCode', 'time'], axis=0, inplace=True)
        assets_d = {k: v for k, v in market_df.groupby('assetCode')}
        quant_base_cols = ['close']#, 'volume']
        assets_proc_d = {}
        market_df['date'] = market_df.time.dt.date
        for lbl, df in assets_d.items():
            quant_feats = {'bartrend_quot': lambda x: x.close / x.open,
                           'bartrend_diff': lambda x: x.close - x.open,
                           'average': lambda x: (x.close + x.open)/ 2,
                           'price_volume': lambda x: x.volume * x.close,
                           'MACD_fast': lambda x: x.close.ewm(span=12).mean() - x.close.ewm(span=26).mean(),
                           'MACD_slow': lambda x: x['MACD_fast'].ewm(span=9).mean(),
                           'MACD_diff': lambda x: x['MACD_fast'] - x['MACD_slow'],
                           'MACD_diff_abs': lambda x: np.abs(x['MACD_diff']),
                           'MACD_diff_sign': lambda x: np.sign(x['MACD_diff']).astype(np.int8),               
                           'close_PrevRaw1_sub_PrevMktres1': lambda x: x['returnsClosePrevRaw1'] - x['returnsClosePrevMktres1'],
                           'close_PrevRaw10_sub_PrevMktres10': lambda x: x['returnsClosePrevRaw10'] - x['returnsClosePrevMktres10'],
                           'open_PrevRaw1_sub_PrevMktres1': lambda x: x['returnsOpenPrevRaw1'] - x['returnsOpenPrevMktres1'],
                           'open_PrevRaw10_sub_PrevMktres10': lambda x: x['returnsOpenPrevRaw10'] - x['returnsOpenPrevMktres10'],
                           }
            quant_feats.update({**{f'{c}_smoothed_9': lambda x: x[c].ewm(span=9).mean() for c in quant_feats if '_sub_' in c},
                                **{f'{c}_smoothed_18': lambda x: x[c].ewm(span=18).mean() for c in quant_feats if '_sub_' in c}})
            # EWMA
            # this ewma is inaccurate since data is not sampled daily
            days_list = [9, 18, 27]
            weeks_list = [11, 22, 33]
            for c in quant_base_cols:
                quant_feats.update({f'{c}_{n}_days_EWMA': lambda x: x[c].ewm(span=n).mean().astype(np.float32) for n in days_list})
                quant_feats.update({f'{c}_{n}_weeks_EWMA': lambda x: x[c].ewm(span=n*7).mean().astype(np.float32) for n in weeks_list})
                for i_1, n_1 in enumerate(days_list):
                    if i_1==0:
                        continue
                    quant_feats[f'{c}_ewma_diff_{n_1}_sub_{days_list[i_1-1]}_days'] = lambda x: x[f'{c}_{n_1}_days_EWMA'] - x[f'{c}_{days_list[i_1-1]}_days_EWMA']
                for i_2, n_2 in enumerate(weeks_list):
                    if i_2==0:
                        continue
                    quant_feats[f'{c}_EWMA_diff_{n_2}_sub_{weeks_list[i_2-1]}_weeks'] = lambda x: x[f'{c}_{n_2}_weeks_EWMA'] - x[f'{c}_{weeks_list[i_2-1]}_weeks_EWMA']
            """channels_dev = {**{k+'_upper_2std': lambda x: x[k].ewm(span=100).std()*2 + x[k] for k in quant_feats if k.endswith('_days_EWMA')},
                           **{k+'_lower_2std': lambda x: -x[k].ewm(span=100).std()*2 + x[k] for k in quant_feats if k.endswith('_days_EWMA')},
                           **{k+'_upper_2std': lambda x: x[k].ewm(span=7*100).std()*2 + x[k] for k in quant_feats if k.endswith('_weeks_EWMA')},
                           **{k+'_lower_2std': lambda x: -x[k].ewm(span=7*100).std()*2 + x[k] for k in quant_feats if k.endswith('_weeks_EWMA')},}
            quant_feats.update(channels_dev)"""
            
            for c in quant_base_cols:
                diff_quant_base_to_ewma = {k+f'_diff_to_{c}': lambda x: x[c] - x[k] for k in quant_feats if '_EWMA' in k}
        
                quant_feats.update({**diff_quant_base_to_ewma, 
                                    **{k+'_sign': lambda x: np.sign(x[k]) for k in quant_feats if k.endswith(f'_diff_to_{c}')},
                                    **{k+'_abs': lambda x: np.abs(x[k]) for k in quant_feats if k.endswith(f'_diff_to_{c}')},
                                     })
            assets_proc_d[lbl] = df.assign(**quant_feats)
        market_df = pd.concat(assets_proc_d.values()).reset_index(drop=True)
        
        return market_df
        
    def hack_for_news(self, df):
        """The news gem is precious and the company offers 
        lucrative incentives for its nourishment"""
        
        drop_list = ['audiences', 'subjects', 'assetName', 
                    'headline', 'firstCreated', 'sourceTimestamp']
        df.drop(drop_list, axis=1, inplace=True)
        
        # Factorize categorical columns
        for col in ['headlineTag', 'provider', 'sourceId']:
            df[col], uniques = pd.factorize(df[col])
            del uniques
        
        # reduce mem usage
        # todo: some columns cant be reduced and throw error, see "dig_in_the_mine"
        #df = self.ask_for_cleansing(df)
        
        # Unstack news_df across asset codes
        stacked_asset_codes = (df['assetCodes'].astype('object')
                                .apply(lambda x: list(eval(x)))
                                .apply(pd.Series)
                                .stack()
                                .reset_index(level=-1, drop=True))
        stacked_asset_codes.name = 'assetCode'  # note: no trailing "s"
        df = df.join(stacked_asset_codes)
        df.drop(['assetCodes'], axis=1, inplace=True)
        
        # group by assetcode and date, aggregate some stats
        agg_agenda = {                                                                                                                                                                                                  
            'urgency': ['min', 'count'],                                                                                                                                                                                   
            'takeSequence': ['max'],                                                                                                                                                                                       
            'bodySize': ['min', 'max', 'mean', 'std'],                                                                                                                                                                     
            'wordCount': ['min', 'max', 'mean', 'std'],                                                                                                                                                                    
            'sentenceCount': ['min', 'max', 'mean', 'std'],                                                                                                                                                                
            'companyCount': ['min', 'max', 'mean', 'std'],                             
            'marketCommentary': ['min', 'max', 'mean', 'std'],
            'relevance': ['min', 'max', 'mean', 'std'],
            'sentimentNegative': ['min', 'max', 'mean', 'std'],
            'sentimentNeutral': ['min', 'max', 'mean', 'std'],         
            'sentimentPositive': ['min', 'max', 'mean', 'std'],
            'sentimentWordCount': ['min', 'max', 'mean', 'std'],   
            'noveltyCount12H': ['min', 'max', 'mean', 'std'],
            'noveltyCount24H': ['min', 'max', 'mean', 'std'],                            
            'noveltyCount3D': ['min', 'max', 'mean', 'std'],
            'noveltyCount5D': ['min', 'max', 'mean', 'std'],   
            'noveltyCount7D': ['min', 'max', 'mean', 'std'],
            'volumeCounts12H': ['min', 'max', 'mean', 'std'],
            'volumeCounts24H': ['min', 'max', 'mean', 'std'],                    
            'volumeCounts3D': ['min', 'max', 'mean', 'std'],
            'volumeCounts5D': ['min', 'max', 'mean', 'std'],
            'volumeCounts7D': ['min', 'max', 'mean', 'std']  
        }    
        df['date'] = df.time.dt.date
        df['position'] = df['firstMentionSentence'] / df['sentenceCount']
        df['coverage'] = df['sentimentWordCount'] / df['wordCount']

        grouped = df.groupby(self.merge_col_anchors).agg(agg_agenda).astype('float32')
        grouped.columns = ['_'.join(c) for c in grouped.columns]
        
        return grouped.reset_index()
    
    def ask_for_cleansing(self, df):
        """Every now and then some gems get dissolved accidentally.
        This is the official statement."""
        
        return self.cleaner.clean_minerals(df)
    

class Directrice(CompanyEmployee):
    """Call the director if things are getting serious. 
    She's keeping track of models and compiles forecasts for the company.
    Her career is the most important thing in her life and so she sticks at nothing. 
    Her sense of thievishness is infamous."""
    
    def __init__(self):
        """Subordinates are just means to an end."""
        super().__init__()
        self.miner = None
        self.models = [LogisticRegression(solver='liblinear'),]
        self.scaler = StandardScaler()
    
    def interrogate(self, miner):
        """Shake a miner down."""
        # make the miner her personal tool
        self.miner = miner
        # Train models according to what the miner has unearthed
        self.train_models()
    
    @CompanyEmployee.clocking_work
    def train_models(self):
        bin_target = (self.miner.tra_df.returnsOpenNextMktres10 >= 0).astype('int8')
        drop_cols = [c for c in ['returnsOpenNextMktres10', 'date', 'universe', 
                        'assetCode', 'assetName', 'time'] if c in self.miner.tra_df.columns] 
        tra_df = self.scaler.fit_transform(self.miner.tra_df.drop(drop_cols, axis=1))
        
        print('Start training models..')
        print(tra_df.shape)
        for model in self.models:
            model.fit(tra_df, bin_target)
            
    def forecast(self, market_obs_df, news_obs_df, pred_templ_df):
        """Predict the future deposit in the mines that will be unearthed."""
        
        # make the miner put in unpaid extra work
        df = self.miner.force_dig(market_obs_df, news_obs_df)
        drop_cols = [c for c in ['returnsOpenNextMktres10', 'date', 'universe', 
                        'assetName', 'time'] if c in df.columns] 

        df = (df.loc[df.assetCode.isin(pred_templ_df.assetCode)]
                .drop(drop_cols, axis=1)
                .set_index('assetCode'))
        
        df = self.scaler.transform(df)
        # heavy lifting
        for i, model in enumerate(self.models):
            df[f'pred_{i}'] = model.predict_proba(df)[:, 1]
        df['pred'] = df.loc[:, [c for c in df.columns if 'pred_' in c]].mean(axis=1)/ len(self.models)
        df['pred'] = np.clip(df.pred*2-1, -1, 1)
        pred_templ_df = (pred_templ_df.set_index('assetCode')
                        .join(df['pred'])
                        .drop(['confidenceValue'], axis=1)
                        .reset_index()
                        .rename(columns={'pred': 'confidenceValue'}))
        print(pred_templ_df.describe())
        return pred_templ_df


if '__main__' == __name__:
    kaggliton = Company()
    kaggliton.grow()