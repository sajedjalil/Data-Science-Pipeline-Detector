# -*- coding: utf-8 -*-

class FFMFormat:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None

    def get_params(self):
        pass
    
    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i,col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            vals = np.unique(df[col])
            for val in vals:
                if np.isnan(val): continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row):
        ffm = []
        for col,val in row.loc[row!=0].to_dict().iteritems():
            name = '{}_{}'.format(col, val)
            ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        return pd.Series({idx: self.transform_row_(row) for idx,row in df.iterrows()})