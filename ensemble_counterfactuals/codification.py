class Codificacion:
    def __init__(self,X,names_order):
        self.X = X
        self.columns = X.columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        self.categorical_cols = categorical_cols
        column_to_values = {}
        for col in categorical_cols:
            column_to_values[col] = list(X[col].unique())
        self.column_to_values = column_to_values
        self.cls = names_order

    def encode(self,desc_list):
        enc_list = []
        for i,e in enumerate(desc_list):
            enc_list.append(self.column_to_values[self.columns[i]].index(e))
        return enc_list

    def decode(self,enc_list):
        desc_list = []
        for i,e in enumerate(enc_list):
            desc_list.append(self.column_to_values[self.columns[i]][round(e)])
        return desc_list
    
    def encode_class(self,desc_class):
        return self.cls.index(desc_class)

    def decode_class(self,enc_class):
        return self.cls[enc_class]

    def get_min(self):
        return [0]*len(self.column_to_values)
    
    def get_max(self):
        max_values = []
        for col in self.columns:
            max_values.append(len(self.column_to_values[col])-1)
        return max_values

    def show_codification(self):
        for x in self.categorical_cols:
            print(f'{x}: {self.column_to_values[x]}')