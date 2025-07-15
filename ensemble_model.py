class EnsembleModel:
    def __init__(self, rf_model, gb_model, rf_weight=0.5, gb_weight=0.5):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.rf_weight = rf_weight
        self.gb_weight = gb_weight
    
    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)
        return self.rf_weight * rf_pred + self.gb_weight * gb_pred
