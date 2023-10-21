# Imports
from main import  country_index_classifier

from visualization import plot_classifier_results

from utils import calculate_confidence
from plotly.offline import plot

# Configuration
LOGIN = "51545562"
PASSWORD = "zop7gsit"
SERVER = "Alpari-MT5-Demo"
initialize = [LOGIN, PASSWORD, SERVER]



## Country Index Classfier
country='USD'

for method in ['fund', 'quant']:
    for TimeFrame in ['1d', '1w']:
        for use_haiken_ashi in [True, False]:

            forecast, prob , outputs= country_index_classifier(method=method, initialize=initialize, country=country, TimeFrame=TimeFrame,
                                                      use_haiken_ashi=use_haiken_ashi, hyper_tune=False, plot_results=False,
                                                      update_fund_data= False)
            
            fig = plot_classifier_results(f'{country}/{TimeFrame} - Method: {method} - Haiken Ashi: {use_haiken_ashi}', calculate_confidence(forecast, prob, method))
            plot(fig)



