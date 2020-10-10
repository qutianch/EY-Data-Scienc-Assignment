# Case Assignment with EY - Data & Analytics
## Prepared by Tiancheng Qu


The assignment is about predicting total sales for every product and store in the next month for a time series dataset consisting of daily sales data.

Submission package should consist of the following:
Script used to conduct the analysis
Excluding cover page, a maximum of 5 slides for purpose of communicating results
There is no set requirement on formatting, however, inclusion of the following sections is recommended.
- Approach including tools used
- Key data cleansing, transformation, variable engineering steps taken
- Model evaluation metric and respective results
- Business implications and recommendations

Dataset descriptions:
- sales.csv: Daily historical data from January 2013 to October 2015.
- items.csv: supplemental information about the items/products.
- item_categories.csv: supplemental information about the items categories.
- shops.csv: supplemental information about the shops.

# Recommendations
- Increase stock for Games root category(includes both PC and Console subcategories) items at all shops.
- Increase stock for Tickets root category(includes both Ticket(digits) and Utilities-Tickets) items at all shops.
- Increase stock for Gift root category items at all shops.
- ‘Khimki TC "Mega"’, Moscow TC "Perlovskiy“, Krasnoyarsk TC "Vzletka Plaza“, Ufa TC "Central“, Mytishchi TRK "XL-3“, St. Petersburg TK "Sennaya“, Yakutsk Ordzhonikidze, 56, RostovNaDonu TRC "Megacenter Horizon" Island and Tyumen SC "Goodwin“ shops predicted to have significant decrease in revenue. It is worth looking into the store management and or other store performance matrix.
- Game Console promotions may be needed for Shop Online Emergencies, Moscow shopping center "Semyonov“, and Moscow shopping center "MEGA Teply Stan" II of.  Since they are predicted to suffer major revenue lost.

# Approach-Process
- Python and its various libraries were used as the scripting language. 
- Jupyter Notebook was used to preform the coding, visualization and analytics.
- Use sales record as the base to create an information rich dataframe that joins the information from item, item category and shops that are active.
- Generalized various of similar item categories into one root category
- Check and correct any ‘incorrect’ information such as unexpected item price.
- Group the sales data by shop name and item’s root category
- Utilizing time series analysis and prediction models (SARIMA, ARIMA) to make subsequent month (2016-01-01) shop root category sales predictions.
- The model with the better test set performance will be the one that is making the sales prediction in each case.
- Identify the shop-root category groups where predicted to have lower sales then pervious years at same month
- Create recommendations based on the root category characteristics

# Approach-Assumptions
- Assume inactive is represented as 'active_flag’ ==‘X’ 
- Assume generalized root categories are sufficient for making ‘detailed’ prediction for total sales for - ‘every product’ and store.
- Assume negative value of item_cnt_day means return of the product and full refund was given.
- Assume empty entry or item_cnt_day indicate no sale or return of such product.
- Assume shop with the root category that has no sales record for 24 months before the end of the sales record are not active and no prediction will be generated.
- Assume extreme high ‘item_price’ during October 2013 were data entry error.

# Key Actions
1. Excluded any row that was marked as inactive.

Drop inactive from item_categories, item_root_categories and shops df:
```python
item_categories_df_active = item_categories_df.loc[item_categories_df['active_flag'] !='X']
item_root_categories_df_active = item_root_categories_df.loc[item_root_categories_df['active_flag'] !='X']
shops_df_active = shops_df.loc[shops_df['active_flag'] !='X']

```

2. Translated ‘Игры’ to ‘Games’ in ‘item_category_name’ 

![translation](/Screenshots/translation.png) 

3. Upon inspected the ‘item_category_name’ column, ‘item_root_categories’ was created and the values were manually added to the file based on the similarity between each category name.

|FIELD1|root_category|item_category_name                                                                                                                                                                                                                                                                                                                            |
|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|0     |Accessories  |['Accessories - PS2' 'Accessories - PS3' 'Accessories - PS4'  'Accessories - PSP' 'Accessories - PSVita' 'Accessories - XBOX 360'  'Accessories - XBOX ONE' 'Games - Accessories for games'  'PC - Headsets / Headphones']                                                                                                                    |
|1     |Books        |['Books - artbook, encyclopedia' 'Books - Audiobooks'  'Books - Audiobooks (figure)' 'Books - Audiobooks 1C'  'Books - Business Books' 'Books - Comics, Manga' 'Books - Computer Books'  'Books - Digital' 'Books - Fiction' 'Books - Informative literature'  'Books - Methodical materials 1C' 'Books - Postcards'  'Books - Travel Guides']|
|2     |Cinema       |['Cinema - Collector']                                                                                                                                                                                                                                                                                                                        |
|3     |Delivery     |['Delivery of goods']                                                                                                                                                                                                                                                                                                                         |
|4     |Game consoles|['Game consoles - Other' 'Game consoles - PS2' 'Game consoles - PS3'  'Game consoles - PS4' 'Game consoles - PSP' 'Game consoles - PSVita'  'Game consoles - XBOX 360' 'Game consoles - XBOX ONE']                                                                                                                                            |
|5     |Games        |['Игры - XBOX ONE' 'Games - PS2' 'Games - PS3' 'Games - PS4' 'Games - PSP'  'Games - PSVita' 'Games - XBOX 360' 'Games Android - Digital'  'Games MAC - Digital' 'Games PC - Additional publications'  "Games PC - Collector's Edition" 'Games PC - Digital'  'Games PC - Standard Edition']                                                  |
|6     |Gifts        |['Gifts - Attributes' 'Gifts - Bags, Albums, Mats d / mouse'  'Gifts - Board Games' 'Gifts - Cards, Stickers'  'Gifts - certificates, services' 'Gifts - Development' 'Gifts - Figures'  'Gifts - gadgets, robots, sports' 'Gifts - Games (compact)'  'Gifts - Soft Toys' 'Gifts - Souvenirs' 'Gifts - Souvenirs (weighed in)']               |
|7     |Movies       |['Movie - Blu-Ray 4K' 'Movie - DVD' 'Movies - Blu-Ray'  'Movies - Blu-Ray 3D']                                                                                                                                                                                                                                                                |
|8     |Musics       |['Music - CD of local production' 'Music - CD production firm'  'Music - Gift Edition' 'Music - MP3' 'Music - Music video'  'Music - Vinyl']                                                                                                                                                                                                  |
|9     |Net carriers |['Net carriers (piece)' 'Net carriers (spire)']                                                                                                                                                                                                                                                                                               |
|10    |Payment Cards|['Payment card - Windows (figure)' 'Payment card (Movies, Music, Games)'  'Payment cards - Live!' 'Payment cards - Live! (Numeral)'  'Payment cards - PSN']                                                                                                                                                                                   |
|11    |Programs     |['Program - 1C: Enterprise 8' 'Program - Educational'  'Program - For home and office' 'Program - Home & Office (Digital)'  'Program - MAC (figure)' 'Programs - Educational (figure)' 'System Tools']                                                                                                                                        |
|12    |Tickets      |['Tickets (digits)' 'Utilities - Tickets']                                                                                                                                                                                                                                                                                                    |
|13    |batteries    |['batteries']                                                                                                                                                                                                                                                                                                                                 |

4. Join the item dataframe with item category dataframe.
```python
item_and_categories_df_working = items_df.set_index('category').join(item_root_categories_df_active.set_index('ID'))
item_and_categories_df_working = item_and_categories_df_working.reset_index().iloc[:, 1:-1]
```
5. Join the sales dataframe with shops dataframe.
```python
sales_df_working = sales_df.set_index('shop_id').join(shops_df_active.set_index('id')).iloc[:, :-1]
```
6. Join the results from Action 3 and 4 to create the comprehensive dataframe, referred as ‘sales_df_working_final’.
```python
sales_df_working_final = sales_df_working.set_index('item_id').join(item_and_categories_df_working.set_index('id'))
```
7. Multiplied the ‘item_price’ with ‘item_cnt_day’ to create the column ‘revenue’ in the ‘sales_df_working’.
```python
sales_df_working_final['revenue'] = sales_df_working_final['item_price']*sales_df_working_final['item_cnt_day']
```
8. Calculated and visualized each shop’s monthly revenue, using aggregation.
```python
sales_df_working_final_store_revenue = sales_df_working_final.set_index('date').groupby(['name']).resample("M").aggregate({'revenue':'sum'}).reset_index()
```
9. Large spike of revenue for almost all shop during October 2013 was observed. Upon close inspection, some of the item_price were abnormally high even compare with its price history within the sales record. Is it likely due to human entry error. 
```python
import plotly.express as px
df = px.data.gapminder()
fig = px.line(sales_df_working_final_store_revenue, x="date", y="revenue", color="name", line_group="name", hover_name="name",
        line_shape="spline", render_mode="svg")

fig.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
fig.show()
```

![spike](/Screenshots/spike.png) 

10. Further research into the true possible item price online confirmed the assumption, a threshold of ₽6000 was determined and any item price that is higher will be divided by 100.
```python
threshold = 6000
sales_df_working_final_oct = sales_df_working_final.loc[(sales_df_working_final['date']>='10.1.2013')&
                                                        (sales_df_working_final['date']<='10.31.2013')&
                                                        (sales_df_working_final['item_price']>=threshold)]
sales_df_working_final_oct_adj = sales_df_working_final_oct.copy()
sales_df_working_final_oct_adj['item_price'] = sales_df_working_final_oct_adj['item_price']/100


sales_df_working_final_oct_exclude = sales_df_working_final.loc[~((sales_df_working_final['date']>='10.1.2013')&
                                                        (sales_df_working_final['date']<='10.31.2013')&
                                                        (sales_df_working_final['item_price']>=threshold))]
sales_df_working_final_adj= pd.concat([sales_df_working_final_oct_exclude, sales_df_working_final_oct_adj], ignore_index=True)
```

11.The sales record dataframe with adjustments was visualized again to verify that the effect of data entry error has been reduced.

| Before | After |
|--------|-------|
| ![spike](/Screenshots/spike.png) | ![reduced](/Screenshots/reduced.png) |

12. Two types of times series model were selected (SARIMAX and ARIMA)
SARIMAX_model
```python
def SARIMAX_model(train,test):
    ## Time series forecasting with SARIMAX
    #### find the optimal set of parameters that yields the best performance for our model
    p = d = q = range(0, 2)
    
    pdq = list(itertools.product(p, d, q))
    
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
                    
    search_result_list = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False,
                                                suppress_warnings=True)
                results = mod.fit()
                search_result_list.append((param,param_seasonal,results.mae))
                #print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    best_parm = min(search_result_list, key = lambda i : i[2])[:]
    SARIMAX_order = best_parm[0]
    SARIMAX_seasonal_order = best_parm[1]
    
    mod = sm.tsa.statespace.SARIMAX(train,
                                order=SARIMAX_order,
                                seasonal_order=SARIMAX_seasonal_order,
                                suppress_warnings=True,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
        
    SARIMAX_test_pred = results.get_prediction(start=test.index.min(), end=test.index.max(),dynamic=False).predicted_mean.tolist()
    store_sales_df_test_truth = test['revenue'].tolist()
    SARIMAX_MAE = mean_absolute_error(store_sales_df_test_truth, SARIMAX_test_pred)
    SARIMAX_pred = results.get_forecast(steps=1).predicted_mean[0]
    
    return (SARIMAX_order,SARIMAX_seasonal_order,SARIMAX_MAE,SARIMAX_pred)
```
ARIMA_model
```python
def ARIMA_model(train,test):
    stepwise_model = pm.auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=False,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    
    ARIMA_order = stepwise_model.order
    ARIMA_seasonal_order = stepwise_model.seasonal_order
    results_ARIMA = stepwise_model.fit(train)
    ARIMA_test_pred = stepwise_model.predict(n_periods=len(test))# This returns an array of predictions:>>>
    store_sales_df_test_truth = test['revenue'].tolist()
    ARIMA_MAE =  mean_absolute_error(store_sales_df_test_truth, ARIMA_test_pred)
    
    ARIMA_pred = stepwise_model.predict(n_periods=len(test)+1)[-1]
    
    return (ARIMA_order,ARIMA_seasonal_order,ARIMA_MAE,ARIMA_pred)
```
Third_model_not used at this time
```python
def BenchMark(train,test):
    ## get benck mark mae
    series = store_sales_df
    # prepare data
    X = series.values
    X = X.astype('float32')
    train_size = int(len(train))
    train_, test_ = X[0:train_size], X[train_size:]
    # walk-forward validation
    history = [x for x in train_]
    predictions = list()
    for i in range(len(test_)):
        # predict
        yhat = history[-1]
        predictions.append(yhat)
        # observation
        obs = test_[i]
        history.append(obs)
        #print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    # report performance
    benchmark_mae = mean_absolute_error(test_, predictions)
    return(benchmark_mae)
```
13. shop with the root category that has no sales record for 24 months before the end of the sales record are considered not active and no predictions were generated.
```python
    if ((len(store_sales_df['2014-01-31':])>=24)):
```
14. The model with the better test set performance will be the one that is making the sales prediction in each case.
Mean Absolut Error was used as target, the lower the better
```python
    store_cat_model_df_result['BestModel'] = store_cat_model_df_result[['mae_sarimax','mae_arima']].idxmin(axis=1)
```
15. All models’ best parameters and predictions were saved in dataframe for reproduction. 
```python
    store_cat_model_df_result_sales['BestPred'] = -1
    for index,row in store_cat_model_df_result_sales.iterrows():
        bestModel = row.BestModel
        bestModel_pred_name = 'pred_'+bestModel[4:]
        bestPred = store_cat_model_df_result_sales.at[index,bestModel_pred_name]
        store_cat_model_df_result_sales.at[index,'BestPred'] = bestPred
    store_cat_model_df_result_sales['Change'] = store_cat_model_df_result_sales['BestPred']-store_cat_model_df_result_sales['revenue']
    store_cat_model_df_result_sales[['name','cat']] = store_cat_model_df_result_sales.name_cat.str.split('_',expand=True) 
```

# Model Evaluation and Results
- There were 704 total possible store name and root category combinations.
- The models were able to make 488 predictions.
- SARIMAX was up to 2x better at predicting the sales winning 331 times, vs. ARIMA at 157 times.
- Based on the predicted store and root category pair, we are expecting to earn ₽15M more revenue compared with 2015 January.
- 9 out of 50 shops are expected to make less revenue compared with 2015
- Khimki TC "Mega“ is predicted to lost the most amount of revenue (- ₽2.47M ). Revenue lost in every root categories. This could be the problem with the model or the store, further investigation required.
- Moscow TRC "Atrium“ is predicted to generate the most amount of revenue ( +₽2.51M )
- The predicted revenue gain leaders are: Games, Tickets, Gifts and Delivery
- The predicted revenue loss leaders are: Game consoles, Payment Cards, Movies and Programs

