from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr
from darts.models import RNNModel
from darts.models import NHiTSModel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta 
import pytz
from darts.metrics import mape, mse, smape



accuracy_metrics_dict = { }
today_date = pd.Timestamp(datetime.today(), tz=pytz.UTC)

data = pd.DataFrame()

crypto = 'SOL'

all_price_data = yf.download(f'{crypto}-USD', interval='1h', period='2y')

if all_price_data.empty:
    print(f"No data available for {crypto} over the selected period.")
else:
    all_price_data.index = all_price_data.index.tz_localize(None)
    idx_dates = pd.date_range(start=min(all_price_data.index), end=today_date.tz_localize(None), freq='H')
    data = all_price_data.reindex(idx_dates)
    data[f'{crypto}'] = all_price_data['Close'].reindex(data.index, method='ffill')

# print(data.head())
# print(data.tail())

data = data.reset_index()
data.rename(columns={"index": "datetime"}, inplace=True)

print(data.head())

data = data[["datetime", "SOL"]]


plt.figure(figsize=(14, 7))

plt.plot(data["datetime"], data["SOL"])

plt.title('SOL Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

plt.legend()

plt.grid(True)
plt.savefig("./output/data.png")
# plt.show()


series = TimeSeries.from_dataframe(data, time_col="datetime", value_cols=["SOL"])

val_set_size = 10
val_set_split_date = max(data["datetime"])-timedelta(days=val_set_size)
train, val = series.split_before(val_set_split_date)



scaler = Scaler()
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
train_scaled = train_scaled.astype("float32")
val_scaled = scaler.transform(val)
val_scaled = val_scaled.astype("float32")




past_covs = concatenate(
    [
        dt_attr(series.time_index, "month", dtype=np.float32) / 12,
        dt_attr(series.time_index, "year", dtype=np.float32) / max(train.time_index.year),
        dt_attr(series.time_index, "day", dtype=np.float32) / 31,
        dt_attr(series.time_index, "dayofweek", dtype=np.float32) / 7,
        dt_attr(series.time_index, "week", dtype=np.float32) / 52,
        dt_attr(series.time_index, "dayofyear", dtype=np.float32) / 365,

    ],
    axis="component",
)


future_covs = concatenate(
    [
        dt_attr(series.time_index, "month", dtype=np.float32) / 12,
        dt_attr(series.time_index, "year", dtype=np.float32) / max(train.time_index.year),
        dt_attr(series.time_index, "day", dtype=np.float32) / 31,
        dt_attr(series.time_index, "dayofweek", dtype=np.float32) / 7,
        dt_attr(series.time_index, "week", dtype=np.float32) / 52,
        dt_attr(series.time_index, "dayofyear", dtype=np.float32) / 365
    ],
    axis="component",
)


# 

input_chunk_length = 2 * len(val)
output_chunk_length = len(val)


rnn_model = RNNModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    random_state=42,
    n_epochs=50,
    training_length=input_chunk_length  # Make sure this is >= input_chunk_length
)

rnn_model.fit(train_scaled, future_covariates=future_covs, verbose=True)
rnn_pred_scaled = rnn_model.predict(n=len(val), future_covariates=future_covs)
rnn_pred = scaler.inverse_transform(rnn_pred_scaled)
rnn_model_pred_scaled = rnn_model.predict(series=train_scaled, future_covariates=future_covs.astype("float32"), n=len(val))
rnn_model_model_pred = scaler.inverse_transform(rnn_model_pred_scaled)

plt.figure(figsize=(10, 6))
val.plot(label="actual")
rnn_model_model_pred.plot(label="forecast")
rnn_model_model_pred.plot(label="forecast")
plt.savefig("./output/RNNModel_forecast_plot.png")



accuracy_metrics_dict["RNNModel"] = { }
accuracy_metrics_dict["RNNModel"]["mse"] = mse(rnn_model_model_pred, val)
accuracy_metrics_dict["RNNModel"]["mape"] = mape(rnn_model_model_pred, val)
accuracy_metrics_dict["RNNModel"]["smape"] = smape(rnn_model_model_pred, val)

metrics_df = pd.DataFrame(accuracy_metrics_dict)
metrics_df.to_csv("./output/RNNModel_metrics.csv")


NHiTSModel_model = NHiTSModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    random_state=42,
    n_epochs=50,
)

NHiTSModel_model.fit(train_scaled, past_covariates=past_covs, verbose=True)
NHiTSModel_pred_scaled = NHiTSModel_model.predict(n=len(val), past_covariates=past_covs)
NHiTSModel_pred = scaler.inverse_transform(NHiTSModel_pred_scaled)
NHiTSModel_pred_scaled = NHiTSModel_model.predict(series=train_scaled, past_covariates=past_covs.astype("float32"), n=len(val))
NHiTSModel_pred = scaler.inverse_transform(NHiTSModel_pred_scaled)

plt.figure(figsize=(10, 6))
val.plot(label="actual")
NHiTSModel_pred.plot(label="forecast")
NHiTSModel_pred.plot(label="forecast")
plt.savefig("./output/NHiTSModel_forecast_plot.png")

accuracy_metrics_dict["NHiTSModel"] = { }
accuracy_metrics_dict["NHiTSModel"]["mse"] = mse(NHiTSModel_pred, val)
accuracy_metrics_dict["NHiTSModel"]["mape"] = mape(NHiTSModel_pred, val)
accuracy_metrics_dict["NHiTSModel"]["smape"] = smape(NHiTSModel_pred, val)

metrics_df = pd.DataFrame(accuracy_metrics_dict)
metrics_df.to_csv("./output/NHiTSModel_metrics.csv")