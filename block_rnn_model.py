from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr
from darts.models import BlockRNNModel
from darts.models import TCNModel
from darts.models import TFTModel


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


BlockRNNModel_model = BlockRNNModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    random_state=42,
    n_epochs=100,
)

BlockRNNModel_model.fit(train_scaled, past_covariates=past_covs, verbose=True)



BlockRNNModel_model_scaled = BlockRNNModel_model.predict(n=len(val), past_covariates=past_covs)
BlockRNNModel_model_pred = scaler.inverse_transform(BlockRNNModel_model_scaled)

plt.figure(figsize=(10, 6))
val.plot(label="actual")
BlockRNNModel_model_pred.plot(label="forecast")

plt.savefig("./output/BlockRNNModel_forecast_plot.png")



accuracy_metrics_dict["BlockRNNModel"] = { }
accuracy_metrics_dict["BlockRNNModel"]["mse"] = mse(BlockRNNModel_model_pred, val)
accuracy_metrics_dict["BlockRNNModel"]["mape"] = mape(BlockRNNModel_model_pred, val)
accuracy_metrics_dict["BlockRNNModel"]["smape"] = smape(BlockRNNModel_model_pred, val)

metrics_df = pd.DataFrame(accuracy_metrics_dict)
metrics_df.to_csv("./output/BlockRNNModelmetrics.csv")

# TCNModel_model = TCNModel(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
#     random_state=42,
#     n_epochs=50,
# )

# TCNModel_model.fit(train_scaled, past_covariates=past_covs, verbose=True)

# TCNModel_pred_scaled = TCNModel_model.predict(series=train_scaled, past_covariates=past_covs.astype("float32"), n=len(val))
# TCNModel_model_pred = scaler.inverse_transform(TCNModel_pred_scaled)

# plt.figure(figsize=(10, 6))
# val.plot(label="actual")
# TCNModel_model_pred.plot(label="forecast")
# plt.savefig("./output/TCNModel_forecast_plot.png")

# accuracy_metrics_dict["TCNModel"] = { }
# accuracy_metrics_dict["TCNModel"]["mse"] = mse(TCNModel_model_pred, val)
# accuracy_metrics_dict["TCNModel"]["mape"] = mape(TCNModel_model_pred, val)
# accuracy_metrics_dict["TCNModel"]["smape"] = smape(TCNModel_model_pred, val)

# metrics_df = pd.DataFrame(accuracy_metrics_dict)
# metrics_df.to_csv("./output/TCNModel_metrics.csv")

# TFTModel_model = TFTModel(
#     input_chunk_length=input_chunk_length,
#     output_chunk_length=output_chunk_length,
#     random_state=42,
#     n_epochs=50,
# )

# TFTModel_model.fit(train_scaled, past_covariates=past_covs, future_covariates=future_covs, verbose=True)

# TFTModel_scaled = TFTModel_model.predict(n=len(val), past_covariates=past_covs, future_covariates=future_covs)
# TFTModel_pred = scaler.inverse_transform(TFTModel_scaled)

# plt.figure(figsize=(10, 6))
# val.plot(label="actual")
# TFTModel_pred.plot(label="forecast")
# plt.savefig("./output/TFTModel_forecast_plot.png")

# accuracy_metrics_dict["TFTModel"] = { }
# accuracy_metrics_dict["TFTModel"]["mse"] = mse(TFTModel_pred, val)
# accuracy_metrics_dict["TFTModel"]["mape"] = mape(TFTModel_pred, val)
# accuracy_metrics_dict["TFTModel"]["smape"] = smape(TFTModel_pred, val)

# metrics_df = pd.DataFrame(accuracy_metrics_dict)
# metrics_df.to_csv("./output/TFTModel_metrics.csv")