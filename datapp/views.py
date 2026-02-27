from django.shortcuts import render, redirect
from django import forms
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q

import os
import json
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings

from pyecharts.charts import Kline, Bar, Grid, Line
from pyecharts import options as opts

from .models import User, Gp

warnings.filterwarnings("ignore")


# -------------------------
# Forms (English UI strings)
# -------------------------

class LoginForm(forms.Form):
    username = forms.CharField(
        label="Username",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Enter username"}),
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "Enter password"}, render_value=True),
    )


class RegisterForm(forms.Form):
    username = forms.CharField(
        label="Username",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Enter username"}),
        max_length=16,
    )
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "Enter password"}),
    )
    password1 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={"class": "form-control", "placeholder": "Re-enter password"}),
    )

    def clean_username(self):
        username = self.cleaned_data.get("username")
        if User.objects.filter(username=username).exists():
            raise ValidationError("This username is already registered.")
        return username

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password1 = cleaned_data.get("password1")
        if password and password1 and password != password1:
            raise ValidationError("The two passwords do not match.")
        return cleaned_data


# -------------------------
# Auth Views
# -------------------------

def login(request):
    if request.method == "GET":
        form = LoginForm()
        return render(request, "login.html", {"form": form})

    form = LoginForm(data=request.POST)
    if not form.is_valid():
        return render(request, "login.html", {"form": form})

    username = form.cleaned_data["username"]
    password = form.cleaned_data["password"]

    user = User.objects.filter(username=username).first()
    if user and user.check_password(password):
        request.session["info"] = {"name": username}
        request.session.set_expiry(6000)
        return redirect("/home/")

    error = "Invalid username or password."
    return render(request, "login.html", {"form": form, "error": error})


def logout(request):
    request.session.clear()
    return redirect("/login/")


def register(request):
    if request.method == "GET":
        form = RegisterForm()
        return render(request, "register.html", {"form": form})

    form = RegisterForm(request.POST)
    if not form.is_valid():
        return render(request, "register.html", {"form": form, "error": "Please correct the errors and try again."})

    username = form.cleaned_data["username"]
    password = form.cleaned_data["password"]
    password1 = form.cleaned_data["password1"]

    if User.objects.filter(username=username).exists():
        return render(request, "register.html", {"form": form, "error": "This username already exists."})

    if password != password1:
        return render(request, "register.html", {"form": form, "error": "The two passwords do not match."})

    if not password.strip():
        return render(request, "register.html", {"form": form, "error": "Password cannot be empty or whitespace."})

    if len(password) < 6:
        return render(request, "register.html", {"form": form, "error": "Password must be at least 6 characters."})

    user = User(username=username, password=password)
    user.save()
    return redirect("/login/")


# -------------------------
# Home (Search / Filter / Pagination)
# -------------------------

def home(request):
    search_query = request.GET.get("search", "")
    selected_code = request.GET.get("code", "")

    datas = Gp.objects.all().order_by("-date", "code")

    if search_query:
        datas = datas.filter(
            Q(code__icontains=search_query) |
            Q(date__icontains=search_query)
        )

    if selected_code:
        datas = datas.filter(code=selected_code)

    codes = Gp.objects.values("code").distinct().order_by("code")

    paginator = Paginator(datas, 50)
    page = request.GET.get("page")

    try:
        data_page = paginator.page(page)
    except PageNotAnInteger:
        data_page = paginator.page(1)
    except EmptyPage:
        data_page = paginator.page(paginator.num_pages)

    return render(
        request,
        "home.html",
        {
            "data_page": data_page,
            "codes": codes,
            "selected_code": selected_code,
            "search_query": search_query,
        },
    )


# -------------------------
# Data Collection (Xueqiu)
# -------------------------

def data(request):
    if request.method == "POST":
        code = request.POST.get("code")
        start = request.POST.get("start")
        end = request.POST.get("end")

        try:
            start_ts = str(int(datetime.timestamp(datetime.strptime(start, "%Y-%m-%d %H:%M:%S")))) + "000"
            end_ts = str(int(datetime.timestamp(datetime.strptime(end, "%Y-%m-%d %H:%M:%S")))) + "000"
            cra(code, start_ts, end_ts)
            msg = f"Successfully collected data for {code}."
            return render(request, "data.html", {"msg": msg})
        except Exception as e:
            msg = f"Failed to collect data for {code}. Error: {e}"
            return render(request, "data.html", {"msg": msg})

    return render(request, "data.html")


def cra(code, start, end):
    cookie = os.getenv("XUEQIU_COOKIE", "")
    if not cookie:
        raise ValueError("Missing XUEQIU_COOKIE. Please set it in your .env file for data collection.")

    headers = {
        "authority": "stock.xueqiu.com",
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cookie": cookie,
        "dnt": "1",
        "origin": "https://xueqiu.com",
        "referer": "https://xueqiu.com",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0",
    }
   
    params = (
        ("symbol", code),
        ("begin", start),
        ("end", end),
        ("period", "day"),
        ("type", "before"),
        ("indicator", "kline"),
    )

    response = requests.get(
        "https://stock.xueqiu.com/v5/stock/chart/kline.json",
        headers=headers,
        params=params,
        timeout=30,
    )
    print("status:", response.status_code)
    print("text:", response.text[:300])
    data = response.json()

    data_list = data["data"]["item"]
    columns = data["data"]["column"]


    for item in data_list:
        timestamp = item[0] / 1000
        date = datetime.fromtimestamp(timestamp)
        item[0] = date.strftime("%Y-%m-%d")

    df = pd.DataFrame(data_list, columns=columns)

    if Gp.objects.filter(code=code).exists():
        Gp.objects.filter(code=code).delete()


    df.to_csv("data1.csv", index=False)

    for _, row in df.iterrows():
        Gp.objects.create(
            code=code,
            date=row["timestamp"],
            open=row["open"],
            close=row["close"],
            high=row["high"],
            low=row["low"],
            volume=row["volume"],
        )


# -------------------------
# Charts (Kline + Volume)
# -------------------------

def create_kline_chart(stock_code, kline_data, dates, stock_name, volumes):
    kline = (
        Kline()
        .add_xaxis(dates)
        .add_yaxis(
            series_name="Candlestick",
            y_axis=kline_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ec0000",
                color0="#00da3c",
                border_color="#8A0000",
                border_color0="#008F28",
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{stock_name} ({stock_code}) Candlestick Chart", pos_left="0"),
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            legend_opts=opts.LegendOpts(is_show=True),
            axispointer_opts=opts.AxisPointerOpts(is_show=True, link=[{"xAxisIndex": "all"}]),
            datazoom_opts=[
                opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1], range_start=0, range_end=100),
                opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1], pos_top="97%", range_start=0, range_end=100),
            ],
        )
    )

    bar = (
        Bar()
        .add_xaxis(dates)
        .add_yaxis(
            series_name="Volume",
            y_axis=volumes,
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True, grid_index=1, boundary_gap=False),
            yaxis_opts=opts.AxisOpts(grid_index=1, is_scale=True),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    grid = Grid(init_opts=opts.InitOpts(width="100%"))
    grid.add(kline, grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_top="10%", height="60%"))
    grid.add(bar, grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%", pos_top="75%", height="15%"))

    return grid


# -------------------------
# LSTM Model (Test vs Pred)
# -------------------------

def lstm(stock_code, stock_data):
    # Prepare data
    df = pd.DataFrame(list(stock_data.values('date', 'close')))
    df = df.set_index('date')

    # Safety: need enough points
    if len(df) < 120:
        raise ValueError("Not enough data for LSTM. Please collect a longer time range (>=120 points).")

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Create dataset
    def create_dataset(dataset, time_step=60):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    if len(X_test) < 5:
        raise ValueError("Not enough test samples for evaluation. Please collect a longer time range.")

    # Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64, verbose=0)

    # Predict
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)

    Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Dates aligned to test set
    test_dates = df.index[train_size + time_step + 1:].astype(str).tolist()

    # Chart
    line = Line(init_opts=opts.InitOpts(width="100%"))
    line.add_xaxis(test_dates)
    line.add_yaxis(
        series_name="Actual (Test)",
        y_axis=Y_test_actual.flatten().tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.add_yaxis(
        series_name="Predicted (Test)",
        y_axis=test_predict.flatten().tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(type_="dashed"),
    )

    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"{stock_code} - LSTM Forecast",
            subtitle="Actual vs Predicted",
            pos_top="2%",
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(type_="category", name="Date"),
        yaxis_opts=opts.AxisOpts(name="Price"),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )

    chart_html = line.render_embed()

    # Metrics
    mse = mean_squared_error(Y_test_actual, test_predict)
    mae = mean_absolute_error(Y_test_actual, test_predict)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(Y_test_actual, test_predict))

    # MAPE (avoid divide-by-zero)
    denom = np.where(Y_test_actual == 0, np.nan, Y_test_actual)
    mape = float(np.nanmean(np.abs((Y_test_actual - test_predict) / denom)) * 100)

    metrics = {
        "MAE": f"{mae:.4f}",
        "RMSE": f"{rmse:.4f}",
        "R2": f"{r2:.4f}",
        "MAPE": f"{mape:.2f}%",
    }

    return chart_html, metrics

# -------------------------
# ARIMA Model (Test vs Pred)
# -------------------------

def arima(stock_code, stock_data):
    df = pd.DataFrame(list(stock_data.values('date', 'close')))
    df = df.set_index('date')

    if len(df) < 60:
        raise ValueError("Not enough data for ARIMA. Please collect a longer time range (>=60 points).")

    def check_stationarity(series):
        result = adfuller(series)
        return result[1] <= 0.05  # p-value <= 0.05 => stationary

    d = 0
    if not check_stationarity(df['close']):
        d = 1
        while d < 3 and not check_stationarity(df['close'].diff(d).dropna()):
            d += 1

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    model = auto_arima(
        train['close'],
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        d=d,
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    arima_model = ARIMA(train['close'], order=model.order)
    fitted_model = arima_model.fit()

    fc = fitted_model.get_forecast(steps=len(test))
    pred = fc.predicted_mean
    conf_int = fc.conf_int()

    test_dates = test.index.astype(str).tolist()
    actual = test['close'].values

    # Chart
    line = Line(init_opts=opts.InitOpts(width="100%", height="600px"))
    line.add_xaxis(test_dates)
    line.add_yaxis(
        series_name="Actual (Test)",
        y_axis=actual.tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.add_yaxis(
        series_name="Predicted (Test)",
        y_axis=pred.tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(type_="dashed"),
    )
    line.add_yaxis(
        series_name="Upper CI",
        y_axis=conf_int.iloc[:, 1].tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(type_="dotted"),
        symbol="none",
    )
    line.add_yaxis(
        series_name="Lower CI",
        y_axis=conf_int.iloc[:, 0].tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(type_="dotted"),
        symbol="none",
        areastyle_opts=opts.AreaStyleOpts(opacity=0.08),
    )

    line.set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"{stock_code} - ARIMA Forecast",
            subtitle="Actual vs Predicted)",
            pos_top="2%",
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(type_="category", name="Date"),
        yaxis_opts=opts.AxisOpts(name="Price"),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )

    chart_html = line.render_embed()

    # Metrics
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(actual, pred))

    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs((actual - pred) / denom)) * 100)

    metrics = {
        "MAE": f"{mae:.4f}",
        "RMSE": f"{rmse:.4f}",
        "R2": f"{r2:.4f}",
        "MAPE": f"{mape:.2f}%",
        "Order": f"{model.order}",
        "d": f"{d}",
    }

    return chart_html, metrics

# -------------------------
# Analysis Page
# -------------------------

def gpfx(request):
    stock_code = request.GET.get("code", "")

    gps = Gp.objects.values("code").distinct().count()
    data_sum = Gp.objects.all().count()
    stock_list = Gp.objects.values("code").distinct()

    # No stock selected yet
    if not stock_code:
        return render(
            request,
            "gpfx.html",
            {"gps": gps, "data_sum": data_sum, "stock_list": stock_list},
        )

    stock_data = Gp.objects.filter(code=stock_code).order_by("date")
    if not stock_data.exists():
        return render(
            request,
            "gpfx.html",
            {
                "gps": gps,
                "data_sum": data_sum,
                "stock_list": stock_list,
                "error": "No data found for this stock code.",
            },
        )

    # Build Kline data
    kline_data, dates, volumes = [], [], []
    for gp in stock_data:
        kline_data.append([gp.open, gp.close, gp.low, gp.high])
        dates.append(gp.date.strftime("%Y-%m-%d"))
        volumes.append(gp.volume)

    chart = create_kline_chart(stock_code, kline_data, dates, stock_code, volumes)
    chart_html = chart.render_embed()


    lstm_chart_html, lstm_metrics = lstm(stock_code, stock_data)
    arima_chart_html, arima_metrics = arima(stock_code, stock_data)

    context = {
        "chart": chart_html,
        "stock_code": stock_code,
        "stock_name": stock_code,
        "gps": gps,
        "data_sum": data_sum,
        "stock_list": stock_list,

        # charts
        "lstm_chart": lstm_chart_html,
        "arima_chart": arima_chart_html,

        # metrics (for template display)
        "lstm_metrics": lstm_metrics,
        "arima_metrics": arima_metrics,
    }
    return render(request, "gpfx.html", context)