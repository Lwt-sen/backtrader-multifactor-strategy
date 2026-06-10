import os
import tushare as ts
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 全局配置 --------------------------
TOKEN = "90e5c61c14270a4c4b56884acb7e9358a4b23d3920bc1a46823bf647e777"
PROXY_URL = "http://jiaoch.site"
pro = ts.pro_api(TOKEN)
pro._DataApi__http_url = PROXY_URL
START_DATE = "20220101"  # 回测开始日期
END_DATE = "20250101"  # 回测结束日期

N_SAMPLE = 50  # 股池样本数量（前50只）
N_TOP = 10  # 每月选股前10名

# 技术指标参数
MA_SHORT = 5  # 短期均线
MA_LONG = 20  # 长期均线
ATR_PERIOD = 14 # ATR时间段选择

# 风控参数
MAX_POS = 0.2  # 单只股票最大仓位20%
STOP_LOSS_PCT = 0.05  # 固定止损5%
STOP_PROFIT_PCT = 0.4  # 固定止盈40%
ATR_MULTI = 2  # ATR移动止损倍数
RISK_PCT = 0.04  # 单只股票最大风险占总资金4%

# 回测参数
INITIAL_CAPITAL = 1000000
COMMISSION = 0.0001  # 佣金万1
SLIPPAGE = 0.001  # 滑点千1

# 创建缓存目录
DATA_CACHE_DIR = "factor_cache"
if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)


def get_data_with_cache(api_name, file_name, cache_days, **kwargs):
    """带缓存和重试的数据获取函数"""
    file_path = os.path.join(DATA_CACHE_DIR, file_name)

    # 读取本地缓存
    if os.path.exists(file_path):
        file_mtime = os.path.getmtime(file_path)
        if time.time() - file_mtime < cache_days * 24 * 3600:
            return pd.read_csv(file_path, encoding="utf-8")

    # 下载新数据
    try:
        print(f"正在下载: {file_name}")
        df = pro.query(api_name, **kwargs)
        time.sleep(0.5)  # 避免触发tushare频率限制

        if df is not None and not df.empty:
            df.to_csv(file_path, encoding="utf-8", index=False)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"{file_name}_{api_name} 下载失败，原因: {e}")
        return pd.DataFrame()


def get_all_stocks():
    """获取全市场股票列表，剔除ST和上市不满1年的次新股"""
    df = get_data_with_cache("stock_basic","all_stocks.csv",30,
                            exchange="",list_status="L",fields="ts_code,name,list_date,industry")

    if df.empty:
        return pd.DataFrame()

    # 剔除ST股
    df = df[~df["name"].str.contains("ST", na=False)]
    # 剔除上市不满1年的次新股
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d")
    df = df[df["list_date"] < pd.to_datetime(START_DATE) - pd.Timedelta(days=365)]

    return df.reset_index(drop=True)


def get_monthly_factor_data(stock_list):
    """生成月度因子数据"""
    # 获取pe/pb
    daily_list = [] # 用列表添加数据，再一次性组合成df，节省内存，加速运算
    # 提前60天下载，确保首月能有交易数据
    factor_start = pd.to_datetime(START_DATE) - pd.Timedelta(days=60)
    factor_start_str = factor_start.strftime("%Y%m%d")
    for idx, row in stock_list.iterrows():
        ts_code = row["ts_code"]
        print(f"处理估值 {idx + 1}/{len(stock_list)}: {ts_code}")
        daily = get_data_with_cache("daily_basic",f"{ts_code}_daily_basic.csv",30,
                                    ts_code=ts_code,start_date=factor_start_str,end_date=END_DATE,fields="ts_code,trade_date,pe,pb")

        if daily.empty or daily is None:
            continue

        daily["trade_date"] = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
        daily_list.append(daily)
    all_daily = pd.concat(daily_list, ignore_index=True)
    if all_daily.empty:
        print("没有获取到任何估值数据")
        return pd.DataFrame()

    # 获取roe/tr_yoy
    fina_list = []
    # 提前1年
    fina_start_date = str(int(START_DATE[:4]) - 1) + "0101"
    for idx, row in stock_list.iterrows():
        ts_code = row["ts_code"]
        print(f"处理财报 {idx + 1}/{len(stock_list)}: {ts_code}")

        fina = get_data_with_cache("fina_indicator",f"{ts_code}_fina_indicator.csv",60,
                                   ts_code=ts_code,start_date=fina_start_date,end_date=END_DATE,report_type=1,fields="ts_code,ann_date,end_date,roe,tr_yoy")

        if fina.empty:
            continue

        fina["ann_date"] = pd.to_datetime(fina["ann_date"], format="%Y%m%d")
        fina = fina.drop_duplicates(subset=["ann_date"], keep="last")
        fina_list.append(fina)
    all_fina = pd.concat(fina_list, ignore_index=True)
    if all_fina.empty:
        print("没有获取到任何财报数据")
        return pd.DataFrame()

    # 时间对齐
    merged_list = []
    for ts_code in all_daily["ts_code"].unique():
        stock_daily = all_daily[all_daily["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        stock_fina = all_fina[all_fina["ts_code"] == ts_code].sort_values("ann_date").reset_index(drop=True)

        if stock_fina.empty:
            continue

        stock_merged = pd.merge_asof(
            left=stock_daily,
            right=stock_fina,
            left_on="trade_date",
            right_on="ann_date",
            by="ts_code",
            direction="backward",
            tolerance=pd.Timedelta(days=365)
        )

        stock_merged = stock_merged[stock_merged["ann_date"] <= stock_merged["trade_date"]]
        merged_list.append(stock_merged)
    merged = pd.concat(merged_list, ignore_index=True)
    if merged.empty:
        print("没有匹配到任何有效数据")
        return pd.DataFrame()

    # 月度筛选
    merged["month"] = merged["trade_date"].dt.to_period("M")

    monthly_factor = merged.loc[
        merged.groupby(["ts_code", "month"])["trade_date"].idxmax()
    ].reset_index(drop=True)        # 按代码和月份分组，获取每月最后一天的数据

    monthly_factor = monthly_factor.dropna(subset=["pe", "pb", "roe", "tr_yoy"])
    print(f"包含股票数量: {monthly_factor['ts_code'].nunique()}")
    print(f"因子时间跨度: {monthly_factor['month'].min()} 至 {monthly_factor['month'].max()}")

    return monthly_factor


def factor_scoring(monthly_df, n_top=N_TOP):
    """月度因子z-score打分函数（当月因子用于下月选股）"""
    if monthly_df.empty:
        return {}

    # pe/pb为负向因子
    monthly_df["pe_neg"] = -monthly_df["pe"]
    monthly_df["pb_neg"] = -monthly_df["pb"]

    # z-score标准化
    def z_score(x):
        return (x - x.mean()) / x.std() if x.std() > 0.0 else 0.0

    # 按月分组计算综合得分
    monthly_df["score"] = (
            0.4 * monthly_df.groupby("month")["roe"].transform(z_score) +
            0.25 * monthly_df.groupby("month")["tr_yoy"].transform(z_score) +
            0.2 * monthly_df.groupby("month")["pe_neg"].transform(z_score) +
            0.15 * monthly_df.groupby("month")["pb_neg"].transform(z_score)
    )

    # 强制转换为float类型，过滤空值
    monthly_df["score"] = pd.to_numeric(monthly_df["score"], errors="coerce")
    monthly_df = monthly_df.dropna(subset=["score"])

    # 当月的因子数据用于下个月选股，转换为字典类型，加速运算
    monthly_df["next_month_str"] = (monthly_df["month"] + 1).astype(str)
    top_stocks = monthly_df.groupby("next_month_str").apply(
        lambda x: x.nlargest(n_top, "score")["ts_code"].tolist()
    ).to_dict()

    # 打印所有选股池月份
    for month in sorted(top_stocks.keys()):
        print(f"{month}: {len(top_stocks[month])}只股票")

    return top_stocks


def get_qfq_data(ts_code):
    """获取前复权日线数据，转换为Backtrader格式"""
    df = get_data_with_cache('daily',f'{ts_code}_qfq.csv',30,
                             ts_code=ts_code,start_date=START_DATE,end_date=END_DATE,adj='qfq')

    if df.empty or df is None or 'trade_date' not in df.columns:
        print(f"{ts_code} 无有效数据，跳过")
        return pd.DataFrame()

    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index('trade_date').sort_index()
    # 过滤停牌数据
    df = df[df['vol'] > 0]
    df.rename(columns={'vol': 'volume'}, inplace=True)
    # 取出backtrader所需列
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df


# -------------------------- 多因子+金叉择时策略 --------------------------
class MultiFactorGoldenCrossStrategy(bt.Strategy):
    params = (
        ('ma_short', MA_SHORT),
        ('ma_long', MA_LONG),
        ('atr_period', ATR_PERIOD),
        ('stop_loss_pct', STOP_LOSS_PCT),
        ('stop_profit_pct', STOP_PROFIT_PCT),
        ('monthly_pool', {}),  # 每月股池
    )

    def __init__(self):
        # 为每只股票初始化独立的指标和状态
        self.inds = {}
        self.highest_price = {}
        self.cost_basis = {}

        # 全局交易统计
        self.total_trades = 0
        self.win_trades = 0
        self.lose_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        for data in self.datas:
            ts_code = data._name
            # 均线和金叉指标
            ma_short = bt.indicators.SMA(data.close, period=self.p.ma_short)
            ma_long = bt.indicators.SMA(data.close, period=self.p.ma_long)
            cross = bt.indicators.CrossOver(ma_short, ma_long)
            atr = bt.indicators.ATR(data, period=self.p.atr_period)

            self.inds[ts_code] = {
                'ma_short': ma_short,
                'ma_long': ma_long,
                'cross': cross,
                'atr': atr
            }

            self.highest_price[ts_code] = 0.0
            self.cost_basis[ts_code] = 0.0

        self.order = None
        self.last_month = None  # 记录上一个月份，用于触发调仓
        self.current_targets = []  # 每月股池

    def notify_order(self, order):
        """订单状态通知 + 逐笔交易盈亏计算"""
        if order.status in [order.Completed]:
            ts_code = order.data._name

            if order.isbuy():
                print(f" 买入成交: {ts_code}")
                self.cost_basis[ts_code] = order.executed.price
                self.highest_price[ts_code] = order.executed.price

            elif order.issell():
                sell_price = order.executed.price
                sell_size = abs(order.executed.size)
                cost_price = self.cost_basis.get(ts_code, 0.0)

                # 成本价异常处理
                if cost_price <= 0:
                    print(f"{ts_code} 成本价异常({cost_price:.2f})，跳过盈亏计算")
                    self.cost_basis[ts_code] = 0.0
                    self.highest_price[ts_code] = 0.0
                    return

                # 计算盈亏
                profit_amount = (sell_price - cost_price) * sell_size
                profit_pct = (sell_price - cost_price) / cost_price * 100

                # 更新全局统计
                self.total_trades += 1
                if profit_amount > 0:
                    self.win_trades += 1
                    self.total_profit += profit_amount
                    trade_type = "盈利"
                else:
                    self.lose_trades += 1
                    self.total_loss += abs(profit_amount)
                    trade_type = "亏损"

                print(
                    f" 卖出成交: {ts_code} ")
                print(f"   {trade_type}: {profit_amount:.2f}元 | 收益率: {profit_pct:.2f}%")

                # 重置状态
                self.cost_basis[ts_code] = 0.0
                self.highest_price[ts_code] = 0.0

            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"订单失败: {order.data._name} | 状态: {order.status}")

    def rebalance(self):
        """月度调仓函数
        1. 更新每月股池，买入由每日金叉信号触发
        2. 卖出所有不在股池里的持仓
        """
        dt = self.datas[0].datetime.date(0)
        current_month_str = dt.strftime("%Y-%m")
        print(f"\n===== {current_month_str} 月度调仓 =====")

        # 更新本月股池
        self.current_targets = self.p.monthly_pool.get(current_month_str, [])
        print(f"本月股池: {self.current_targets}")

        if not self.current_targets:
            print("本月无选股数据，清空所有持仓")
            for data in self.datas:
                pos = self.getposition(data)
                if pos:
                    self.close(data)
                    print(f"清仓: {data._name}")
            return

        # 卖出所有不在本月股池的持仓
        sell_count = 0
        for data in self.datas:
            ts_code = data._name
            pos = self.getposition(data)
            if pos and ts_code not in self.current_targets:
                self.close(data)
                print(f"调仓卖出: {ts_code} (不在本月股池)")
                sell_count += 1

        if sell_count == 0:
            print("没有要调仓卖出的股票")

    def next(self):
        """每日数据处理
        1. 月度调仓检查
        2. 处理所有持仓的止盈止损
        3. 遍历本月股池，寻找金叉买入信号
        """
        current_month = self.datas[0].datetime.date(0).month
        if self.last_month is None or current_month != self.last_month:
            self.rebalance()
            self.last_month = current_month

        # 跳过指标未成形阶段
        if len(self.data) < max(self.p.ma_long, self.p.atr_period):
            return

        # 止盈止损逻辑
        for data in self.datas:
            ts_code = data._name
            pos = self.getposition(data)

            if not pos:
                continue

            self.highest_price[ts_code] = max(self.highest_price[ts_code], data.high[0])
            cost = self.cost_basis.get(ts_code, 0.0)
            atr = self.inds[ts_code]['atr'][0]
            low = data.low[0]
            cross = self.inds[ts_code]['cross'][0]

            if atr <= 0:
                atr = data.close[0] * 0.02

            # 固定止损5%
            if low <= cost * (1 - self.p.stop_loss_pct):
                self.close(data)
                print(f" 【固定止损】: {ts_code} ")
                continue

            # ATR移动止损
            if low <= cost - atr * ATR_MULTI:
                self.close(data)
                print(f"ATR移动【止损】: {ts_code} ")
                continue

            # 固定止盈40%
            if self.highest_price[ts_code] >= cost * (1 + self.p.stop_profit_pct):
                self.close(data)
                print(f"【固定止盈】: {ts_code} ")
                continue

            # 盈利10%后启动移动止盈
            if self.highest_price[ts_code] >= cost * 1.1:
                if low <= self.highest_price[ts_code] * (1 - self.p.stop_loss_pct):
                    self.close(data)
                    print(f"【移动止盈】: {ts_code} ")
                    continue

            # 5. 均线死叉卖出
            if cross < 0:
                self.close(data)
                print(f"【死叉卖出】: {ts_code}")
                continue

        # 金叉买入
        if not self.current_targets:
            return

        cash = self.broker.get_cash()

        # 遍历本月股池
        for ts_code in self.current_targets:
            data = next((d for d in self.datas if d._name == ts_code), None)
            if not data:
                continue

            # 已经持仓，跳过
            pos = self.getposition(data)
            if pos:
                continue

            # 金叉信号判断
            cross = self.inds[ts_code]['cross'][0]
            if cross <= 0:
                continue  # 没有金叉，跳过

            # 涨跌停判断
            limit_up = data.open[0] * 1.09
            if data.open[0] >= limit_up:
                print(f"{ts_code} 涨停，跳过买入")
                continue

            atr = self.inds[ts_code]['atr'][0]
            if atr <= 0:
                atr = data.close[0] * 0.02

            # 计算买入数量
            risk_shares = int((cash * RISK_PCT) / (atr * ATR_MULTI) / 100) * 100
            max_shares = int((cash * MAX_POS) / data.open[0] / 100) * 100
            per_stock_cash = cash / len(self.current_targets)
            avg_shares = int(per_stock_cash / data.open[0] / 100) * 100

            buy_shares = min(risk_shares, max_shares, avg_shares)
            if buy_shares >= 100:
                print(f"【金叉买入】: {ts_code} ")
                self.buy(data=data, size=buy_shares, exectype=bt.Order.Market)


# -------------------------- 主回测流程 --------------------------
if __name__ == "__main__":
    # 获取股票池
    print("===== 获取股票池 =====")
    stock_list = get_all_stocks()
    if stock_list.empty:
        print("没有获取到有效股票列表")
        exit()

    stock_list = stock_list.head(N_SAMPLE)
    print(f"筛选后样本股数量: {len(stock_list)}")

    # 生成月度因子数据
    monthly_factor = get_monthly_factor_data(stock_list)
    if monthly_factor.empty:
        print("因子数据生成失败")
        exit()
    monthly_factor.to_csv("monthly_factor_result.csv", encoding="utf-8", index=False)
    print("\n 因子数据已保存到 monthly_factor_result.csv")

    # 月度因子打分，生成选股池
    print("\n===== 筛选月度股池 =====")
    monthly_stock_pool = factor_scoring(monthly_factor)
    print(f"共生成 {len(monthly_stock_pool)} 个月的选股池")

    # 获取所有需要的股票的前复权行情数据
    all_needed_stocks = set()
    for stocks in monthly_stock_pool.values():
        all_needed_stocks.update(stocks)
    all_needed_stocks = list(all_needed_stocks)
    print(f"需要下载行情的股票数量: {len(all_needed_stocks)}")
    data_feeds = {}
    for ts_code in all_needed_stocks:
        df = get_qfq_data(ts_code)
        if df is not None and not df.empty:
            data_feeds[ts_code] = df
            print(f"{ts_code} 行情数据加载完成")
    if not data_feeds:
        print(" 没有获取到任何行情数据")
        exit()

    # 5. 初始化Backtrader回测引擎
    print("\n===== 初始化回测引擎 =====")
    cerebro = bt.Cerebro(stdstats=False)  # bt绘图太混乱，关闭默认的统计指标绘图

    # 添加所有股票数据，全部关闭绘图
    for ts_code, df in data_feeds.items():
        data = bt.feeds.PandasData(
            dataname=df,
            name=ts_code,
            fromdate=pd.to_datetime(START_DATE),
            todate=pd.to_datetime(END_DATE),
            plot=False  # 彻底关闭该股票的所有绘图
        )
        cerebro.adddata(data)

    # 添加策略
    cerebro.addstrategy(MultiFactorGoldenCrossStrategy, monthly_pool=monthly_stock_pool)

    # 设置回测参数
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(SLIPPAGE)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='daily_return', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years, riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # 6. 运行回测
    print(f"\n初始资金: {INITIAL_CAPITAL:,.2f}")
    print("回测开始...\n")
    results = cerebro.run()
    start = results[0]

    # 输出回测结果
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sharpe = start.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    max_drawdown = start.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
    annual_return = start.analyzers.returns.get_analysis().get('rnorm100', 0)

    print("\n" + "=" * 60)
    print("回测结果统计：")
    print(f"最终资金: {final_value:,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"年化收益率: {annual_return:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")

    # 全局交易统计
    print("\n" + "-" * 60)
    print("交易统计")
    print(f"总交易次数: {start.total_trades}")
    print(f"盈利次数: {start.win_trades}")
    print(f"亏损次数: {start.lose_trades}")
    if start.total_trades > 0:
        win_rate = start.win_trades / start.total_trades * 100
        print(f"胜率: {win_rate:.2f}%")
    print(f"总盈利金额: {start.total_profit:.2f}元")
    print(f"总亏损金额: {start.total_loss:.2f}元")
    print(f"净盈利金额: {start.total_profit - start.total_loss:.2f}元")
    print("=" * 60)

    # 绘制收益曲线
    daily_returns = start.analyzers.daily_return.get_analysis()
    returns_df = pd.DataFrame.from_dict(daily_returns, orient='index', columns=['daily_return'])
    returns_df.index = pd.to_datetime(returns_df.index)

    # 计算累计净值
    returns_df['cumulative_return'] = (1 + returns_df['daily_return']).cumprod()
    returns_df['net_value'] = returns_df['cumulative_return'] * INITIAL_CAPITAL

    # 绘制图表
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(returns_df.index, returns_df['net_value'], color='red', linewidth=1, label='策略净值')

    # 添加基准线（初始资金）
    plt.axhline(y=INITIAL_CAPITAL, color='blue', linestyle='--', linewidth=1.5, label='初始资金')

    # 只有当存在最大回撤数据时才标注
    drawdown = start.analyzers.drawdown.get_analysis()
    if 'max' in drawdown and 'datetime' in drawdown['max']:
        max_dd_date = pd.to_datetime(drawdown['max']['datetime'])
        if max_dd_date in returns_df.index:
            max_dd_value = returns_df.loc[max_dd_date, 'net_value']
            plt.scatter(max_dd_date, max_dd_value, color='#d62728', s=100, zorder=5,
                        label=f'最大回撤点 ({max_drawdown:.2f}%)')

    # 图表美化
    plt.title('多因子+金叉择时策略收益曲线', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('账户净值 (元)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 保存高清图片
    plt.savefig('strategy_return.png', dpi=150, bbox_inches='tight')
    print("收益曲线已保存为 strategy_return.png")

    # 显示图表
    plt.show()