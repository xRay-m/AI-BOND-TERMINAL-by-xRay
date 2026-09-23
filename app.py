import os, json, time, re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import deque
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT=Path(__file__).resolve().parent
from config.settings import load_settings, save_settings, ensure_runtime
from core.moex import MOEXClient
from brokers.tinvest_rest import TInvestREST
from brokers.tinvest_stream import TInvestStream
from risk.engine import RiskEngine, RiskLimits
from trading.risk_gate import RiskGate
from trading.reconciliation import reconcile, load_halt
from trading.execution_journal import record as execution_record, read as execution_read
from analytics.market import normalize_orderbook, dom_rows, spread_metrics, liquidity_score, issuer_concentration
from strategy.manager import load_strategies, save_strategies, evaluate_all, evaluate_strategy, install_preset, PRESETS, FIELDS, OPS, add_strategy
from strategy.optimizer import optimize_strategy
from backtesting.strategy_backtest import run_strategy_backtest
from ai.decision_journal import record as journal_record, read as journal_read
from ui.workstation import load as load_workspaces, save as save_workspaces, notify, notifications
from ui.layout import load as load_layouts, save as save_layouts
from ui.hotkeys import render as render_hotkeys
from diagnostics.first_run import diagnostics
from trading.auto_trader import AutoTrader, AUTO_PRESETS
from trading.auto_engine import AutoEngine, PaperBroker
from trading.execution_algorithms import execution_plan
from analytics.bond_quant import add_bond_metrics, yield_curve, relative_value, carry_roll_down, spread_analytics, scenario_analysis
from strategy.quant_strategies import STRATEGIES, ensemble
from ai.model_ensemble import predict as ml_predict
from monitoring.watchdog import check as watchdog_check
from monitoring.kill_switch import KillSwitch
from oms.audit import AuditLog
from tests.production_readiness import run as production_readiness
from tests.failover_recovery import run as failover_recovery
from tests.websocket_recovery_test import run as websocket_recovery_test
from tests.redis_postgres_recovery import run as redis_postgres_recovery
from trading.order_lifecycle import inspect_orders
from realtime.event_bus import emit as event_emit
from ml.pd_pipeline import PDPipeline
from monitoring.metrics import process_metrics, runtime_snapshot, save_metrics
from realtime.redis_cache import ping as redis_ping
from database.service import counts as db_counts
from tests.test_runner import run_all
from tests.integration.sandbox_e2e_runner import run_sandbox_e2e

settings=load_settings(); ensure_runtime(); client=MOEXClient(settings['MOEX_BASE_URL'])
st.set_page_config(page_title='AI BOND TERMINAL V3.8',page_icon='📈',layout='wide',initial_sidebar_state='expanded')
DEFAULT_STATE={'orders':[],'bonds':pd.DataFrame(),'selected':None,'sandbox_e2e':None,'notifications':notifications(),'workspace':settings.get('WORKSPACE','Default'),'kill':False,'stream':None,'stream_state':{},'stream_events':deque(maxlen=500),'dom':{},'tape':[],'chart':pd.DataFrame(),'selected_order':None,'risk_decision':None,'command':'','recon':None,'halt_reason':''}
for k,v in DEFAULT_STATE.items():
    if k not in st.session_state: st.session_state[k]=v

st.markdown('''<style>
:root{--bg:#06101c;--panel:#0d1a2b;--panel2:#12243a;--line:#29445f;--accent:#22d6aa;--blue:#4ea1ff;--danger:#ff6177;--warn:#ffc766;--muted:#91a7c2}
.stApp{background:linear-gradient(135deg,#06101c,#091729 55%,#07111e);color:#eaf2fb}.block-container{padding-top:.55rem;max-width:1800px}
[data-testid="stSidebar"]{background:#06101c;border-right:1px solid #20344c}.metric-card{background:linear-gradient(145deg,#102139,#0b1727);border:1px solid #27415f;border-radius:13px;padding:11px}.metric-label{color:#8fa6c1;font-size:11px;text-transform:uppercase}.metric-value{font-size:23px;font-weight:700}.section{background:#0d1a2c;border:1px solid #223a57;border-radius:14px;padding:14px;margin:8px 0 14px}.statusbar{background:#091727;border:1px solid #233b58;border-radius:12px;padding:8px 10px;margin-bottom:10px}.chip{display:inline-block;padding:4px 8px;border-radius:999px;border:1px solid #2a4565;margin-right:5px;font-size:12px}.ok{color:#20d4a7}.warn{color:#ffcc66}.bad{color:#ff6b7d}.small{font-size:12px;color:#91a7c2}.command{background:#0d1a2c;border:1px solid #315273;border-radius:10px;padding:4px 8px}.hero{background:radial-gradient(circle at 20% 0%,#1b5a72 0,#0d1a2c 45%,#081321 100%);border:1px solid #2f6380;border-radius:18px;padding:20px;box-shadow:0 12px 35px rgba(0,0,0,.25)}.stButton>button{border-radius:10px;border:1px solid #315273}.stTabs [data-baseweb="tab"]{font-weight:700}.dataframe{border-radius:12px}
@media(max-width:900px){.block-container{padding-left:.45rem;padding-right:.45rem}.metric-value{font-size:18px}}
</style>''',unsafe_allow_html=True)

def sf(x,default=np.nan):
    try:return float(x)
    except:return default

def money(obj):
    if not isinstance(obj,dict): return sf(obj,0.0)
    return sf(obj.get('units',0),0)+sf(obj.get('nano',0),0)/1e9

def norm(df):
    if df is None or df.empty:return pd.DataFrame()
    d=df.copy()
    for c in ['LAST','PREVPRICE','FACEVALUE','YIELD','YIELDTOROFFER','VOLUME','COUPONPERCENT']:
        d[c]=pd.to_numeric(d[c],errors='coerce') if c in d else np.nan
    d['price']=d['LAST'].where(d['LAST'].notna(),d['PREVPRICE']); d['nominal']=d['FACEVALUE'].fillna(1000); d['price_pct']=d['price']/d['nominal']*100; d['ytm_pct']=d['YIELD'].where(d['YIELD'].notna(),d['YIELDTOROFFER']); d['coupon_pct']=d['COUPONPERCENT'] if 'COUPONPERCENT' in d else d.get('coupon_pct',pd.Series(np.nan,index=d.index)); d['volume_rub']=d['VOLUME'].fillna(0); d['spread_bps']=pd.to_numeric(d['spread_bps'],errors='coerce') if 'spread_bps' in d else pd.Series(50.0,index=d.index); d['liquidity_score']=pd.to_numeric(d['liquidity_score'],errors='coerce') if 'liquidity_score' in d else np.clip(np.log1p(d['volume_rub'])/np.log(1e9)*100,0,100); d['pd']=d.get('pd',pd.Series(np.nan,index=d.index));
    return d

def show_api(title,payload):
    st.caption(title)
    if isinstance(payload,dict):
        rows=[]
        def walk(x,p=''):
            if isinstance(x,dict):
                for k,v in x.items():walk(v,f'{p}.{k}' if p else k)
            elif isinstance(x,list):
                for i,v in enumerate(x):walk(v,f'{p}[{i}]')
            else:rows.append({'Поле':p,'Значение':x})
        walk(payload); st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        with st.expander('Технический ответ API'): st.code(json.dumps(payload,ensure_ascii=False,indent=2,default=str),language='json')
    else: st.write(payload)

def metric(label,val): st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)

def refresh_market():
    d=norm(client.bonds(0,500)); st.session_state.bonds=d; d.to_csv(ROOT/'runtime/exports/moex_bonds_latest.csv',index=False); return d

def get_sb():
    if not settings.get('TINVEST_TOKEN'): raise RuntimeError('TINVEST_TOKEN не настроен')
    return TInvestREST(token=settings['TINVEST_TOKEN'],account_id=settings.get('TINVEST_ACCOUNT_ID',''),mode=settings.get('TINVEST_MODE','SANDBOX'))

def account_id(): return settings.get('TINVEST_ACCOUNT_ID','')

def topbar():
    halted=load_halt().get('halted',False); live=bool(settings.get('LIVE_ENABLED',False)); ws=st.session_state.workspace
    try: sql=bool(db_counts())
    except: sql=False
    try: redis=bool(redis_ping())
    except: redis=False
    stream=st.session_state.stream_state or {}; sc='🟢' if stream.get('connected') else '⚪'
    st.markdown(f'''<div class="statusbar"><span class="chip">MODE: <b>{settings.get('TINVEST_MODE')}</b></span><span class="chip">{'🔴 LIVE' if live else '🟢 LIVE BLOCKED'}</span><span class="chip">MOEX 🟢</span><span class="chip">WS {sc}</span><span class="chip">Redis {'🟢' if redis else '🟡'}</span><span class="chip">SQL {'🟢' if sql else '🟡'}</span><span class="chip">Risk {'🔴 HALT' if halted else '🟢 READY'}</span><span class="chip">Workspace: {ws}</span><span class="chip">{'🛑 TRADING HALT' if halted or st.session_state.kill else 'Trading Ready'}</span></div>''',unsafe_allow_html=True)

def sidebar():
    st.sidebar.markdown('## 📈 AI BOND TERMINAL'); st.sidebar.caption('V3.8 AUTONOMOUS AUTO TRADING')
    mode=st.sidebar.selectbox('Режим интерфейса',['Beginner','Pro'],index=0 if settings.get('UI_MODE')=='Beginner' else 1)
    if mode!=settings.get('UI_MODE'): settings['UI_MODE']=mode; save_settings(settings); st.rerun()
    ks=KillSwitch(ROOT/'runtime/risk/kill_switch.json'); active=ks.is_active(); st.session_state.kill=active
    if st.sidebar.button('🛑 АКТИВИРОВАТЬ KILL SWITCH' if not active else '🟢 СНЯТЬ KILL SWITCH',use_container_width=True):
        (ks.activate('operator_ui') if not active else ks.release('operator_ui')); st.session_state.kill=ks.is_active(); st.rerun()
    ws=load_workspaces(); names=list(ws) or ['Default']; cur=st.sidebar.selectbox('Рабочее пространство',names,index=names.index(st.session_state.workspace) if st.session_state.workspace in names else 0); st.session_state.workspace=cur
    pages=['🏠 Дашборд','🤖 Автоторговля','💹 Trading Desk','🔎 AI Сканер','📊 Облигация','🧪 T-Invest Sandbox','🎯 Стратегии','💼 Портфель','📋 Заявки','🧠 AI / ML','🧪 Backtest','📰 Новости','🔔 Уведомления','🧾 AI Decision Journal','🖥 Мониторинг','⚙ Настройки','🩺 First-run Diagnostics','🛡 Production Readiness']
    if mode=='Pro': pages+=['🗄 SQL / Redis / Файлы']
    return st.sidebar.radio('Рабочие окна',pages)

def command_palette():
    st.markdown('### ⌘ Command Palette')
    cmd=st.text_input('Быстрая команда',placeholder='например: DOM, BUY, SELL, PORTFOLIO, E2E, RECON',key='command_input')
    actions={'DOM':'💹 Trading Desk','TRADING':'💹 Trading Desk','BUY':'💹 Trading Desk','SELL':'💹 Trading Desk','PORTFOLIO':'💼 Портфель','ORDERS':'📋 Заявки','STRATEGY':'🎯 Стратегии','BACKTEST':'🧪 Backtest','E2E':'🧪 T-Invest Sandbox','RECON':'💹 Trading Desk','RISK':'💹 Trading Desk','JOURNAL':'🧾 AI Decision Journal','DASHBOARD':'🏠 Дашборд'}
    if cmd and cmd.upper().strip() in actions:
        st.info(f"Команда распознана: **{actions[cmd.upper().strip()]}**. Выберите окно в меню слева для выполнения действия.")

def start_stream(inst):
    if not settings.get('TINVEST_TOKEN'): raise RuntimeError('Для WebSocket нужен TINVEST_TOKEN')
    old=st.session_state.stream
    if old: old.stop()
    def on_msg(msg):
        st.session_state.stream_events.append({'ts':datetime.now(timezone.utc).isoformat(),'type':next(iter(msg), 'event'),'data':msg})
        for key in ['orderbook','trade','last_price']:
            if key in msg: st.session_state[f'ws_{key}']=msg[key]
    def on_state(state): st.session_state.stream_state=state
    stream=TInvestStream(settings['TINVEST_TOKEN'],settings.get('TINVEST_MODE','SANDBOX'),on_msg,on_state); stream.start([inst],settings.get('ORDERBOOK_DEPTH',20)); st.session_state.stream=stream; st.session_state.stream_state=stream.snapshot()

def stop_stream():
    if st.session_state.stream: st.session_state.stream.stop(); st.session_state.stream=None; st.session_state.stream_state={'state':'STOPPED','connected':False}

def fetch_candles(inst,interval='CANDLE_INTERVAL_DAY',days=180):
    sb=get_sb(); now=datetime.now(timezone.utc); frm=(now-timedelta(days=days)).isoformat(); r=sb.candles(inst,frm,now.isoformat(),interval,min(2400,days+20)); rows=[]
    for c in r.get('candles',[]): rows.append({'time':c.get('time'),'open':money(c.get('open',0)),'high':money(c.get('high',0)),'low':money(c.get('low',0)),'close':money(c.get('close',0)),'volume':c.get('volume',0)})
    return pd.DataFrame(rows)

def fetch_tape(inst,hours=1):
    sb=get_sb(); now=datetime.now(timezone.utc); r=sb.last_trades(inst,(now-timedelta(hours=hours)).isoformat(),now.isoformat()); rows=[]
    for x in r.get('trades',[]): rows.append({'time':x.get('time'),'price':money(x.get('price',0)),'quantity':x.get('quantity',0),'direction':x.get('direction',x.get('tradeDirection',''))})
    return pd.DataFrame(rows)

def local_order_snapshot():
    rows=execution_read(1000); orders={}
    for e in rows:
        cid=e.get('client_id') or e.get('order_id') or e.get('id');
        if cid: orders[cid]=e
    return list(orders.values())

def risk_gate_for(inst,side,lots,price,pd_prob=0.0,portfolio_value=100000,daily_used=0,issuer_pct=0,spread_pct=0,liq=100,nominal=1000.0):
    # T-Invest quotes bonds in price points: price/100 * nominal is the value of one bond.
    amount=float(lots)*float(price or 0)*float(nominal)/100.0; position_after=min(100,float(amount/portfolio_value*100)) if portfolio_value else 100
    gate=RiskGate(RiskLimits(settings['MAX_ORDER_RUB'],settings['MAX_POSITION_PCT'],settings['MAX_DAILY_RUB'],settings['MAX_PD']))
    return gate.evaluate(side,inst,amount,portfolio_value,position_after,sf(pd_prob,0),daily_used,sf(spread_pct,0),sf(liq,100),sf(issuer_pct,0))

def submit_order(inst,side,lots,price,order_type='ORDER_TYPE_LIMIT',one_click=False,nominal=1000.0):
    if st.session_state.kill or load_halt().get('halted'): raise RuntimeError('Торговля заблокирована Kill Switch/Trading Halt')
    sb=get_sb(); gate=risk_gate_for(inst,side,lots,price,nominal=nominal)
    st.session_state.risk_decision=gate
    if not gate.allowed: notify('RISK','Заявка заблокирована','; '.join(gate.reasons)); raise RuntimeError('Risk Gate: '+', '.join(gate.reasons))
    direction='ORDER_DIRECTION_BUY' if side=='BUY' else 'ORDER_DIRECTION_SELL'
    execution_record('ORDER_RISK_PASS',instrument=inst,side=side,lots=lots,price=price,checks=gate.checks)
    journal_record('order_intent',instrument=inst,side=side,lots=lots,price=price,risk=gate.checks,mode=settings['TINVEST_MODE'])
    r=sb.post_order(inst,int(lots),direction,None if order_type=='ORDER_TYPE_MARKET' else float(price),order_type,'PRICE_TYPE_POINT','TIME_IN_FORCE_DAY',account_id())
    oid=r.get('orderId') or r.get('order_id') or r.get('orderRequestId')
    execution_record('ORDER_SUBMITTED',order_id=oid,instrument=inst,side=side,lots=lots,price=price,response=r)
    journal_record('order_submitted',order_id=oid,instrument=inst,side=side,lots=lots,price=price,response=r)
    notify('TRADE','Заявка отправлена',f'{side} {inst} × {lots}, order_id={oid}')
    return r

def build_broker_snapshot(sb,aid):
    port=sb.portfolio(aid); pos=sb.positions(aid); orders=sb.orders(aid)
    cash=0.0
    for m in (port.get('totalAmountCurrencies',[]) if isinstance(port,dict) else []):
        if str(m.get('currency','RUB')).upper()=='RUB': cash+=money(m.get('amount',m))
    pv=money(port.get('totalAmountPortfolio',0)) if isinstance(port,dict) else 0
    positions=pos.get('securities',[]) if isinstance(pos,dict) else []
    broker_orders=orders.get('orders',[]) if isinstance(orders,dict) else []
    return {'cash':cash,'portfolio':pv,'positions':positions,'orders':broker_orders}


def auto_trader():
    return AutoTrader(ROOT, RiskLimits(settings['MAX_ORDER_RUB'], settings['MAX_POSITION_PCT'], settings['MAX_DAILY_RUB'], settings['MAX_PD']))


def _find_number(obj, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys:
                try:
                    return money(v) if isinstance(v, dict) else float(v)
                except Exception:
                    pass
        for v in obj.values():
            x = _find_number(v, keys)
            if x is not None:
                return x
    elif isinstance(obj, list):
        for v in obj:
            x = _find_number(v, keys)
            if x is not None:
                return x
    return None


def fetch_auto_market(inst, nominal, pd_source='ML model', manual_pd=0.05):
    sb = get_sb()
    prices = sb.get_last_prices([inst])
    last_price = _find_number(prices, {'price', 'lastprice', 'last_price'})
    if last_price is None:
        book = sb.get_order_book(inst, int(settings.get('ORDERBOOK_DEPTH', 20)))
        m = spread_metrics(book)
        last_price = m.get('mid') or m.get('ask') or m.get('bid')
    if last_price is None:
        raise RuntimeError('T-Invest не вернул текущую цену')
    values = sb.market_values([inst], ['INSTRUMENT_VALUE_LAST_PRICE', 'INSTRUMENT_VALUE_YIELD'])
    ytm = None
    def _yield_from_market_values(obj):
        if isinstance(obj, dict):
            typ=str(obj.get('type','')).upper()
            if 'YIELD' in typ and 'value' in obj:
                try: return money(obj['value'])
                except Exception: pass
            for v in obj.values():
                x=_yield_from_market_values(v)
                if x is not None: return x
        elif isinstance(obj, list):
            for v in obj:
                x=_yield_from_market_values(v)
                if x is not None: return x
        return None
    ytm = _yield_from_market_values(values)
    book = sb.get_order_book(inst, int(settings.get('ORDERBOOK_DEPTH', 20)))
    sm = spread_metrics(book); liq = liquidity_score(book)
    price_pct = float(last_price) / float(nominal) * 100.0
    pd_prob = float(manual_pd)
    pd_note = 'ручной ввод'
    if pd_source == 'ML model':
        try:
            model = PDPipeline.load(ROOT / settings.get('PD_MODEL_PATH', 'models/pd_model.joblib'))
            frame = pd.DataFrame([{'ytm_pct': ytm if ytm is not None else np.nan,
                                   'price_pct': price_pct, 'coupon_pct': np.nan,
                                   'volume_rub': 0.0, 'liquidity_score': liq,
                                   'spread_bps': (sm.get('spread_pct') or 0.0) * 100.0}])
            pd_prob = float(model.predict_proba(frame)[0])
            pd_note = 'ML-модель'
        except Exception as e:
            raise RuntimeError(f'Не удалось рассчитать PD ML-моделью: {e}')
    if ytm is None:
        raise RuntimeError('T-Invest не вернул YTM. Для автостратегии нужна YTM.')
    return {'price': float(last_price), 'execution_price': float(sm.get('ask') or last_price),
            'price_pct': price_pct, 'ytm_pct': float(ytm), 'pd': pd_prob,
            'spread_pct': float(sm.get('spread_pct') or 0.0), 'liquidity_score': float(liq),
            'pd_note': pd_note}




def _auto_daily_used(nominal):
    today=datetime.now(timezone.utc).date().isoformat(); total=0.0
    for e in execution_read(2000):
        if e.get('event')!='AUTO_ORDER_SUBMITTED': continue
        if str(e.get('ts',''))[:10] != today: continue
        try: total += float(e.get('lots',0))*float(e.get('price',0))*float(nominal)/100.0
        except Exception: pass
    return total


def _auto_cycle_once(at, pd_source, manual_pd, issuer_pct):
    inst = at.state.instrument_id
    market = fetch_auto_market(inst, at.state.nominal, pd_source, manual_pd)
    status = get_sb().trading_status(inst)
    # T-Invest exposes api_trade_available_flag and limit_order_available_flag in GetTradingStatus.
    trading_allowed = bool(status.get('apiTradeAvailableFlag', status.get('api_trade_available_flag', False))) and bool(status.get('limitOrderAvailableFlag', status.get('limit_order_available_flag', False)))
    portfolio = build_broker_snapshot(get_sb(), account_id())
    daily_used = _auto_daily_used(at.state.nominal)
    result = at.cycle(market, portfolio.get('portfolio', 0.0), daily_used,
                      issuer_pct, market['spread_pct'], market['liquidity_score'],
                      lambda side, lots, price: submit_order(inst, side, lots, price, 'ORDER_TYPE_LIMIT', True, nominal=at.state.nominal),
                      trading_allowed=trading_allowed)
    result['market'] = market; result['broker_status'] = status
    return result


def render_auto_trading():
    st.title('🤖 Автоторговля')
    st.caption('Единое рабочее окно: Auto Scanner → AI Score → Position Sizing → Risk Gate → Execution → OMS → Lifecycle → Exit. Настройки стратегий находятся во вкладке «🎯 Стратегии».')
    engine=AutoEngine(ROOT,RiskLimits())
    paper=PaperBroker(ROOT,float(settings.get('PAPER_START_RUB',100000)))
    tabs=st.tabs(['🔎 Auto Scanner','🎯 Decision Engine','📐 Position Sizing','⚡ Execution','🔗 Order Lifecycle','📒 Journal','🛡 Watchdog / Contours'])
    halted=load_halt().get('halted',False)
    with tabs[0]:
        st.subheader('Auto Scanner')
        c1,c2,c3,c4=st.columns(4)
        with c1: mode=st.selectbox('Контур',['PAPER','SANDBOX','LIVE'],index=['PAPER','SANDBOX','LIVE'].index(settings.get('TINVEST_MODE','SANDBOX')) if settings.get('TINVEST_MODE','SANDBOX') in ['PAPER','SANDBOX','LIVE'] else 0)
        with c2: min_score=st.slider('Минимальный AI Score',0,100,70)
        with c3: max_pd=st.slider('Макс. PD 12м',0.0,0.30,float(settings.get('MAX_PD',.08)),0.01)
        with c4: min_liq=st.slider('Мин. Liquidity',0,100,55)
        cols=['discount_value','carry','roll_down','relative_value','spread_reversion','coupon_event','momentum','mean_reversion']
        selected=st.multiselect('Strategy Ensemble',cols,default=cols[:5],format_func=lambda x:STRATEGIES[x]['name'])
        if st.button('🔎 Сканировать рынок',type='primary'):
            try:
                d=refresh_market(); d=add_bond_metrics(d)
                d['liquidity_score']=np.clip(np.log1p(d.get('volume_rub',pd.Series(0,index=d.index)))/np.log(1e9)*100,0,100)
                d['spread_bps']=d.get('spread_bps',pd.Series(50,index=d.index)).fillna(50)
                p,models=ml_predict(d)
                d['pd']=np.clip(p,0,1) if len(p)==len(d) and np.isfinite(p).any() else d.get('pd',.05)
                d=engine.scan(d,selected)
                d['AI Score']=d['strategy_ensemble_score']
                d['PD 12m']=d['pd']
                d['Action']=d['ensemble_action']
                d['Models']=' + '.join(models) if models else 'ML model not loaded'
                st.session_state.auto_scan=d
                st.session_state.auto_models=models
                emit=event_emit('AUTO_SCAN',{'rows':len(d),'models':models})
            except Exception as e: st.error(f'Auto Scanner: {e}')
        d=st.session_state.get('auto_scan',pd.DataFrame())
        if not d.empty:
            view=d[['SECID','SHORTNAME','price_pct','ytm_pct','PD 12m','liquidity_score','spread_bps','AI Score','Action','auto_eligible']].head(100).copy()
            view.columns=['SECID','Облигация','Цена %','YTM %','PD 12м','Liquidity','Spread bps','AI Score','Action','Risk Eligible']
            st.dataframe(view,use_container_width=True,hide_index=True)
            st.success(f'Найдено кандидатов: {int(d.auto_eligible.sum())} · моделей ML: {len(st.session_state.get("auto_models",[]))}')
            st.download_button('⬇️ Скачать результаты Auto Scanner CSV',d.to_csv(index=False).encode('utf-8-sig'),'auto_scanner.csv','text/csv')
    with tabs[1]:
        st.subheader('AI Decision Engine')
        d=st.session_state.get('auto_scan',pd.DataFrame())
        if d.empty: st.info('Сначала запустите Auto Scanner.')
        else:
            eligible=d[d['auto_eligible']].head(20).copy()
            if eligible.empty: st.warning('Risk-eligible кандидатов нет.')
            else:
                st.dataframe(eligible[['SECID','SHORTNAME','AI Score','ytm_pct','pd','liquidity_score','spread_bps','ensemble_action']],use_container_width=True,hide_index=True)
                inst=st.selectbox('Инструмент для решения',eligible['SECID'].astype(str).tolist())
                row=eligible[eligible.SECID.astype(str)==inst].iloc[0]
                reasons=[]
                for k in selected:
                    reasons.append(f"{STRATEGIES[k]['name']}: {row.get(k+'_score',np.nan):.1f}")
                st.markdown(f"### {row.get('SHORTNAME','')} → **{row['ensemble_action']}**")
                st.write(f"AI Score: **{row['AI Score']:.1f}/100** · PD: **{row['pd']:.2%}** · YTM: **{row['ytm_pct']:.2f}%**")
                st.write(' · '.join(reasons))
                if st.button('💾 Записать Decision в Journal'): journal_record('auto_decision',instrument=inst,action=row['ensemble_action'],score=float(row['AI Score']),pd=float(row['pd']),reasons=reasons); st.success('Decision сохранено')
    with tabs[2]:
        st.subheader('Position Sizing')
        d=st.session_state.get('auto_scan',pd.DataFrame())
        portfolio_value=st.number_input('Стоимость портфеля ₽',1000.,100_000_000.,100000.,1000.)
        current_position=st.number_input('Текущая позиция в инструменте ₽',0.,100_000_000.,0.,1000.)
        price=st.number_input('Цена % номинала',0.01,200.,95.,.01); nominal=st.number_input('Номинал ₽',1.,1_000_000.,1000.,100.)
        pdv=st.number_input('PD 12м',0.,1.,.05,.01); liq=st.slider('Liquidity',0,100,80)
        engine.config.max_position_pct=st.number_input('Макс. позиция %',1.,100.,10.,1.)
        engine.config.risk_per_trade_pct=st.number_input('Риск на сделку %',0.1,10.,1.,.1)
        engine.config.max_order_rub=st.number_input('Max order ₽',100.,10_000_000.,float(settings['MAX_ORDER_RUB']),100.)
        size=engine.position_size(portfolio_value,price,nominal,liq,pdv,current_position)
        m=st.columns(4)
        for c,(l,v) in zip(m,[('Рекомендуемые лоты',size['lots']),('Сумма заявки',f"{size['budget_rub']:,.0f} ₽"),('Risk factor',f"{size['risk_factor']:.2f}"),('Liquidity factor',f"{size['liquidity_factor']:.2f}")]):
            with c: metric(l,v)
        st.info('Размер позиции ограничивается одновременно риском, максимальной долей, ликвидностью, PD и лимитом заявки.')
        if not d.empty:
            candidates=d[d.auto_eligible].copy(); candidates['recommended_lots']=candidates.apply(lambda r:engine.position_size(portfolio_value,r.price_pct,nominal,r.liquidity_score,r.pd,0)['lots'],axis=1)
            st.dataframe(candidates[['SECID','SHORTNAME','AI Score','recommended_lots']].head(30),use_container_width=True,hide_index=True)
    with tabs[3]:
        st.subheader('Execution Engine')
        side=st.selectbox('Side',['BUY','SELL']); algo=st.selectbox('Algorithm',['LIMIT','TWAP']); total_lots=st.number_input('Лоты',1,100000,10); bid=st.number_input('Best Bid %',0.01,200.,94.,.01); ask=st.number_input('Best Ask %',0.01,200.,94.5,.01)
        plan=execution_plan(int(total_lots),side,bid,ask,algo,5); st.dataframe(pd.DataFrame(plan),use_container_width=True,hide_index=True)
        st.caption('TWAP здесь формирует план исполнения. Фактическая отправка в Sandbox/Live выполняется только через OMS и Risk Gate.')
        inst=st.text_input('Instrument ID',value=settings.get('AUTO_TRADING_INSTRUMENT',''),key='auto_exec_inst')
        if st.button('🧪 Выполнить план в PAPER',type='primary') and inst:
            try:
                fills=[]
                for x in plan:
                    if engine.duplicate(inst,side,float(x['price']),int(x['lots'])):
                        execution_record('DUPLICATE_BLOCK',instrument=inst,side=side,lots=int(x['lots']),price=float(x['price']))
                        continue
                    fill=paper.submit(inst,side,int(x['lots']),float(x['price']),1000); fills.append(fill); engine.mark_seen(inst,side,float(x['price']),int(x['lots']),fill.get('order_id','')); execution_record('PAPER_EXECUTION',**fill)
                st.success(f'PAPER исполнено: {len(fills)} частей; duplicate protection активен'); st.dataframe(pd.DataFrame(fills),use_container_width=True,hide_index=True)
            except Exception as e: st.error(str(e))
    with tabs[4]:
        st.subheader('Полный Order Lifecycle')
        j=pd.DataFrame(execution_read(1000))
        if not j.empty:
            st.dataframe(j,use_container_width=True,hide_index=True)
        oid=st.text_input('Broker Order ID',key='auto_lifecycle_oid')
        c1,c2,c3=st.columns(3)
        with c1:
            if st.button('🔄 Refresh State') and oid and settings.get('TINVEST_TOKEN'):
                try: r=get_sb().order_state(oid,account_id=account_id()); execution_record('ORDER_STATUS_REFRESH',order_id=oid,status=r.get('executionReportStatus'),response=r); st.json(r)
                except Exception as e: st.error(str(e))
        with c2:
            if st.button('⏱ Проверить зависшие'):
                stale=inspect_orders(execution_read(1000),int(settings.get('AUTO_ORDER_TIMEOUT_SEC',30))); st.write(f'Зависших заявок: {len(stale)}'); st.dataframe(pd.DataFrame(stale),use_container_width=True,hide_index=True)
        with c3:
            if st.button('🧹 Очистить локальный duplicate cache'): engine.state['seen']={}; engine.save(); st.success('Cache очищен')
        st.markdown('**Состояния:** CREATED → RISK_CHECK → SUBMITTED → PARTIALLY_FILLED → FILLED / CANCEL_PENDING → CANCELLED / REJECTED → RECONCILED')
    with tabs[5]:
        st.subheader('Execution / Decision Journal')
        j=pd.DataFrame(execution_read(1000));
        if not j.empty: st.dataframe(j,use_container_width=True,hide_index=True)
        p=paper.state; st.write('PAPER account'); st.json({'cash':p['cash'],'positions':p['positions'],'trades':len(p['trades'])})
    with tabs[6]:
        st.subheader('Watchdog / Failover / Audit / Trading Contours')
        ws=st.session_state.stream_state or {}; wd=watchdog_check(ws_connected=bool(ws.get('connected')),ws_last_message=ws.get('last_message_at'))
        st.json(wd)
        c=st.columns(3)
        with c[0]: st.metric('PAPER','🟢 READY')
        with c[1]: st.metric('SANDBOX','🟢 READY' if settings.get('TINVEST_TOKEN') else '🟡 TOKEN NEEDED')
        with c[2]: st.metric('LIVE','🔴 ARMED OFF' if not settings.get('AUTO_LIVE_ARMED') else '🟠 ARMED')
        if not wd['ok']:
            st.warning('Watchdog рекомендует FAILOVER/HALT. Реальный перевод в Trading Halt выполняется только при критическом состоянии и после политики подтверждения.')
        st.markdown('**Failover:** при потере WebSocket/Redis/PostgreSQL worker должен остановить новые авто-заявки, сохранить состояние и дождаться восстановления.')

def render_production_readiness():
    st.title('🛡 Production Readiness — контроль перед реальными деньгами')
    st.caption('Никакой тест не включает LIVE автоматически. Sandbox E2E требует TINVEST_TOKEN и TINVEST_E2E_INSTRUMENT_ID.')
    r=production_readiness(ROOT); df=pd.DataFrame([{'check':k,'status':'PASS' if v else 'BLOCKED'} for k,v in r['checks'].items()]); st.dataframe(df,use_container_width=True,hide_index=True)
    c=st.columns(4)
    with c[0]: st.metric('Sandbox E2E',r['sandbox_e2e'])
    with c[1]: st.metric('LIVE gate','BLOCKED' if not r['overall_ready_for_live'] else 'READY')
    with c[2]: st.metric('Audit chain','PASS' if AuditLog().verify().get('ok') else 'FAIL')
    with c[3]: st.metric('Kill Switch','ACTIVE' if KillSwitch(ROOT/'runtime/risk/kill_switch.json').is_active() else 'OFF')
    st.subheader('Recovery tests')
    if st.button('▶ Запустить Failover test'): st.json(failover_recovery(ROOT))
    if st.button('▶ Запустить WebSocket recovery test'): st.json(websocket_recovery_test(ROOT))
    if st.button('▶ Запустить Redis/PostgreSQL recovery test'): st.json(redis_postgres_recovery(ROOT))
    st.subheader('Append-only Audit verification'); st.json(AuditLog().verify())
    st.subheader('Реальный Sandbox E2E')
    if st.button('▶ Выполнить реальный Sandbox E2E'):
        if not settings.get('TINVEST_TOKEN') or not os.getenv('TINVEST_E2E_INSTRUMENT_ID'): st.warning('Нужны TINVEST_TOKEN и TINVEST_E2E_INSTRUMENT_ID.')
        else:
            try: st.json(run_sandbox_e2e(settings['TINVEST_TOKEN'],settings.get('TINVEST_ACCOUNT_ID',''),os.getenv('TINVEST_E2E_INSTRUMENT_ID'),lots=1))
            except Exception as e: st.error(str(e))

def render_portfolio():
    st.title('💼 Portfolio Risk Map / Issuer Concentration')
    if not settings.get('TINVEST_TOKEN') or not account_id():st.warning('Нужен T-Invest token + account ID');return
    try:
        snap=build_broker_snapshot(get_sb(),account_id()); conc=issuer_concentration(snap['positions']);st.dataframe(pd.DataFrame(conc),use_container_width=True,hide_index=True)
        if conc:st.bar_chart(pd.DataFrame(conc).set_index('issuer')['share_pct'])
    except Exception as e:st.error(str(e))

def render_orders():
    st.title('📋 Orders / Trades History');
    j=pd.DataFrame(execution_read(1000));
    if not j.empty:st.dataframe(j,use_container_width=True,hide_index=True)
    if settings.get('TINVEST_TOKEN') and account_id() and st.button('Load broker active orders'):
        try:show_api('Broker Orders',get_sb().orders(account_id()))
        except Exception as e:st.error(str(e))

def render_backtest():
    st.title('🧪 Backtest');
    try:
        df=pd.read_csv(ROOT/'backtest_data'/'bond_backtest_sample.csv');df['date']=pd.to_datetime(df['date']);
        st.dataframe(df.head(50),use_container_width=True,hide_index=True);s=load_strategies()[0]
        if st.button('Run default strategy'):r=run_strategy_backtest(df,s);st.json(r.metrics);st.line_chart(r.equity)
    except Exception as e:st.error(str(e))

def main():
    page=sidebar(); topbar(); render_hotkeys(); command_palette()
    if not settings.get('ONBOARDING_DONE'):
        st.title('🚀 First-run Wizard');st.info('LIVE заблокирован. Начните с SANDBOX.')
        a=st.radio('Режим',['SANDBOX','PAPER'],horizontal=True); token=st.text_input('T-Invest token',type='password');
        if st.button('Запустить диагностику'):st.dataframe(pd.DataFrame(diagnostics(ROOT)),use_container_width=True,hide_index=True)
        if st.button('Сохранить и открыть терминал',type='primary'):
            settings['TINVEST_MODE']=a;settings['TINVEST_TOKEN']=token;settings['ONBOARDING_DONE']=True;save_settings(settings);st.rerun()
        st.stop()
    if page=='🏠 Дашборд':render_dashboard()
    elif page=='🤖 Автоторговля':render_auto_trading()
    elif page=='💹 Trading Desk':render_desk()
    elif page=='🧪 T-Invest Sandbox':render_sandbox()
    elif page=='🎯 Стратегии':render_strategy_lab()
    elif page=='💼 Портфель':render_portfolio()
    elif page=='📋 Заявки':render_orders()
    elif page=='🔔 Уведомления':render_notifications()
    elif page=='🧾 AI Decision Journal':render_journal()
    elif page=='⚙ Настройки':render_settings()
    elif page=='🩺 First-run Diagnostics':render_diagnostics()
    elif page=='🛡 Production Readiness':render_production_readiness()
    elif page=='🧪 Backtest':render_backtest()
    elif page=='🔎 AI Сканер':
        st.title('🔎 AI Scanner');
        if st.button('Refresh MOEX'):refresh_market()
        d=st.session_state.bonds
        if not d.empty:st.dataframe(d[['SECID','SHORTNAME','price','price_pct','ytm_pct','volume_rub','pd']].head(200),use_container_width=True,hide_index=True)
    elif page=='📊 Облигация':
        st.title('📊 Bond Workbench');inst=st.text_input('Instrument');
        if inst:st.info('Используйте Trading Desk для Chart ↔ DOM ↔ Order ↔ Position linkage.')
    elif page=='🧠 AI / ML':st.title('🧠 AI / ML');st.dataframe(pd.DataFrame(execution_read(200)),use_container_width=True,hide_index=True)
    elif page=='📰 Новости':st.title('📰 Новости');st.info('Новости доступны через модуль news/.')
    elif page=='🖥 Мониторинг':st.title('🖥 Monitoring');st.json({'process':process_metrics(),'runtime':runtime_snapshot()})
    elif page=='🧪 Все тесты':
        st.title('🧪 Test Center');
        if st.button('Run tests'):st.session_state.tests_report=run_all();st.json(st.session_state.tests_report)
        if st.session_state.get('tests_report'):show_api('Latest report',st.session_state.tests_report)
    elif page=='🗄 SQL / Redis / Файлы':st.title('🗄 SQL / Redis / Files');st.write({'redis':redis_ping(),'db':db_counts()})

if __name__=='__main__':main()
