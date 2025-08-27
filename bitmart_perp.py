from typing import List, Optional
import ccxt.async_support as ccxt
import asyncio
import pandas as pd
import time
import itertools
from pydantic import BaseModel
from decimal import Decimal, getcontext
from datetime import datetime
from enum import Enum


class UsdtBalance(BaseModel):
    total: float
    free: float
    used: float


class Info(BaseModel):
    success: bool
    message: str


class Order(BaseModel):
    id: str
    pair: str
    type: str
    side: str
    price: float
    size: float
    reduce: bool
    filled: float
    remaining: float
    timestamp: int


class TriggerOrder(BaseModel):
    id: str
    pair: str
    type: str
    side: str
    price: float
    trigger_price: float
    size: float
    reduce: bool
    timestamp: int


class Position(BaseModel):
    pair: str
    side: str
    size: float
    usd_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    liquidation_price: float
    margin_mode: str
    leverage: float
    hedge_mode: bool
    open_timestamp: int
    take_profit_price: float
    stop_loss_price: float


class TransactionType(str, Enum):
    TRANSFER = "Transfer"
    REALIZED_PNL = "Realized PNL"
    FUNDING_FEE = "Funding Fee"
    COMMISSION_FEE = "Commission Fee"
    LIQUIDATION_CLEARANCE = "Liquidation Clearance"


class Transaction(BaseModel):
    symbol: Optional[str]
    type: TransactionType
    amount: float
    asset: str
    account: str
    timestamp: int
    transaction_id: str
    datetime: Optional[datetime] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp:
            self.datetime = datetime.fromtimestamp(self.timestamp / 1000)


class PerpBitmart:
    def __init__(self, public_api=None, secret_api=None, uid=None):
        bitmart_auth_object = {
            "apiKey": public_api,
            "secret": secret_api,
            "uid": uid,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
            },
        }
        getcontext().prec = 10
        if bitmart_auth_object["secret"] == None:
            self._auth = False
            self._session = ccxt.bitmart()
        else:
            self._auth = True
            self._session = ccxt.bitmart(bitmart_auth_object)

    async def load_markets(self):
        self.market = await self._session.load_markets()

    async def close(self):
        await self._session.close()

    def ext_pair_to_pair(self, ext_pair) -> str:
        return f"{ext_pair}:USDT" 
    
    def ext_pair_to_int_pair(self, ext_pair) -> str:
        return ext_pair.split("/")[0] + "USDT"
    
    def int_pair_to_ext_pair(self, int_pair) -> str:
        return int_pair.replace("USDT", "/USDT")

    def pair_to_ext_pair(self, pair) -> str:
        return pair.replace(":USDT", "")

    def get_pair_info(self, ext_pair) -> str:
        pair = self.ext_pair_to_pair(ext_pair)
        if pair in self.market:
            return self.market[pair]
        else:
            return None

    # def amount_to_precision(self, pair: str, amount: float) -> float:
    #     contract_size = (self.get_pair_info(pair))["contractSize"]
    #     amount = amount / contract_size
    #     pair = self.ext_pair_to_pair(pair)
    #     try:
    #         return self._session.amount_to_precision(pair, amount)
    #     except Exception as e:
    #         return 0

    def price_to_precision(self, pair: str, price: float) -> float:
        pair = self.ext_pair_to_pair(pair)
        return self._session.price_to_precision(pair, price)

    async def get_last_ohlcv(self, pair, timeframe, limit=1000) -> pd.DataFrame:
        pair = self.ext_pair_to_pair(pair)
        bitmart_limit = 500
        ts_dict = {
            "1m": 1 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - ((limit) * ts_dict[timeframe])
        current_ts = start_ts
        tasks = []
        while current_ts < end_ts:
            req_end_ts = min(current_ts + (bitmart_limit * ts_dict[timeframe]), end_ts)
            tasks.append(
                self._session.fetch_ohlcv(
                    pair,
                    timeframe,
                    params={
                        "start_time": str(int(current_ts / 1000)),
                        "end_time": str(int(req_end_ts / 1000)),
                    },
                )
            )
            current_ts += (bitmart_limit * ts_dict[timeframe]) + 1
        ohlcv_unpack = await asyncio.gather(*tasks)
        ohlcv_list = list(itertools.chain.from_iterable(ohlcv_unpack))
        df = pd.DataFrame(
            ohlcv_list, columns=["date", "open", "high", "low", "close", "volume"]
        )
        df = df.set_index(df["date"])
        df.index = pd.to_datetime(df.index, unit="ms")
        df = df.sort_index()
        del df["date"]
        return df

    async def get_balance(self) -> UsdtBalance:
        resp = await self._session.fetch_balance(params={"defaultType": "swap"})
        resp_data = resp["info"]["data"]
        usdt_data = [r for r in resp_data if r["currency"] == "USDT"][0]
        return UsdtBalance(
            total=usdt_data["equity"],
            free=usdt_data["available_balance"],
            used=usdt_data["position_deposit"],
        )

    async def set_margin_mode_and_leverage(self, pair, margin_mode, leverage):
        if margin_mode not in ["cross", "isolated"]:
            raise Exception("Margin mode must be either 'cross' or 'isolated'")
        pair = self.ext_pair_to_pair(pair)
        try:
            await self._session.set_leverage(
                leverage,
                pair,
                params={
                    "open_type": margin_mode,
                    "marginMode": margin_mode,
                },
            )
        except Exception as e:
            raise e

        return Info(
            success=True,
            message=f"Margin mode and leverage set to {margin_mode} and {leverage}x",
        )

    async def get_open_positions(self) -> List[Position]:
        resp = await self._session.fetch_positions()
        return_positions = []
        for position in resp:
            if position["contracts"] == 0:
                continue
            liquidation_price = 0
            take_profit_price = 0
            stop_loss_price = 0
            hedge_mode = False
            if position["liquidationPrice"]:
                liquidation_price = position["liquidationPrice"]
            if position["takeProfitPrice"]:
                take_profit_price = position["takeProfitPrice"]
            if position["stopLossPrice"]:
                stop_loss_price = position["stopLossPrice"]
            if position["hedged"]:
                hedge_mode = True
            return_positions.append(
                Position(
                    pair=self.pair_to_ext_pair(position["symbol"]),
                    side=position["info"]["position_side"],
                    size=Decimal(position["contracts"])
                    * Decimal(position["contractSize"]),
                    usd_size=round(
                        position["notional"],
                        2,
                    ),
                    entry_price=position["entryPrice"],
                    current_price=Decimal(position["markPrice"]),
                    unrealized_pnl=position["info"]["unrealized_pnl"],
                    realized_pnl=position["info"]["realized_value"],
                    liquidation_price=liquidation_price,
                    leverage=position["leverage"],
                    margin_mode=position["info"]["open_type"],
                    hedge_mode=hedge_mode,
                    open_timestamp=position["info"]["open_timestamp"],
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                )
            )
        return return_positions

    async def place_order(
        self,
        pair,
        side,
        price,
        size,
        type="limit",
        reduce=False,
        margin_mode="cross",
        leverage=1,
        error=True,
    ) -> Order:
        try:
            contract_size = (self.get_pair_info(pair))["contractSize"]
            pair = self.ext_pair_to_pair(pair)
            size = Decimal(size) / Decimal(contract_size)
            # trade_side = "Open" if reduce is False else "Close"
            resp = await self._session.create_order(
                symbol=pair,
                type=type,
                side=side,
                amount=self._session.amount_to_precision(pair, size),
                price=price,
                params={
                    "reduceOnly": reduce,
                    "marginMode": margin_mode,
                    "leverage": leverage,
                },
            )
            order_id = resp["id"]
            pair = self.pair_to_ext_pair(resp["symbol"])
            order = await self.get_order_by_id(order_id, pair)
            return order
        except Exception as e:
            if error:
                raise e
            else:
                print(e)
                return None


    async def get_order_by_id(self, order_id, pair) -> Order:
        contract_size = (self.get_pair_info(pair))["contractSize"]
        pair = self.ext_pair_to_pair(pair)
        resp = await self._session.fetch_order(order_id, pair)
        reduce = False
        if resp["info"]["side"] in [2, 3]:
            reduce = True
        return Order(
            id=resp["id"],
            pair=self.pair_to_ext_pair(resp["symbol"]),
            type=resp["type"],
            side=resp["side"],
            price=resp["price"],
            size=Decimal(resp["amount"]) * Decimal(contract_size),
            reduce=reduce,
            filled=Decimal(resp["filled"]) * Decimal(contract_size),
            remaining=Decimal(resp["remaining"]) * Decimal(contract_size),
            timestamp=resp["timestamp"],
        )

    async def cancel_orders(self, pair, ids=[]):
        try:
            pair = self.ext_pair_to_pair(pair)
            resp = await self._session.cancel_orders(
                ids=ids,
                symbol=pair,
            )
            return Info(success=True, message=f"{len(resp)} Orders cancelled")
        except Exception as e:
            return Info(success=False, message="Error or no orders to cancel")

    async def cancel_trigger_orders(self, pair, ids=[]):
        try:
            pair = self.ext_pair_to_pair(pair)
            resp = await self._session.cancel_orders(
                ids=ids, symbol=pair, params={"stop": True}
            )
            return Info(success=True, message=f"{len(resp)} Trigger Orders cancelled")
        except Exception as e:
            return Info(success=False, message="Error or no orders to cancel")
        
    async def get_trades(self, pair):
        pair = self.ext_pair_to_int_pair(pair)
        resp = await self._session.private_get_contract_private_trades({"symbol": pair, "account": "futures"})
        return resp
    
    async def get_transactions(
        self, 
        pair: Optional[str] = None,
        flow_type: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page_size: int = 100
    ) -> List[Transaction]:
        """
        Get transaction history with pagination support
        
        Args:
            pair: pair of the contract (e.g. "BTC/USDT")
            flow_type: Type of transaction (0=All, 1=Transfer, 2=Realized PNL, 3=Funding Fee, 4=Commission Fee, 5=Liquidation)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            page_size: Number of records per page (max 1000, default 100)
            
        Returns:
            List of Transaction objects
        """
        all_transactions = []
        has_more = True
        current_end_time = end_time
        
        while has_more:
            params = {
                "account": "futures",
                "page_size": min(page_size, 1000)
            }
            
            if pair is not None:
                params["symbol"] = self.ext_pair_to_int_pair(pair)
            
            if flow_type is not None:
                params["flow_type"] = flow_type
                
            if start_time is not None:
                params["start_time"] = start_time
                
            if current_end_time is not None:
                params["end_time"] = current_end_time
            
            try:
                resp = await self._session.private_get_contract_private_transaction_history(params)
                
                transactions_data = resp.get("data", [])
                
                if not transactions_data:
                    has_more = False
                    break
                
                for tx in transactions_data:
                    transaction = Transaction(
                        symbol=self.int_pair_to_ext_pair(tx.get("symbol", "")) or None,
                        type=TransactionType(tx["type"]),
                        amount=float(tx["amount"]),
                        asset=tx["asset"],
                        account=tx["account"],
                        timestamp=int(tx["time"]),
                        transaction_id=tx["tran_id"]
                    )
                    all_transactions.append(transaction)
                
                # Check if we got less than page_size records
                if len(transactions_data) < page_size:
                    has_more = False
                else:
                    # Use the timestamp of the last transaction for next page
                    last_timestamp = int(transactions_data[-1]["time"])
                    current_end_time = last_timestamp - 1
                    
            except Exception as e:
                raise e
        
        return all_transactions