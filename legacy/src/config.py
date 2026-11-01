import torch

CONFIG = {
    # Local Directory
    # 'data_dir': 'db',
    # 'fama_french_dir': 'factors',
    # 'ckpt_dir': 'artifacts/checkpoints',

    # Modal Directory
    'data_dir': '/mnt/data/db',
    'fama_french_dir': '/mnt/data/factors',
    'ckpt_dir': '/mnt/data/checkpoints',
    # S&P500 Tickers List
    'tickers': [
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALLE', 'ALL', 'ALGN', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP', 'APTV', 'ARE', 'ARES', 'ATO', 'ATR', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BAC', 'BALL', 'BA', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG', 'BIIB', 'BKNG', 'BLDR', 'BLK', 'BKR', 'BMY', 'BNY', 'BRK.B', 'BR', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAH', 'CARR', 'CASY', 'CAT', 'CB', 'CBOE', 'CBRE', 'CDNS', 'CDW', 'CEG', 'CFG', 'CF', 'CHD', 'CHRW', 'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COHR', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPT', 'CPRT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EBAY', 'ECHO', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EME', 'EMR', 'ENPH', 'EOG', 'EQT', 'EQIX', 'ERIE', 'ESS', 'ES', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'FAST', 'FCX', 'FDX', 'FDXF', 'FE', 'FERG', 'FFIV', 'FICO', 'FIS', 'FISV', 'FITB', 'FIX', 'FLEX', 'FLS', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'F', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HD', 'HCA', 'HCI', 'HIG', 'HLT', 'HONA', 'HON', 'HOOD', 'HPQ', 'HPE', 'HRL', 'HST', 'HSIC', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IFF', 'IEX', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'KEY', 'KEYS', 'KHC', 'KDP', 'KIM', 'KLAC', 'KMB', 'KMI', 'KKR', 'KO', 'KR', 'KVUE', 'LH', 'LHX', 'LIN', 'LITE', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'L', 'LDOS', 'LEN', 'LII', 'LYB', 'LYV', 'LULU', 'LUV', 'LVS', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST', 'MO', 'MOS', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MRTN', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NI', 'NFLX', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYX', 'PCAR', 'PCG', 'PFG', 'PFE', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PPG', 'PPL', 'PRU', 'PSA', 'PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'RDDT', 'REG', 'REGN', 'RF', 'RJF', 'RMD', 'ROL', 'ROK', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBUX', 'SBAC', 'SCHW', 'SHW', 'SJM', 'SNA', 'SNPS', 'SNDK', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STLD', 'STT', 'STE', 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TROW', 'TRV', 'TRMB', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VICI', 'VLO', 'VMC', 'VMRK', 'VRSK', 'VRSN', 'VRT', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
    ],
    'force_start_date': '2020-01-01',
    'seq_len': 60,
    'pred_horizon': 5,
    'beta_window': 126,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}