"""Download Fama-French 5 factors + RF (Ken French Data Library) and convert to."""

import _bootstrap  # noqa: F401

import io
import urllib.request
import zipfile

import pandas as pd

from as_fin.config import load_config
from as_fin.utils import get_logger

log = get_logger("fetch_ff")

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")


def _download_csv(url: str) -> str:
    raw = urllib.request.urlopen(url, timeout=30).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    return z.read(name).decode("latin1")


def _parse_monthly(text: str) -> pd.DataFrame:
    """Extract the monthly block (rows keyed by a 6-digit YYYYMM)."""
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 7 and cells[0].isdigit() and len(cells[0]) == 6:
            try:
                ym = cells[0]
                vals = [float(x) for x in cells[1:7]]
            except ValueError:
                continue
            rows.append([ym] + vals)
    df = pd.DataFrame(rows, columns=["ym", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    df["year"] = df["ym"].str[:4].astype(int)
    df["month"] = df["ym"].str[4:6].astype(int)
    df["quarter"] = (df["month"] - 1) // 3 + 1
    # Percent -> decimal.
    for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        df[c] = df[c] / 100.0
    return df


def _compound_to_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compound the three monthly returns of each calendar quarter: prod(1+r)-1."""
    facs = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    g = monthly.groupby(["year", "quarter"])
    # Require all 3 months present so a partial quarter is not under-counted.
    full = g["month"].transform("size") == 3
    m = monthly[full]
    q = (m.groupby(["year", "quarter"])[facs]
           .apply(lambda x: (1 + x).prod() - 1)
           .reset_index())
    q["qid"] = q["year"] * 4 + (q["quarter"] - 1)
    return q[["qid", "year", "quarter"] + facs].sort_values("qid").reset_index(drop=True)


def main():
    cfg = load_config()
    out_path = cfg.path("data_raw") / "ff_factors.parquet"
    log.info("Downloading FF5 factors from Ken French Data Library ...")
    text = _download_csv(FF5_URL)
    monthly = _parse_monthly(text)
    quarterly = _compound_to_quarterly(monthly)
    quarterly.to_parquet(out_path, index=False)
    log.info("Wrote %s: %d quarters, %s..%s (qid %d..%d)", out_path.name,
             len(quarterly), int(quarterly["year"].min()), int(quarterly["year"].max()),
             int(quarterly["qid"].min()), int(quarterly["qid"].max()))
    log.info("columns: %s", [c for c in quarterly.columns])
    print(quarterly.tail(4).to_string(index=False))


if __name__ == "__main__":
    main()
