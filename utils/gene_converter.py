"""
Gene-name conversion using a static HGNC dump.

The class downloads (once, on first use) the HGNC complete-set TSV and serves
human gene-identifier translations across four naming systems:

    "symbol"   - HGNC approved symbols (e.g. "TP53"); also resolves prev_symbol
                 and alias_symbol with ambiguity handling
    "ensembl"  - Ensembl gene IDs (e.g. "ENSG00000141510"); version suffixes
                 like ".15" are stripped before lookup
    "entrez"   - NCBI Entrez gene IDs (e.g. "7157")
    "hgnc"     - HGNC IDs (e.g. "HGNC:11998")

Example:
    conv = GeneNameConverter.download(dest_dir="/data/benchmark/data/gene_maps")
    ensg = conv.convert(["TP53", "MYC", "BRCA1"], from_="symbol", to="ensembl")
    new_adata = conv.translate_anndata(adata, from_="symbol", to="ensembl")
"""

from __future__ import annotations

import datetime as _dt
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

HGNC_URL = (
    "ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
    "tsv/tsv/hgnc_complete_set.txt"
)

NameSystem = Literal["symbol", "ensembl", "entrez", "hgnc"]
MissingPolicy = Literal["none", "drop", "passthrough", "raise"]


class GeneNameConverter:
    """Static-source human gene-name converter backed by HGNC complete_set."""

    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)
        self._df: pd.DataFrame | None = None
        self._build_lookups()

    @classmethod
    def download(
        cls,
        dest_dir: str | Path = ".",
        url: str = HGNC_URL,
        force: bool = False,
    ) -> "GeneNameConverter":
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "hgnc_complete_set.txt"
        if force or not dest.exists():
            logger.info("Downloading HGNC complete set: %s -> %s", url, dest)
            urllib.request.urlretrieve(url, dest)
        return cls(dest)

    def _build_lookups(self) -> None:
        # na_filter=False so empty cells become "" not NaN; lookups treat "" as absent.
        df = pd.read_csv(self.source_path, sep="\t", dtype=str, na_filter=False)
        df = df[df["status"] == "Approved"].reset_index(drop=True)
        self._df = df

        mtime = _dt.datetime.fromtimestamp(self.source_path.stat().st_mtime)
        logger.info(
            "Loaded HGNC dump %s (mtime=%s, %d approved records)",
            self.source_path, mtime.isoformat(timespec="seconds"), len(df),
        )

        self._by_hgnc: dict[str, dict[str, str]] = {
            r["hgnc_id"]: r for r in df.to_dict("records")
        }
        self._symbol_to_hgnc: dict[str, str] = {
            r["symbol"].upper(): r["hgnc_id"] for r in self._by_hgnc.values()
        }
        self._ensembl_to_hgnc: dict[str, str] = {
            r["ensembl_gene_id"]: r["hgnc_id"]
            for r in self._by_hgnc.values()
            if r.get("ensembl_gene_id")
        }
        self._entrez_to_hgnc: dict[str, str] = {
            r["entrez_id"]: r["hgnc_id"]
            for r in self._by_hgnc.values()
            if r.get("entrez_id")
        }

        self._prev_to_hgnc: dict[str, list[str]] = defaultdict(list)
        self._alias_to_hgnc: dict[str, list[str]] = defaultdict(list)
        for r in self._by_hgnc.values():
            for prev in (r.get("prev_symbol") or "").split("|"):
                prev = prev.strip().upper()
                if prev:
                    self._prev_to_hgnc[prev].append(r["hgnc_id"])
            for alias in (r.get("alias_symbol") or "").split("|"):
                alias = alias.strip().upper()
                if alias:
                    self._alias_to_hgnc[alias].append(r["hgnc_id"])

    def _resolve_to_hgnc(self, name: str, from_: NameSystem) -> str | None:
        if not name:
            return None
        if from_ == "symbol":
            key = name.upper()
            if key in self._symbol_to_hgnc:
                return self._symbol_to_hgnc[key]
            for index, kind in (
                (self._prev_to_hgnc, "prev_symbol"),
                (self._alias_to_hgnc, "alias"),
            ):
                hits = index.get(key, [])
                if len(hits) == 1:
                    return hits[0]
                if len(hits) > 1:
                    logger.debug(
                        "Symbol %r ambiguous via %s: %s", name, kind, hits
                    )
                    return None
            return None
        if from_ == "ensembl":
            return self._ensembl_to_hgnc.get(name.split(".")[0])
        if from_ == "entrez":
            key = str(name).strip()
            # pandas often hands back floats like "7157.0" from int columns with NaN
            if key.endswith(".0") and key[:-2].isdigit():
                key = key[:-2]
            return self._entrez_to_hgnc.get(key)
        if from_ == "hgnc":
            return name if name in self._by_hgnc else None
        raise ValueError(f"Unknown source naming system: {from_!r}")

    def _hgnc_to(self, hgnc_id: str | None, to: NameSystem) -> str | None:
        if hgnc_id is None:
            return None
        rec = self._by_hgnc.get(hgnc_id)
        if rec is None:
            return None
        if to == "symbol":
            return rec["symbol"] or None
        if to == "ensembl":
            return rec.get("ensembl_gene_id") or None
        if to == "entrez":
            return rec.get("entrez_id") or None
        if to == "hgnc":
            return hgnc_id
        raise ValueError(f"Unknown target naming system: {to!r}")

    def convert(
        self,
        names: Iterable[str],
        from_: NameSystem,
        to: NameSystem,
        missing: MissingPolicy = "none",
    ) -> list[str | None]:
        """Convert a sequence of gene names from one system to another.

        missing:
            "none"        - emit None for unmapped (positional alignment kept)
            "drop"        - omit unmapped from output (length may shrink)
            "passthrough" - emit the original input unchanged
            "raise"       - raise KeyError on first miss
        """
        out: list[str | None] = []
        for name in names:
            hgnc = self._resolve_to_hgnc(name, from_)
            target = self._hgnc_to(hgnc, to)
            if target is None:
                if missing == "raise":
                    raise KeyError(f"No mapping for {name!r} ({from_} -> {to})")
                if missing == "drop":
                    continue
                if missing == "passthrough":
                    out.append(name)
                else:
                    out.append(None)
            else:
                out.append(target)
        return out

    def convert_one(
        self,
        name: str,
        from_: NameSystem,
        to: NameSystem,
        missing: MissingPolicy = "none",
    ) -> str | None:
        return self.convert([name], from_=from_, to=to, missing=missing)[0]

    def coverage(
        self, names: Iterable[str], from_: NameSystem, to: NameSystem
    ) -> dict:
        names = list(names)
        results = self.convert(names, from_=from_, to=to, missing="none")
        n = len(names)
        n_mapped = sum(1 for r in results if r is not None)
        return {
            "n_input": n,
            "n_mapped": n_mapped,
            "n_missing": n - n_mapped,
            "coverage": n_mapped / n if n else 0.0,
        }

    def translate_anndata(
        self,
        adata,
        from_: NameSystem,
        to: NameSystem,
        drop_unmapped: bool = True,
        deduplicate: Literal["first", "sum", "error"] = "first",
    ):
        """Return a new AnnData with var_names translated to `to` system.

        deduplicate: how to handle several input genes mapping to the same target
            "first" - keep the first occurrence, drop duplicates
            "sum"   - sum count columns of duplicates (requires sparse/dense X)
            "error" - raise if duplicates are produced
        """
        new_names = self.convert(
            list(adata.var_names), from_=from_, to=to, missing="none"
        )
        if drop_unmapped:
            keep = [i for i, n in enumerate(new_names) if n is not None]
            adata = adata[:, keep].copy()
            new_names = [new_names[i] for i in keep]
        else:
            new_names = [
                n if n is not None else orig
                for n, orig in zip(new_names, adata.var_names)
            ]
            adata = adata.copy()

        adata.var_names = pd.Index(new_names)

        if not adata.var_names.is_unique:
            if deduplicate == "error":
                raise ValueError(
                    "Translation produced duplicate var_names; "
                    "set deduplicate='first' or 'sum'"
                )
            if deduplicate == "first":
                # np.unique returns *sorted* unique values; sort the indices to
                # preserve original column order instead.
                _, first_pos = np.unique(
                    adata.var_names.to_numpy(), return_index=True
                )
                first_pos.sort()
                adata = adata[:, first_pos].copy()
            elif deduplicate == "sum":
                groups = pd.Index(adata.var_names)
                uniq = groups.unique()
                col_idx = {g: i for i, g in enumerate(uniq)}
                X = adata.X
                if sparse.issparse(X):
                    coo = X.tocoo()
                    new_cols = np.array(
                        [col_idx[groups[c]] for c in coo.col]
                    )
                    summed = sparse.coo_matrix(
                        (coo.data, (coo.row, new_cols)),
                        shape=(adata.n_obs, len(uniq)),
                    ).tocsr()
                else:
                    summed = np.zeros((adata.n_obs, len(uniq)), dtype=X.dtype)
                    for j, name in enumerate(adata.var_names):
                        summed[:, col_idx[name]] += X[:, j]
                import anndata as ad

                # Shallow-copying obsm/obsp/uns is safe here: `adata` is the
                # function-local copy we made earlier and goes out of scope on
                # return, so no aliasing back to the caller's input.
                adata = ad.AnnData(
                    X=summed,
                    obs=adata.obs.copy(),
                    var=pd.DataFrame(index=uniq),
                    obsm=dict(adata.obsm),
                    obsp=dict(adata.obsp),
                    uns=dict(adata.uns),
                )

        return adata


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Quick HGNC-converter sanity check.")
    parser.add_argument("--dest", default="/data/benchmark/data/gene_maps")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    conv = GeneNameConverter.download(dest_dir=args.dest, force=args.force)
    samples = ["TP53", "MYC", "BRCA1", "AL627309.1", "FAM41C", "HLA-DRB1"]
    print("symbol -> ensembl:")
    for s in samples:
        print(f"  {s:15s} -> {conv.convert_one(s, 'symbol', 'ensembl')}")
    print("ensembl -> symbol:")
    for e in ["ENSG00000141510", "ENSG00000136997", "ENSG00000141510.15"]:
        print(f"  {e:25s} -> {conv.convert_one(e, 'ensembl', 'symbol')}")
    print("coverage on a fake list:")
    print(conv.coverage(samples + ["NOT_A_GENE"], "symbol", "ensembl"))
