import time
import anndata
import numpy as np
import pandas as pd
import functools
from datetime import datetime
from scipy import sparse

class StructureWatcher:
    def __init__(self, adata, func_name="Unknown Function"):
        self.adata = adata
        self.func_name = func_name
        # Capture start state
        self.start_time = time.time()
        self.start_state = self._snapshot(adata)

    def _get_type_desc(self, obj):
        """Helper to get a clean description of an object's type and shape."""
        shape_str = ""
        if hasattr(obj, "shape"):
            shape_str = f"{obj.shape[0]}x{obj.shape[1]}" if len(obj.shape) == 2 else str(obj.shape)
        
        if sparse.issparse(obj):
            return f"(sparse matrix, {shape_str})"
        elif isinstance(obj, np.ndarray):
            return f"(array, {shape_str})"
        elif isinstance(obj, pd.DataFrame):
            return f"(dataframe, {shape_str})"
        elif isinstance(obj, dict):
            return "(dictionary)"
        else:
            return f"({type(obj).__name__})"

    def _snapshot(self, adata):
        """Captures keys, types, and shapes for comparison."""
        snapshot = {
            "shape": adata.shape,
            "obs": set(adata.obs.columns),
            "var": set(adata.var.columns),
            "uns": set(adata.uns.keys()),
            # For complex slots, store a dict of {key: description_string}
            "obsm": {k: self._get_type_desc(v) for k, v in adata.obsm.items()},
            "varm": {k: self._get_type_desc(v) for k, v in adata.varm.items()},
            "layers": {k: self._get_type_desc(v) for k, v in adata.layers.items()},
            "obsp": {k: self._get_type_desc(v) for k, v in adata.obsp.items()},
        }
        return snapshot

    def finish(self):
        """Compares states and prints the formatted report."""
        end_time = time.time()
        duration = round(end_time - self.start_time, 4)
        end_state = self._snapshot(self.adata)
        
        self._print_report(self.start_state, end_state, duration)
        self._log_to_uns(self.start_state, end_state, duration)

    def _print_report(self, start, end, duration):
        print("\n[STRUCTURE CHANGE DETECTED]")
        print("-" * 57)
        print(f"Function: {self.func_name}")
        print(f"Duration: {duration}s\n")

        # 1. DIMENSIONS
        if start['shape'] == end['shape']:
            print(f"1. DIMENSIONS: Unchanged {start['shape']}")
        else:
            print(f"1. DIMENSIONS: CHANGED {start['shape']} -> {end['shape']}")

        # 2. UNS
        self._print_section_diff("2. UNS (Unstructured Data)", start['uns'], end['uns'], is_uns=True)

        # 3. OBSP
        self._print_dict_diff("3. OBSP (Pairwise)", start['obsp'], end['obsp'])

        # 4. OBSM
        self._print_dict_diff("4. OBSM (Multi-dim Observation)", start['obsm'], end['obsm'])
        
        # 5. LAYERS (Optional, but good to have)
        self._print_dict_diff("5. LAYERS (Matrices)", start['layers'], end['layers'])

        print("-" * 57)

    def _print_section_diff(self, title, start_set, end_set, is_uns=False):
        added = end_set - start_set
        # You could also track removed items here
        
        print(f"\n{title}:")
        if not added:
            print("   (No changes)")
            return

        for key in sorted(added):
            val = self.adata.uns[key]
            type_desc = self._get_type_desc(val)
            print(f"   (+) Added '{key}' {type_desc}")
            
            # SPECIAL LOGIC: If it's a dict (like 'neighbors'), try to show 'params'
            if is_uns and isinstance(val, dict) and 'params' in val:
                print(f"       └─ params: {val['params']}")

    def _print_dict_diff(self, title, start_dict, end_dict):
        """Compare dictionaries of {key: description}."""
        start_keys = set(start_dict.keys())
        end_keys = set(end_dict.keys())
        added = end_keys - start_keys
        
        print(f"\n{title}:")
        if not added:
            print("   (No changes)")
            return

        for key in sorted(added):
            desc = end_dict[key]
            print(f"   (+) Added '{key}' {desc}")

    def _log_to_uns(self, start, end, duration):
        """Save a short summary to adata.uns for permanence."""
        if 'history_log' not in self.adata.uns:
            self.adata.uns['history_log'] = []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "function": self.func_name,
            "duration": duration,
            "shape": str(end['shape'])
        }
        self.adata.uns['history_log'].append(entry)

# --- DECORATOR ---
def monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Auto-detect name if it's a partial or bound method
        name = getattr(func, "__qualname__", func.__name__)
        
        # Attempt to find 'adata' in arguments
        adata = None
        for arg in args:
            if isinstance(arg, anndata.AnnData):
                adata = arg
                break
        
        if adata is None:
            adata = kwargs.get("adata")

        watcher = None
        if adata is not None:
            watcher = StructureWatcher(adata, func_name=name)

        try:
            result = func(*args, **kwargs)
        finally:
            if watcher:
                watcher.finish()
        return result
    return wrapper

