import pathlib
import uuid
from datetime import datetime, timezone
import json

LOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR/"experiments.jsonl"

FIELDS = (
    # identity -- log.py generates these, callers must not pass them
    "run_id",
    "timestamp",
    # the design
    "cover_fingerprint",     # fingerprint(terms)
    "terms_json",            # serialize(terms) -- reconstructable
    "n_terms",
    "strategy_tags",         # e.g. ["baseline"], ["octagon", "mirror-paired"]
    # the measurement
    "cache_key",             # pointer to the cached QASM; do not log the QASM
    "cache_hit",             # False = this call actually paid for synthesis
    "width",
    "real_depth",
    "real_cx",
    # synthesis refused the design -- a result, not a crash. None when it succeeded.
    "error",                 # first line of the ClassiqAPIError
    "required_width",        # qubits the server said it needed, parsed from that message
    # the verification, when it was run
    "ok",
    "max_error",
    "ancilla_error",
    "norm_error",
    # settings, in full -- cheap, and the only thing that survives new knobs
    "seed",
    "syn_prefs",
    "syn_constraints",
    "trans_prefs",
    # cost accounting
    "wall_time",             # seconds for the whole evaluate() call, verify included
    # provenance
    "classiq_version",       # client
    "server_version",        # parsed from the QASM header -- not in the cache key
    "source_tool",           # "classiq" now; others once 0.8 lands
    # Phase 1 -- proxy features and predictions. None until then.
    "n_ancillas",
    "max_fanin",
    "total_controls",
    "proxy_depth",
    "proxy_cx",
    # Phase 4 -- search lineage. None until then.
    "parent_run_id",
    "move_applied",
)

def append(row: dict) -> str:
    """appends to jsonl log of experiments, returns run_id"""
    diff = set(row) - set(FIELDS) # check if the row contains any unsupported fields
    if diff:
        raise ValueError(f"Unsupported fields: {diff}")
    run_id = uuid.uuid4().hex[:12] # keep first 12 characters of uuid
    time = datetime.now(timezone.utc).isoformat()
    
    d = {k: row.get(k) for k in FIELDS} # only loop through keys in FIELDS
    d["run_id"] = run_id
    d["timestamp"] = time

    with LOG_PATH.open("a", encoding="utf-8") as f: # append to file
        f.write(json.dumps(d) + "\n")

    return run_id

def rows():
    rows = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for l in f:
                rows.append(json.loads(l))
    return rows

        
    
    

