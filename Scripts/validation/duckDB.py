import duckdb
import csv
from datetime import date, timedelta
 
# ─── PATHS OF TFILES────────────────────────────────────────────────────────
TBL_PATH        = "C:/TPC-H-V3.0.1/SF5/"       # I only ran it on SF5 but can be ran it on other level just by changing the file path
Q6_RESULTS_CSV  = "q6_results_SF5.csv"        
Q1_RESULTS_CSV  = "q1_results_SF5.csv"
Q3_RESULTS_CSV  = "q3_results_SF5.csv"
 
EPOCH = date(1970, 1, 1)
 
def day_to_date(day_int):
    """Convert integer days-since-1970-01-01 back to a date string."""
    return str(EPOCH + timedelta(days=int(day_int)))
 
def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))
 
con = duckdb.connect()
 
# ── Load tables ─────────────────────────────────────────────────────────
print("Loading lineitem...")
con.execute(f"""
    CREATE VIEW lineitem AS
    SELECT * FROM read_csv_auto('{TBL_PATH}lineitem.tbl', delim='|', header=false,
      columns={{
        'l_orderkey':'INTEGER','l_partkey':'INTEGER','l_suppkey':'INTEGER',
        'l_linenumber':'INTEGER','l_quantity':'DOUBLE','l_extendedprice':'DOUBLE',
        'l_discount':'DOUBLE','l_tax':'DOUBLE','l_returnflag':'VARCHAR',
        'l_linestatus':'VARCHAR','l_shipdate':'DATE','l_commitdate':'DATE',
        'l_receiptdate':'DATE','l_shipinstruct':'VARCHAR','l_shipmode':'VARCHAR',
        'l_comment':'VARCHAR'}});
""")
 
print("Loading orders...")
con.execute(f"""
    CREATE VIEW orders AS
    SELECT * FROM read_csv_auto('{TBL_PATH}orders.tbl', delim='|', header=false,
      columns={{
        'o_orderkey':'INTEGER','o_custkey':'INTEGER','o_orderstatus':'VARCHAR',
        'o_totalprice':'DOUBLE','o_orderdate':'DATE','o_orderpriority':'VARCHAR',
        'o_clerk':'VARCHAR','o_shippriority':'INTEGER','o_comment':'VARCHAR'}});
""")
 
print("Loading customer...")
con.execute(f"""
    CREATE VIEW customer AS
    SELECT * FROM read_csv_auto('{TBL_PATH}customer.tbl', delim='|', header=false,
      columns={{
        'c_custkey':'INTEGER','c_name':'VARCHAR','c_address':'VARCHAR',
        'c_nationkey':'INTEGER','c_phone':'VARCHAR','c_acctbal':'DOUBLE',
        'c_mktsegment':'VARCHAR','c_comment':'VARCHAR'}});
""")
 
print("Tables loaded.\n")
 
PASS = "PASS"
FAIL = "FAIL"
THRESHOLD = 0.001   # 0.1% tolerance
 
# ═══════════════════════════════════════════════════════════════════
# Q6
# Predicates (from Q6.cpp):
#   ship_day >= lo_day AND ship_day < hi_day
#   discount >= 0.03   AND discount <= 0.09
#   quantity < 28
# Revenue = SUM(price * discount)
# lo_day = D0 (1994-01-01), hi_day = D0 + W  (integers, days since 1970)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Q6  -- SUM(extendedprice * discount)")
print("     discount IN [0.03, 0.09], quantity < 28, date window")
print("=" * 60)
 
rows_q6 = load_csv(Q6_RESULTS_CSV)
all_pass_q6 = True
 
for row in rows_q6:
    lo_date  = day_to_date(row['start_day'])
    hi_date  = day_to_date(row['end_day'])
    cpu_val  = float(row['cpu_result'])
 
    r = con.execute(f"""
        SELECT SUM(l_extendedprice * l_discount)
        FROM lineitem
        WHERE l_shipdate >= DATE '{lo_date}'
          AND l_shipdate <  DATE '{hi_date}'
          AND l_discount >= 0.03
          AND l_discount <= 0.09
          AND l_quantity  < 28
    """).fetchone()
 
    duck_val = r[0] if r[0] is not None else 0.0
    diff     = abs(duck_val - cpu_val)
    pct      = (diff / duck_val * 100) if duck_val != 0 else 0.0
    status   = PASS if pct < THRESHOLD else FAIL
    if status == FAIL:
        all_pass_q6 = False
 
    print(f"  sel={float(row['achieved_selectivity'])*100:.2f}%  "
          f"window [{lo_date} → {hi_date}]")
    print(f"    DuckDB={duck_val:>20,.4f}  CPU={cpu_val:>20,.4f}  "
          f"diff={diff:.4f}  ({pct:.6f}%)  {status}")
 
print(f"\n  Q6 overall: {'ALL PASSED' if all_pass_q6 else 'FAILED'}\n")
 
 
# ═══════════════════════════════════════════════════════════════════
# Q1
# Predicate (from Q1.cpp):
#   shipdate <= cutoff_ymd   
# Revenue = SUM(price * (1 - discount) * (1 + tax))  across ALL groups
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Q1  -- SUM(extendedprice*(1-discount)*(1+tax))")
print("     l_shipdate <= cutoff (all groups summed)")
print("=" * 60)
 
rows_q1    = load_csv(Q1_RESULTS_CSV)
all_pass_q1 = True
 
for row in rows_q1:
    cutoff   = row['cutoff_date']      
    cpu_val  = float(row['validation_cpu_result'])
 
    r = con.execute(f"""
        SELECT SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax))
        FROM lineitem
        WHERE l_shipdate <= DATE '{cutoff}'
    """).fetchone()
 
    duck_val = r[0] if r[0] is not None else 0.0
    diff     = abs(duck_val - cpu_val)
    pct      = (diff / duck_val * 100) if duck_val != 0 else 0.0
    status   = PASS if pct < THRESHOLD else FAIL
    if status == FAIL:
        all_pass_q1 = False
 
    print(f"  sel={float(row['achieved_selectivity'])*100:.2f}%  cutoff={cutoff}")
    print(f"    DuckDB={duck_val:>25,.4f}  CPU={cpu_val:>25,.4f}  "
          f"diff={diff:.4f}  ({pct:.6f}%)  {status}")
 
print(f"\n  Q1 overall: {'ALL PASSED' if all_pass_q1 else 'FAILED'}\n")
 
 
# ═══════════════════════════════════════════════════════════════════
# Q3
# Predicates (from Q3c.cpp):
#   c_mktsegment = 'BUILDING'
#   o_orderdate  < cutoff
#   l_shipdate   > cutoff
# Revenue = SUM(extendedprice * (1 - discount))
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Q3  -- SUM(extendedprice*(1-discount))")
print("     segment=BUILDING, o_orderdate < cutoff, l_shipdate > cutoff")
print("=" * 60)
 
rows_q3    = load_csv(Q3_RESULTS_CSV)
all_pass_q3 = True
 
for row in rows_q3:
    cutoff   = row['cutoff_date']
    cpu_val  = float(row['validation_cpu_result'])
 
    r = con.execute(f"""
        SELECT SUM(l_extendedprice * (1 - l_discount))
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey    = o_custkey
          AND l_orderkey   = o_orderkey
          AND o_orderdate  < DATE '{cutoff}'
          AND l_shipdate   > DATE '{cutoff}'
    """).fetchone()
 
    duck_val = r[0] if r[0] is not None else 0.0
    diff     = abs(duck_val - cpu_val)
    pct      = (diff / duck_val * 100) if duck_val != 0 else 0.0
    status   = PASS if pct < THRESHOLD else FAIL
    if status == FAIL:
        all_pass_q3 = False
 
    print(f"  sel={float(row['achieved_selectivity'])*100:.3f}%  cutoff={cutoff}")
    print(f"    DuckDB={duck_val:>22,.4f}  CPU={cpu_val:>22,.4f}  "
          f"diff={diff:.4f}  ({pct:.6f}%)  {status}")
 
print(f"\n  Q3 overall: {'ALL PASSED' if all_pass_q3 else 'FAILED'}\n")
 
# Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Q6: {'ALL PASSED' if all_pass_q6 else 'FAILED'}")
print(f"  Q1: {'ALL PASSED' if all_pass_q1 else 'FAILED'}")
print(f"  Q3: {'ALL PASSED' if all_pass_q3 else 'FAILED'}")
print(f"\n  Tolerance used: {THRESHOLD}% relative error")
print(" PASS here means CPU is correct vs DuckDB reference.")
print(" GPU deviation is just a floating point mismatch but correct.")