"""Reruns the 120 hemisphere scenes of sim_stress_test.py (same seeds) with
the current vgap_pipeline.py. Usage: python code/vgap/test_hemisphere_axis_sim.py out.csv"""
import csv, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_stress_test as st
jobs, seed = [], 1000
for k in st.MESHES:
  for c in st.COLORS:
    for s in st.SCENARIOS:
      for t in st.TILTS:
        for rep in range(2):
          seed += 1
          if k == "hemisphere":
            jobs.append((k, c, s, t, rep, seed))
if __name__ == "__main__":
  t0 = time.time()
  with Pool(4, initializer=st._init) as p:
    rows = list(p.imap_unordered(st.trial, jobs))
  keys = sorted({k for r in rows for k in r})
  with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
  print(st.summarize(rows).split("=== WITH")[0])
  print("time", round(time.time() - t0))
