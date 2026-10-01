set -u
REPS="${REPS:-5}"; WORKERS="${WORKERS:-20}"; RUN_TIMEOUT="${RUN_TIMEOUT:-18000}"

CENSUSDP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DAS1940="${DAS1940_HOME:-$HOME/das1940}"
WHO="$(id -un)"
DATADIR="$CENSUSDP/data/ipums_1940"; OUT="$CENSUSDP/data/out/ipums_1940"
CAMP="$OUT/campaign"; LOG="$CAMP/campaign.log"

[ -f "$CENSUSDP/benchmarks/metrics.py" ] || { echo "$CENSUSDP is not the CensusDP tree"; exit 1; }
[ -f "$DAS1940/data/EXT1940USCB.dat" ] || { echo "no .dat in $DAS1940/data"; exit 1; }
mkdir -p "$CAMP" "$DATADIR"
cd "$CENSUSDP" || exit 1        # common.py resolves data/ relative to the repository root

STARTED="$(date +%s)"
CORES="$(nproc)"
# Busy and total CPU jiffies. A measured run has the host to itself, so machine-wide IS
# the run -- and it is the same instrument the vmstat reading during the DAS run used.
jiffies() { awk '/^cpu /{i=$5+$6; t=0; for(k=2;k<=NF;k++) t+=$k; print t-i, t}' /proc/stat; }
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
meminfo_mb() { awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%d", (t-a)/1024}' /proc/meminfo; }
# Everything deleted is a CSV this script created under $OUT; nothing else is ever removed.
drop() { for p in "$@"; do case "$p" in "$OUT"/*.csv) rm -f -- "$p" ;;
         *) echo "refusing to delete $p" >&2; exit 1 ;; esac; done; }

. "$CENSUSDP/venv/bin/activate" || exit 1
say "=== $REPS replicate(s) per arm, $WORKERS workers, timeout ${RUN_TIMEOUT}s"

# ─────────────────────────────────── preparation, skipped when already done
if [ ! -f "$DATADIR/EXT1940USCB_AK.dat" ]; then   # prepare.py always builds the sample first
  ln "$DAS1940/data/EXT1940USCB_AK.dat" "$DATADIR/EXT1940USCB_AK.dat" || exit 1
fi
if [ ! -f "$DATADIR/ipums_1940.parquet" ]; then
  if [ ! -f "$DATADIR/EXT1940USCB.dat.gz" ]; then
    # prepare.py reads the .gz and DuckDB takes the codec off the extension, so the plain
    # 40G .dat has to be compressed once. pigz uses every core.
    ZIP="$(command -v pigz || command -v gzip)"
    say "prep: compressing the .dat with $ZIP (one time)"
    "$ZIP" -c "$DAS1940/data/EXT1940USCB.dat" > "$DATADIR/EXT1940USCB.dat.gz" || exit 1
  fi
  say "prep: building the Parquet"
  python -m benchmarks.ipums_1940.prepare --full 2>&1 | tee -a "$LOG" || exit 1
fi
if [ ! -f "$DATADIR/ipums_1940_nodes.json" ]; then
  # runner.py requires this: a timed run must not spend its own time counting nodes.
  say "prep: counting nodes, checking the declared domains"
  python -m benchmarks.ipums_1940.columns 2>&1 | tee -a "$LOG" || exit 1
fi

# ─────────────────────────────────── one run
run_one() {
  name="$2"; csv="$OUT/$name.csv"
  # Both arms: pure DP at the same epsilon as the DAS -- no delta, no zCDP conversion -- under
  # their uniform per-level split, so the measurement structure is the only thing that differs.
  if [ "$1" = fj ]; then args="--full-joint --workload das --epsilon 4 --composition uniform --unit-bounds"
  # mip rather than the default sweep: at 4,640 cells a global controlled rounding is the closer
  # analogue of what the DAS does.
  else args="--structure das --epsilon 4 --rounding mip --composition uniform --unit-bounds"; fi

  say ""; say "───────── $name : $args --workers $WORKERS"
  drop "$csv" "$OUT/${name}_5col.csv"

  t0="$(date +%s)"; peak=0; local n=0; set -- $(jiffies); jb=$1; jt=$2
  # shellcheck disable=SC2086
  timeout "$RUN_TIMEOUT" python -m benchmarks.ipums_1940.driver $args \
      --workers "$WORKERS" --name "$name" > "$CAMP/$name.log" 2>&1 &
  pid=$!
  base="$(meminfo_mb)"; peak_init=0; peak_est=0; peak_merge=0; pooled=0
  while kill -0 "$pid" 2>/dev/null; do
    rss="$(meminfo_mb)"
    procs="$(pgrep -u "$WHO" -f multiprocessing.spawn 2>/dev/null | wc -l)"
    if [ "${procs:-0}" -gt 0 ]; then
      pooled=1; [ "$rss" -gt "$peak_est" ] && peak_est="$rss"
    elif [ "$pooled" = 1 ]; then
      [ "$rss" -gt "$peak_merge" ] && peak_merge="$rss"
    else
      [ "$rss" -gt "$peak_init" ] && peak_init="$rss"
    fi
    [ "${rss:-0}" -gt "$peak" ] && peak="$rss"
    n=$((n + 1))
    [ $((n % 90)) -eq 0 ] && say "    ...$(( ($(date +%s) - t0) / 60 )) min, peak rss ${peak}M"
    sleep 20
  done
  wait "$pid"; status=$?; elapsed=$(( $(date +%s) - t0 ))
  set -- $(jiffies)
  cores="$(awk "BEGIN{d=$2-$jt; print (d>0 ? ($1-$jb)/d*$CORES : 0)}")"
  pkill -u "$WHO" -f 'benchmarks.ipums_1940.driver' 2>/dev/null   # a worker can outlive a kill

  five=none
  if [ "$status" -eq 0 ]; then
    say "    done in ${elapsed}s, peak rss ${peak}M, $(printf %.1f "$cores") of $CORES cores busy on average"
    five="$(python -m benchmarks.ipums_1940.score_five "$name" 2>>"$LOG")" || five=none
    if [ "$five" != none ]; then
      python -m benchmarks.metrics ipums_1940 "$five" 2>&1 | tee -a "$LOG" | tail -5
    fi
  else
    say "    FAILED (exit $status) after ${elapsed}s -- see $CAMP/$name.log"
  fi
  drop "$csv" "$OUT/${name}_5col.csv"

  printf '{"name": "%s", "status": %d, "wall_seconds": %d, "peak_rss_gb": %.2f, "peak_initialize_gb": %.2f, "peak_estimation_gb": %.2f, "peak_merge_gb": %.2f, "baseline_rss_gb": %.2f, "mean_cores_busy": %.2f, "host_cores": %d, "rss_samples": %d, "run_record": "%s.json", "metrics": "%s_metrics.json"}\n' \
    "$name" "$status" "$elapsed" "$(awk "BEGIN{print $peak/1024}")" "$(awk "BEGIN{print $peak_init/1024}")" "$(awk "BEGIN{print $peak_est/1024}")" "$(awk "BEGIN{print $peak_merge/1024}")" "$(awk "BEGIN{print $base/1024}")" "$cores" "$CORES" "$n" "$name" "$five" \
    > "$CAMP/$name.campaign.json"
}

for arm in fj mg; do
  n=1; while [ "$n" -le "$REPS" ]; do run_one "$arm" "ours_${arm}_rep$n"; n=$((n + 1)); done
done

say ""; say "=== campaign done, $(( ($(date +%s) - STARTED) / 60 )) min total"
cat "$CAMP"/*.campaign.json 2>/dev/null | tee -a "$LOG"
