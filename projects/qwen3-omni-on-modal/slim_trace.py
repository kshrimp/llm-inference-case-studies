"""Stream a large Chrome-trace JSON (raw or .gz) and keep only kernel and CUDA-runtime
events with the fields analyze_trace.py uses, so it fits in a small machine's memory.

The torch exporter writes one event per indented block (a line "  {" through a line
"  }," or "  }"). Usage: python slim_trace.py <in.trace.json[.gz]> <out.slim.json.gz>
"""
import gzip
import json
import sys

KEEP = {"kernel", "cuda_runtime"}


def events(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    buf: list[str] | None = None
    with opener(path, "rt") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s == "  {":
                buf = ["{"]
            elif buf is not None and s.startswith("  }"):
                buf.append("}")
                yield json.loads("".join(buf))
                buf = None
            elif buf is not None:
                buf.append(s)


def main(src: str, dst: str) -> None:
    kept = seen = 0
    with gzip.open(dst, "wt") as out:
        out.write('{"traceEvents": [\n')
        first = True
        for ev in events(src):
            seen += 1
            if ev.get("cat") not in KEEP:
                continue
            slim = {"cat": ev["cat"], "name": ev["name"], "ts": ev["ts"], "dur": ev.get("dur", 0),
                    "args": {"correlation": (ev.get("args") or {}).get("correlation"),
                             "stream": (ev.get("args") or {}).get("stream")}}
            out.write(("" if first else ",\n") + json.dumps(slim))
            first = False
            kept += 1
        out.write("\n]}\n")
    print(f"{src}: {seen} events, kept {kept} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
