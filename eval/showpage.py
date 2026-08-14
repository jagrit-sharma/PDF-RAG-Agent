import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingest


def main(prefix, page):
    for p in ingest.loadCorpus():
        if p["source"].startswith(prefix) and p["page"] == page:
            print(f"\n--- {p['source']}  page {page} ---\n")
            print(textwrap.fill(" ".join((p["text"] or "").split()), 100))
            return

    print(f"no page {page} in a file starting with {prefix!r}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
