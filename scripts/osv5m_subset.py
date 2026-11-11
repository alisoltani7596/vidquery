#!/usr/bin/env python
"""Download the OSV-5M test split (CC BY-SA 4.0) and write fixed val/test manifests.

    python scripts/osv5m_subset.py --out-dir $SCRATCH/vidquery/data/osv5m --download

Images stay inside the downloaded zips; see vidquery.geo.osv5m.
"""

from vidquery.geo.osv5m import main

if __name__ == "__main__":
    raise SystemExit(main())
