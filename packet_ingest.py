"""
packet_ingest.py
================
Simulates raw packet/flow ingestion from CICIoT2023 CSV files.
(Replaces scapy/PCAP ingestion when raw captures are unavailable.)

When real PCAP files become available, replace `ingest_csv()` with a
scapy-based reader that populates the same FlowRecord dataclass.

Usage
-----
    from packet_ingest import ingest_csv, FlowRecord
    for record in ingest_csv("Dataset/CICIOT23/test/test.csv", n_flows=100):
        print(record.flow_duration, record.label)
"""

from dataclasses import dataclass, field
from typing import Generator, Optional
import csv
import os


# ── FlowRecord: mirrors the CICIoT2023 feature schema ────────────────────────

@dataclass
class FlowRecord:
    # Timing & header
    flow_duration:    float = 0.0
    Header_Length:    float = 0.0
    Protocol_Type:    float = 0.0
    Duration:         float = 0.0
    Rate:             float = 0.0
    Srate:            float = 0.0
    Drate:            float = 0.0
    # TCP flags
    fin_flag_number:  float = 0.0
    syn_flag_number:  float = 0.0
    rst_flag_number:  float = 0.0
    psh_flag_number:  float = 0.0
    ack_flag_number:  float = 0.0
    ece_flag_number:  float = 0.0
    cwr_flag_number:  float = 0.0
    # Counts
    ack_count:        float = 0.0
    syn_count:        float = 0.0
    fin_count:        float = 0.0
    urg_count:        float = 0.0
    rst_count:        float = 0.0
    # Protocol indicators
    HTTP:             float = 0.0
    HTTPS:            float = 0.0
    DNS:              float = 0.0
    Telnet:           float = 0.0
    SMTP:             float = 0.0
    SSH:              float = 0.0
    IRC:              float = 0.0
    TCP:              float = 0.0
    UDP:              float = 0.0
    DHCP:             float = 0.0
    ARP:              float = 0.0
    ICMP:             float = 0.0
    IPv:              float = 0.0
    LLC:              float = 0.0
    # Statistical features
    Tot_sum:          float = 0.0
    Min:              float = 0.0
    Max:              float = 0.0
    AVG:              float = 0.0
    Std:              float = 0.0
    Tot_size:         float = 0.0
    IAT:              float = 0.0
    Number:           float = 0.0
    Magnitue:         float = 0.0
    Radius:           float = 0.0
    Covariance:       float = 0.0
    Variance:         float = 0.0
    Weight:           float = 0.0
    # Ground truth (absent during real edge inference)
    label:            Optional[str] = None


# ── Column name mapping: CSV header → FlowRecord field ───────────────────────

_COL_MAP = {
    "flow_duration":   "flow_duration",
    "Header_Length":   "Header_Length",
    "Protocol Type":   "Protocol_Type",
    "Duration":        "Duration",
    "Rate":            "Rate",
    "Srate":           "Srate",
    "Drate":           "Drate",
    "fin_flag_number": "fin_flag_number",
    "syn_flag_number": "syn_flag_number",
    "rst_flag_number": "rst_flag_number",
    "psh_flag_number": "psh_flag_number",
    "ack_flag_number": "ack_flag_number",
    "ece_flag_number": "ece_flag_number",
    "cwr_flag_number": "cwr_flag_number",
    "ack_count":       "ack_count",
    "syn_count":       "syn_count",
    "fin_count":       "fin_count",
    "urg_count":       "urg_count",
    "rst_count":       "rst_count",
    "HTTP":            "HTTP",
    "HTTPS":           "HTTPS",
    "DNS":             "DNS",
    "Telnet":          "Telnet",
    "SMTP":            "SMTP",
    "SSH":             "SSH",
    "IRC":             "IRC",
    "TCP":             "TCP",
    "UDP":             "UDP",
    "DHCP":            "DHCP",
    "ARP":             "ARP",
    "ICMP":            "ICMP",
    "IPv":             "IPv",
    "LLC":             "LLC",
    "Tot sum":         "Tot_sum",
    "Min":             "Min",
    "Max":             "Max",
    "AVG":             "AVG",
    "Std":             "Std",
    "Tot size":        "Tot_size",
    "IAT":             "IAT",
    "Number":          "Number",
    "Magnitue":        "Magnitue",
    "Radius":          "Radius",
    "Covariance":      "Covariance",
    "Variance":        "Variance",
    "Weight":          "Weight",
    "label":           "label",
}


def _parse_row(row: dict) -> FlowRecord:
    """Convert a CSV row dict → FlowRecord."""
    record = FlowRecord()
    for csv_col, attr in _COL_MAP.items():
        if csv_col in row:
            val = row[csv_col]
            if attr == "label":
                setattr(record, attr, val if val else None)
            else:
                try:
                    setattr(record, attr, float(val))
                except (ValueError, TypeError):
                    setattr(record, attr, 0.0)
    return record


def ingest_csv(
    csv_path: str,
    n_flows: Optional[int] = None,
) -> Generator[FlowRecord, None, None]:
    """
    Yield FlowRecord objects one-at-a-time from a CICIoT2023 CSV.

    Parameters
    ----------
    csv_path : str
        Path to a train/test/validation CSV.
    n_flows  : int or None
        Maximum number of flows to yield (None = all).

    Yields
    ------
    FlowRecord
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _parse_row(row)
            count += 1
            if n_flows is not None and count >= n_flows:
                break


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"Dataset\CICIOT23\test\test.csv"
    print(f"Reading 5 flows from: {path}")
    for i, rec in enumerate(ingest_csv(path, n_flows=5)):
        print(f"  Flow {i+1}: duration={rec.flow_duration:.3f}  label={rec.label}")
