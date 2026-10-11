# Data dictionary

Each row is one retained traffic-active one-second window from a device-specific capture.

| Column | Meaning |
|---|---|
| `device` | IoT device name/category used in experiment |
| `capture_id` | Identifier grouping windows from the same device-specific capture; keep all such windows in one CV partition |
| `activity` | Experimental activity label for the capture |
| `sec` | Time-window index/offset as recorded by the original extraction; do not assume an absolute timestamp |
| `bytes_per_s` | Traffic bytes counted in the retained one-second window |
| `pkts_per_s` | Packet count in the retained one-second window |
| `mean_pkt_size` | Mean observed packet size in window |
| `std_pkt_size` | Standard deviation of observed packet size in window |
| `mean_iat` | Mean inter-arrival time as computed by the original pipeline; unit and precise handling require confirmation from the original extraction code |

Only the five numerical traffic features are used as classification inputs. `activity` is the target and `capture_id` is a grouping key. The absence of the original feature-extraction script prevents independently verifying the exact definition/unit of every extracted feature. See the source paper for methodology.
