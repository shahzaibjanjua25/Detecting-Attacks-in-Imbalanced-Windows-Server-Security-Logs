"""
Generate a synthetic Windows Server 2016 Security event dataset.

This is a CONTROLLED BENCHMARK, not real-world data.
Real-world deployment requires validation on production logs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configuration
TOTAL_EVENTS = 50_000
ATTACK_RATIO = 0.02  # 2% — realistic for SOC environments
N_ATTACKS = int(TOTAL_EVENTS * ATTACK_RATIO)
N_BENIGN = TOTAL_EVENTS - N_ATTACKS

# MITRE ATT&CK techniques and their associated Windows Event IDs
# Source: https://attack.mitre.org/ and Windows Security auditing docs
ATTACK_TECHNIQUES = {
    "T1046": {  # Network Service Scanning
        "name": "Network Service Scanning",
        "event_ids": [4624, 4625, 4648],
    },
    "T1003": {  # OS Credential Dumping
        "name": "OS Credential Dumping",
        "event_ids": [4672, 4624, 4688],
    },
    "T1047": {  # WMI Execution
        "name": "WMI Execution",
        "event_ids": [4688],
    },
    "T1078": {  # Valid Accounts
        "name": "Valid Accounts",
        "event_ids": [4720, 4732, 4672],
    },
    "T1070": {  # Indicator Removal
        "name": "Indicator Removal",
        "event_ids": [1102, 1100],
    },
    "T1110": {  # Brute Force
        "name": "Brute Force",
        "event_ids": [4625, 4624],
    },
}

# Benign event IDs — common in normal Windows Server operations
BENIGN_EVENT_IDS = [4624, 4625, 4634, 4648, 4672, 4688, 4720, 4732]

# Logon types (Windows Security docs)
LOGON_TYPES = [2, 3, 5, 7, 8, 9, 10, 11, 12]


def generate_benign_events(n):
    """Generate n benign Windows Security events."""
    events = []
    # Simulate 7 days of activity
    end_time = datetime(2026, 3, 1, 23, 59, 59)
    start_time = end_time - timedelta(days=7)

    for _ in range(n):
        # Benign activity: mostly business hours, some after-hours
        if random.random() < 0.85:
            # Business hours: 8am-6pm
            day_offset = random.randint(0, 6)
            hour = random.randint(8, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
        else:
            # After hours: 6pm-8am
            day_offset = random.randint(0, 6)
            hour = random.choice(list(range(18, 24)) + list(range(0, 8)))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)

        timestamp = start_time + timedelta(
            days=day_offset, hours=hour, minutes=minute, seconds=second
        )

        event_id = random.choices(
            BENIGN_EVENT_IDS,
            # Realistic distribution: logons dominate
            weights=[40, 15, 10, 10, 5, 15, 3, 2],
        )[0]

        events.append({
            "timestamp": timestamp,
            "event_id": event_id,
            "logon_type": random.choice(LOGON_TYPES),
            "process_id": random.randint(100, 9999),
            "message": generate_benign_message(event_id),
            "is_malicious": 0,
            "attack_type": "Benign",
        })

    return events


def generate_benign_message(event_id):
    """Generate a plausible benign message for an event ID."""
    messages = {
        4624: "An account was successfully logged on. Subject: Security ID: S-1-5-18.",
        4625: "An account failed to log on. Subject: Security ID: S-1-5-18.",
        4634: "An account was logged off. Logon Type: 3.",
        4648: "A logon was attempted using explicit credentials.",
        4672: "Special privileges assigned to new logon.",
        4688: "A new process has been created. New Process Name: C:\\Windows\\System32\\svchost.exe",
        4720: "A user account was created. New Account Name: testuser.",
        4732: "A member was added to a security-enabled local group.",
    }
    return messages.get(event_id, "Windows Security event.")


def generate_attack_events(n):
    """Generate n attack events across the 6 MITRE techniques."""
    events = []
    end_time = datetime(2026, 3, 1, 23, 59, 59)
    start_time = end_time - timedelta(days=7)

    techniques = list(ATTACK_TECHNIQUES.keys())

    for _ in range(n):
        technique_id = random.choice(techniques)
        technique = ATTACK_TECHNIQUES[technique_id]

        # Attacks often occur after hours
        day_offset = random.randint(0, 6)
        hour = random.choice(list(range(0, 6)) + list(range(22, 24)))
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        timestamp = start_time + timedelta(
            days=day_offset, hours=hour, minutes=minute, seconds=second
        )

        event_id = random.choice(technique["event_ids"])

        events.append({
            "timestamp": timestamp,
            "event_id": event_id,
            "logon_type": random.choice(LOGON_TYPES),
            "process_id": random.randint(100, 9999),
            "message": generate_attack_message(technique_id, event_id),
            "is_malicious": 1,
            "attack_type": technique_id,
        })

    return events


def generate_attack_message(technique_id, event_id):
    """Generate a plausible attack message."""
    if technique_id == "T1046":
        return "Network scanning activity detected. Multiple connection attempts."
    elif technique_id == "T1003":
        return "Credential dumping tool executed. LSASS memory access detected."
    elif technique_id == "T1047":
        return "WMI process created. Command: wmic process call create cmd.exe"
    elif technique_id == "T1078":
        return "Suspicious account creation and privilege assignment."
    elif technique_id == "T1070":
        return "Security log cleared. Event log was cleared by user."
    elif technique_id == "T1110":
        return "Multiple failed logon attempts followed by success."
    return "Suspicious activity."


def main():
    print(f"Generating {TOTAL_EVENTS} events ({N_ATTACKS} attacks, {N_BENIGN} benign)...")

    benign = generate_benign_events(N_BENIGN)
    attacks = generate_attack_events(N_ATTACKS)

    df = pd.DataFrame(benign + attacks)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    df.to_csv("data/windows_security_events.csv", index=False)

    print(f"Done. Saved to data/windows_security_events.csv")
    print(f"Attack ratio: {df['is_malicious'].mean():.4f}")
    print(f"\nAttack type distribution:")
    print(df[df["is_malicious"] == 1]["attack_type"].value_counts())


if __name__ == "__main__":
    main()