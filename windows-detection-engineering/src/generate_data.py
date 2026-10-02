"""
Generate a synthetic Windows Server 2016 Security event dataset.

This is a CONTROLLED BENCHMARK, not real-world data.
Real-world deployment requires validation on production logs.

IMPORTANT DESIGN NOTES:
- Benign and attack messages are drawn from templates with similar length
  distributions, so the model cannot cheat by memorizing message length.
- Benign activity includes after-hours events (backups, scheduled tasks).
- Attack activity includes business-hours events (living-off-the-land).
- Label noise is injected to simulate real-world mislabeled data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOTAL_EVENTS = 50_000
ATTACK_RATIO = 0.02          # 2% — realistic for SOC environments
N_ATTACKS = int(TOTAL_EVENTS * ATTACK_RATIO)
N_BENIGN = TOTAL_EVENTS - N_ATTACKS
LABEL_NOISE_RATE = 0.01      # 1% of labels flipped randomly

# ---------------------------------------------------------------------------
# MITRE ATT&CK techniques and their associated Windows Event IDs
# Source: https://attack.mitre.org/ and Windows Security auditing docs
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------
# Design principle: benign and attack templates must have OVERLAPPING length
# and word-count distributions. Otherwise the model learns to detect the
# synthetic generator, not real attack behavior.

BENIGN_TEMPLATES = {
    4624: [
        "An account was successfully logged on. Subject: Security ID: S-1-5-18 Account Name: SYSTEM Account Domain: NT AUTHORITY Logon ID: 0x3E7 Logon Type: 5.",
        "An account was successfully logged on. Subject: Security ID: S-1-5-21-1234567890 Account Name: Administrator Account Domain: CORP Logon Type: 3.",
        "An account was successfully logged on. Logon Type: 10 Remote Desktop Services.",
        "An account was successfully logged on. New Logon: Account Name: jdoe Account Domain: CORP Logon ID: 0x1A2B3C.",
        "An account was successfully logged on. Logon Type: 2 Interactive.",
    ],
    4625: [
        "An account failed to log on. Subject: Security ID: S-1-0-0 Account Name: - Account Domain: - Logon Type: 3.",
        "An account failed to log on. Failure Reason: Unknown user name or bad password. Status: 0xC000006D Sub Status: 0xC000006A.",
        "An account failed to log on. Logon Type: 10 Remote Desktop Services.",
        "An account failed to log on. Account Name: administrator Source Network Address: 192.168.1.50.",
        "An account failed to log on. Failure Reason: Account locked out.",
    ],
    4634: [
        "An account was logged off. Logon Type: 3.",
        "An account was logged off. Subject: Security ID: S-1-5-21-1234567890 Account Name: jdoe.",
        "An account was logged off. Logon Type: 2 Interactive.",
    ],
    4648: [
        "A logon was attempted using explicit credentials. Target Server: localhost Target User: admin.",
        "A logon was attempted using explicit credentials. Target Server: DC01 Target User: backup_svc.",
        "A logon was attempted using explicit credentials. Process: C:\\Windows\\System32\\runas.exe.",
    ],
    4672: [
        "Special privileges assigned to new logon. Subject: Security ID: S-1-5-18 Privileges: SeAssignPrimaryTokenPrivilege SeTcbPrivilege SeSecurityPrivilege.",
        "Special privileges assigned to new logon. Subject: Security ID: S-1-5-21-1234567890 Account Name: Administrator Privileges: SeDebugPrivilege.",
        "Special privileges assigned to new logon. Account Name: SYSTEM Privileges: SeBackupPrivilege SeRestorePrivilege.",
    ],
    4688: [
        "A new process has been created. New Process Name: C:\\Windows\\System32\\svchost.exe Creator Process Name: C:\\Windows\\System32\\services.exe.",
        "A new process has been created. New Process Name: C:\\Program Files\\Google\\Chrome\\chrome.exe Command Line: chrome.exe --type=renderer.",
        "A new process has been created. New Process Name: C:\\Windows\\System32\\cmd.exe Command Line: cmd.exe /c dir.",
        "A new process has been created. New Process Name: C:\\Windows\\System32\\notepad.exe.",
        "A new process has been created. New Process Name: C:\\Windows\\System32\\taskhostw.exe.",
    ],
    4720: [
        "A user account was created. New Account Name: tempuser New Account Domain: CORP.",
        "A user account was created. New Account Name: svc_backup New Account Domain: CORP.",
        "A user account was created. New Account Name: test_account.",
    ],
    4732: [
        "A member was added to a security-enabled local group. Member Name: tempuser Group Name: Administrators.",
        "A member was added to a security-enabled local group. Member Name: svc_backup Group Name: Backup Operators.",
        "A member was added to a security-enabled local group. Member Name: jdoe Group Name: Remote Desktop Users.",
    ],
}

ATTACK_TEMPLATES = {
    "T1046": [
        "An account was successfully logged on. Logon Type: 3 Source Network Address: 192.168.1.100.",
        "An account failed to log on. Logon Type: 3 Source Network Address: 10.0.0.55.",
        "A logon was attempted using explicit credentials. Target Server: DC01.",
        "An account was successfully logged on. Logon Type: 3 New Logon: admin.",
        "An account failed to log on. Source Network Address: 10.0.0.99.",
    ],
    "T1003": [
        "Special privileges assigned to new logon. Subject: Security ID: S-1-5-21-9876543210 Account Name: attacker.",
        "An account was successfully logged on. Logon Type: 9 New Logon: admin.",
        "A new process has been created. New Process Name: C:\\Temp\\mimikatz.exe Command Line: mimikatz.exe sekurlsa::logonpasswords.",
        "Special privileges assigned to new logon. Account Name: SYSTEM Privileges: SeDebugPrivilege.",
        "A new process has been created. New Process Name: C:\\Temp\\procdump.exe.",
    ],
    "T1047": [
        "A new process has been created. New Process Name: C:\\Windows\\System32\\wbem\\WMIC.exe Command Line: wmic process call create calc.exe.",
        "A new process has been created. New Process Name: C:\\Windows\\System32\\cmd.exe Command Line: cmd.exe /c whoami.",
        "A new process has been created. New Process Name: C:\\Windows\\System32\\powershell.exe Command Line: powershell -enc SQBFAFgA.",
        "A new process has been created. New Process Name: C:\\Windows\\System32\\wbem\\WMIC.exe.",
    ],
    "T1078": [
        "A user account was created. New Account Name: backdoor New Account Domain: CORP.",
        "A member was added to a security-enabled local group. Member Name: backdoor Group Name: Administrators.",
        "Special privileges assigned to new logon. Subject: Security ID: S-1-5-21-9876543210 Account Name: backdoor.",
        "A user account was created. New Account Name: svc_temp.",
    ],
    "T1070": [
        "The audit log was cleared. Subject: Security ID: S-1-5-21-9876543210 Account Name: attacker.",
        "The event logging service has shut down. Reason: The log was cleared by user.",
        "The audit log was cleared. Account Name: administrator.",
    ],
    "T1110": [
        "An account failed to log on. Failure Reason: Unknown user name or bad password.",
        "An account failed to log on. Logon Type: 3 Source Network Address: 10.0.0.99.",
        "An account was successfully logged on. Logon Type: 3 New Logon: admin.",
        "An account failed to log on. Account Name: administrator.",
        "An account failed to log on. Failure Reason: Account locked out.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_benign_message(event_id):
    """Pick a random benign message template for the given event ID."""
    options = BENIGN_TEMPLATES.get(event_id, ["Windows Security event."])
    return random.choice(options)


def random_attack_message(technique_id, event_id):
    """Pick a random attack message template for the given technique."""
    options = ATTACK_TEMPLATES.get(technique_id, ["Suspicious activity."])
    return random.choice(options)


# ---------------------------------------------------------------------------
# Event generators
# ---------------------------------------------------------------------------
def generate_benign_events(n):
    """
    Generate n benign Windows Security events.

    Design: 85% of benign events occur during business hours, but 15% occur
    after hours (backups, scheduled tasks, remote workers). This prevents
    the model from trivially using "hour" as a discriminator.
    """
    events = []
    end_time = datetime(2026, 3, 1, 23, 59, 59)
    start_time = end_time - timedelta(days=7)

    for _ in range(n):
        day_offset = random.randint(0, 6)

        if random.random() < 0.85:
            # Business hours
            hour = random.randint(8, 17)
        else:
            # After-hours (backups, scheduled tasks)
            hour = random.choice(list(range(0, 8)) + list(range(18, 24)))

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        timestamp = start_time + timedelta(
            days=day_offset, hours=hour, minutes=minute, seconds=second
        )

        event_id = random.choices(
            BENIGN_EVENT_IDS,
            weights=[40, 15, 10, 10, 5, 15, 3, 2],
        )[0]

        events.append({
            "timestamp": timestamp,
            "event_id": event_id,
            "logon_type": random.choice(LOGON_TYPES),
            "process_id": random.randint(100, 9999),
            "message": random_benign_message(event_id),
            "is_malicious": 0,
            "attack_type": "Benign",
        })

    return events


def generate_attack_events(n):
    """
    Generate n attack events across the 6 MITRE techniques.

    Design: 30% of attacks occur during business hours (living-off-the-land),
    and 70% after hours. This prevents the model from trivially using "hour"
    as a discriminator.
    """
    events = []
    end_time = datetime(2026, 3, 1, 23, 59, 59)
    start_time = end_time - timedelta(days=7)

    techniques = list(ATTACK_TECHNIQUES.keys())

    for _ in range(n):
        technique_id = random.choice(techniques)
        technique = ATTACK_TECHNIQUES[technique_id]

        day_offset = random.randint(0, 6)

        if random.random() < 0.30:
            # Business-hours attack
            hour = random.randint(8, 17)
        else:
            # After-hours attack
            hour = random.choice(list(range(0, 8)) + list(range(18, 24)))

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
            "message": random_attack_message(technique_id, event_id),
            "is_malicious": 1,
            "attack_type": technique_id,
        })

    return events


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------
def inject_label_noise(df, noise_rate=LABEL_NOISE_RATE):
    """
    Flip a fraction of labels randomly to simulate real-world mislabeled data.

    Real SOC data contains mislabeled events (false alarms marked as benign,
    true incidents marked as benign, etc.). Without label noise, models can
    achieve unrealistic performance.
    """
    n_noise = int(len(df) * noise_rate)
    noise_idx = np.random.choice(df.index, size=n_noise, replace=False)
    df.loc[noise_idx, "is_malicious"] = 1 - df.loc[noise_idx, "is_malicious"]
    # Also flip attack_type to keep consistency
    df.loc[noise_idx & (df["is_malicious"] == 1), "attack_type"] = "Benign"
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Generating {TOTAL_EVENTS} events "
          f"({N_ATTACKS} attacks, {N_BENIGN} benign)...")

    benign = generate_benign_events(N_BENIGN)
    attacks = generate_attack_events(N_ATTACKS)

    df = pd.DataFrame(benign + attacks)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Inject label noise
    print(f"Injecting {LABEL_NOISE_RATE:.0%} label noise...")
    df = inject_label_noise(df, LABEL_NOISE_RATE)

    # Save
    df.to_csv("data/windows_security_events.csv", index=False)

    print(f"Done. Saved to data/windows_security_events.csv")
    print(f"Attack ratio: {df['is_malicious'].mean():.4f}")
    print(f"\nAttack type distribution:")
    print(df[df["is_malicious"] == 1]["attack_type"].value_counts())
    print(f"\nMessage length statistics by class:")
    print(df.groupby("is_malicious")["message"].apply(
        lambda x: x.str.len().describe()
    ))


if __name__ == "__main__":
    main()