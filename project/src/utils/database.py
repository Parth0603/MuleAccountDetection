import os
import sqlite3
import json
import random
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from project.src.utils.logger import logger

# Try importing supabase; if background install isn't done, it will fall back to SQLite anyway
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

class DatabaseManager:
    def __init__(self):
        """
        Initializes the DatabaseManager.
        Detects if Supabase credentials are configured in the environment.
        If yes, routes operations to the Supabase cloud table layer.
        If no, falls back to a local SQLite database at project/data/cybershield.db.
        """
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        # We can force local SQLite by setting USE_SUPABASE=false
        self.use_supabase = (
            os.getenv("USE_SUPABASE", "true").lower() == "true"
            and self.supabase_url is not None
            and self.supabase_key is not None
            and create_client is not None
        )
        
        self.sqlite_db_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "cybershield.db")
        )
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.sqlite_db_path), exist_ok=True)
        
        if self.use_supabase:
            logger.info("DatabaseManager: Initializing Supabase cloud client...")
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                logger.info("DatabaseManager: Connected to Supabase successfully.")
            except Exception as e:
                logger.error(f"DatabaseManager: Supabase connection failed: {e}. Falling back to SQLite.")
                self.use_supabase = False
        
        if not self.use_supabase:
            logger.info(f"DatabaseManager: Using local SQLite mode database at: {self.sqlite_db_path}")
            self.conn = sqlite3.connect(self.sqlite_db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
        self.initialize_tables()

    def initialize_tables(self):
        """
        Creates all required tables if they do not exist.
        """
        if self.use_supabase:
            # Table schemas must be created via the Supabase SQL editor or migration files.
            # We assume tables exist or will be verified. In local SQLite, we create them dynamically.
            logger.info("DatabaseManager: Table structures should be initialized in Supabase dashboard.")
            return

        logger.info("DatabaseManager: Ensuring local SQLite tables exist...")
        cursor = self.conn.cursor()
        
        # 1. Accounts Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                customer_name TEXT,
                customer_segment TEXT,
                product_category TEXT,
                account_longevity_months INTEGER,
                balance_volume REAL,
                anomaly_index REAL,
                behavioral_features TEXT, -- JSON string representation
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Cases Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                account_id TEXT,
                status TEXT,
                assigned_analyst TEXT,
                escalation_level TEXT,
                opened_time TIMESTAMP,
                closed_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts(id)
            )
        """)
        
        # 3. Alerts Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                account_id TEXT,
                trigger_reason TEXT,
                severity TEXT,
                triggered_at TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts(id)
            )
        """)
        
        # 4. Investigations (Timeline Events) Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                case_id TEXT,
                actor TEXT,
                action TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            )
        """)
        
        # 5. Analyst Notes Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyst_notes (
                id TEXT PRIMARY KEY,
                case_id TEXT,
                analyst TEXT,
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            )
        """)
        
        # 6. Reports Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                case_id TEXT,
                analyst TEXT,
                report_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            )
        """)
        
        self.conn.commit()
        logger.info("DatabaseManager: Local SQLite tables verified successfully.")

    def seed_data_if_empty(self, raw_dataset_path: str = "dataset.csv"):
        """
        Seeds the database with 5000 accounts, 100 investigations, 50 alerts, and 20 high-risk cases
        by sampling directly from the actual raw dataset file if the database is currently empty.
        """
        # Check if already seeded
        if self.use_supabase:
            try:
                res = self.client.table("accounts").select("id", count="exact").limit(1).execute()
                if res.count and res.count > 0:
                    logger.info(f"DatabaseManager: Supabase already contains {res.count} accounts. Seeding skipped.")
                    return
            except Exception as e:
                logger.error(f"DatabaseManager: Seeding check failed on Supabase: {e}. SQL schema might not be configured.")
                return
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM accounts")
            count = cursor.fetchone()[0]
            if count > 0:
                logger.info(f"DatabaseManager: Local SQLite already contains {count} accounts. Seeding skipped.")
                return

        logger.info("DatabaseManager: Ingesting dataset.csv to perform premium seeding...")
        if not os.path.exists(raw_dataset_path):
            logger.error(f"DatabaseManager: Raw dataset.csv not found at {raw_dataset_path}. Seeding aborted.")
            return

        try:
            raw_df = pd.read_csv(raw_dataset_path)
            total_rows = len(raw_df)
            logger.info(f"DatabaseManager: Loaded dataset.csv containing {total_rows} rows.")
            
            # Select 5,000 rows
            # We want to ensure all 81 suspicious accounts (F3924 == 1) are included in our 5000 sample
            mule_df = raw_df[raw_df["F3924"] == 1]
            legit_df = raw_df[raw_df["F3924"] == 0]
            
            # Sample legit rows to fill up to 5000
            sample_legit_count = min(5000 - len(mule_df), len(legit_df))
            sampled_legit = legit_df.sample(n=sample_legit_count, random_state=42)
            
            # Combine
            seeding_df = pd.concat([mule_df, sampled_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            logger.info(f"DatabaseManager: Extracted 5,000 samples (including {len(mule_df)} active mules) for persistence.")
            
            # Define naming arrays for realistic Indian accounts
            first_names = ["Suresh", "Priyanka", "Arjun", "Deepa", "Ramesh", "Sunita", "Amit", "Kiran", "Vijay", "Anjali", "Rajesh", "Neeta", "Sanjay", "Meena", "Rahul", "Pooja", "Vikram", "Swati", "Anil", "Geeta"]
            last_names = ["Kumar", "Patel", "Sharma", "Nair", "Gupta", "Joshi", "Singh", "Reddy", "Mehta", "Rao", "Verma", "Choudhury", "Das", "Sen", "Pillai", "Iyer", "Banerjee", "Mishra", "Patil", "Saxena"]
            
            analysts = ["A. Sharma (Senior Forensic)", "K. Patel (Triage Specialist)", "S. Nair (Lead Cyber Cell)", "M. Sen (Compliance Auditor)"]
            
            accounts_data = []
            cases_data = []
            alerts_data = []
            investigations_data = []
            
            # 1. Generate 5000 Accounts
            logger.info("DatabaseManager: Formatting accounts payload...")
            for idx, row in seeding_df.iterrows():
                act_id = f"BOI-ACT-{100000 + idx}"
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
                
                # Check segment F3893 or set RETAIL
                segment = str(row.get("F3893", "RETAIL")).upper()
                if pd.isna(row.get("F3893")):
                    segment = "RETAIL"
                    
                product = str(row.get("F3886", "Savings"))
                if pd.isna(row.get("F3886")):
                    product = "Savings"
                    
                longevity = int(row.get("F3887", 24))
                if pd.isna(row.get("F3887")):
                    longevity = 24
                    
                balance = float(row.get("F3836", 0.0))
                if pd.isna(row.get("F3836")):
                    balance = 0.0
                    
                # Anomaly Index calculation fallback
                anomaly_idx = 0.015  # standard base
                
                # Serialize the row to JSON to act as our behavioral features
                features_dict = row.to_dict()
                # Clean up any NaN values to make it valid JSON
                cleaned_features = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in features_dict.items()}
                features_json = json.dumps(cleaned_features)
                
                accounts_data.append((
                    act_id,
                    name,
                    segment,
                    product,
                    longevity,
                    balance,
                    anomaly_idx,
                    features_json
                ))

            # Bulk save accounts
            if self.use_supabase:
                # Format to JSON list for Supabase
                sb_accounts = [
                    {
                        "id": x[0],
                        "customer_name": x[1],
                        "customer_segment": x[2],
                        "product_category": x[3],
                        "account_longevity_months": x[4],
                        "balance_volume": x[5],
                        "anomaly_index": x[6],
                        "behavioral_features": json.loads(x[7])
                    } for x in accounts_data
                ]
                # Chunked insert because 5000 is large
                for i in range(0, len(sb_accounts), 500):
                    self.client.table("accounts").insert(sb_accounts[i:i+500]).execute()
            else:
                cursor.executemany("""
                    INSERT INTO accounts (id, customer_name, customer_segment, product_category, account_longevity_months, balance_volume, anomaly_index, behavioral_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, accounts_data)
                self.conn.commit()
                
            logger.info("DatabaseManager: Successfully persisted 5,000 authentic accounts.")

            # 2. Select Mules to establish active 20 High-Risk Cases
            # Find Mule accounts in accounts_data
            mule_account_ids = []
            for idx, row in seeding_df.iterrows():
                if row["F3924"] == 1:
                    mule_account_ids.append(f"BOI-ACT-{100000 + idx}")

            logger.info(f"DatabaseManager: Identified {len(mule_account_ids)} real mule accounts in sampled set.")
            
            # We will create exactly 20 cases and 50 alerts
            random.seed(42)
            selected_mules = mule_account_ids[:20] if len(mule_account_ids) >= 20 else mule_account_ids
            
            # Fill up to 20 cases with standard random accounts if needed
            while len(selected_mules) < 20:
                random_act = f"BOI-ACT-{100000 + random.randint(0, 4999)}"
                if random_act not in selected_mules:
                    selected_mules.append(random_act)

            # Let's seed 50 Alerts first
            # Alerts severity and severity triggers
            alert_reasons = [
                ("Behavioral Anomaly Index alert triggered", "CRITICAL"),
                ("High volume transaction velocity spike", "HIGH"),
                ("ATM balance extraction outlier", "MEDIUM"),
                ("Dormant account reactivation wave", "HIGH"),
                ("Device verification failure threshold exceeded", "MEDIUM"),
                ("Immediate fund disbursement rate outlier", "CRITICAL")
            ]
            
            alert_ids = []
            for a_idx in range(50):
                a_id = f"ALR-2026-{10000 + a_idx}"
                # 20 active cases get alerts, others get some alerts
                if a_idx < 20:
                    act_id = selected_mules[a_idx]
                else:
                    act_id = f"BOI-ACT-{100000 + random.randint(20, 4999)}"
                
                reason, severity = random.choice(alert_reasons)
                triggered_time = (datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M")
                
                alerts_data.append((
                    a_id,
                    act_id,
                    reason,
                    severity,
                    triggered_time
                ))
                alert_ids.append(a_id)

            if self.use_supabase:
                sb_alerts = [
                    {
                        "id": x[0],
                        "account_id": x[1],
                        "trigger_reason": x[2],
                        "severity": x[3],
                        "triggered_at": x[4]
                    } for x in alerts_data
                ]
                self.client.table("alerts").insert(sb_alerts).execute()
            else:
                cursor.executemany("""
                    INSERT INTO alerts (id, account_id, trigger_reason, severity, triggered_at)
                    VALUES (?, ?, ?, ?, ?)
                """, alerts_data)
                self.conn.commit()
            
            logger.info("DatabaseManager: Successfully persisted 50 alerts.")

            # 3. Create 20 Active Cases
            # 15 Open Cases, 5 Closed/Resolved Cases
            for c_idx, act_id in enumerate(selected_mules):
                case_id = f"CASE-2026-{1000 + c_idx}"
                
                if c_idx < 12:
                    status = "Open"
                    escalation = "Triage Logged"
                    assigned = random.choice(analysts[:2])
                elif c_idx < 16:
                    status = "Escalated - Frozen"
                    escalation = "Hard Hold Applied"
                    assigned = analysts[2]
                else:
                    status = "Closed - Resolved"
                    escalation = "None - Resolved"
                    assigned = analysts[1]
                    
                opened_time = (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:%M")
                closed_time = (datetime.datetime.now() - datetime.timedelta(hours=random.randint(1, 24))).strftime("%Y-%m-%d %H:%M") if status == "Closed - Resolved" else None
                
                cases_data.append((
                    case_id,
                    act_id,
                    status,
                    assigned,
                    escalation,
                    opened_time,
                    closed_time
                ))

            if self.use_supabase:
                sb_cases = [
                    {
                        "id": x[0],
                        "account_id": x[1],
                        "status": x[2],
                        "assigned_analyst": x[3],
                        "escalation_level": x[4],
                        "opened_time": x[5],
                        "closed_time": x[6]
                    } for x in cases_data
                ]
                self.client.table("cases").insert(sb_cases).execute()
            else:
                cursor.executemany("""
                    INSERT INTO cases (id, account_id, status, assigned_analyst, escalation_level, opened_time, closed_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, cases_data)
                self.conn.commit()
                
            logger.info("DatabaseManager: Successfully persisted 20 active cases.")

            # 4. Generate 100 Investigations (Timeline Events / logs)
            # Create a rich history for our 20 cases
            investigation_actions = [
                ("System Engine", "Behavioral Anomaly Index alert triggered", "Triggered based on extreme outbound credit velocity ratios."),
                ("System Engine", "SHAP decision explanations calibrated", "Mapped top positive risk drivers."),
                ("Forensic Analyst", "Case initialized and routed", "Assigned analyst for behavioral profile reviews."),
                ("Forensic Analyst", "Added verification inquest note", "OTP transaction confirmation dispatched to primary contact."),
                ("Forensic Analyst", "Hard debit freeze applied", "Hard debit freeze logged. Funds locked under regulatory fair-lending safeguards."),
                ("Forensic Analyst", "Cyber Cell escalated", "Case brief package formatted and securely transmitted to IT cell.")
            ]
            
            invest_counter = 0
            for case in cases_data:
                c_id = case[0]
                o_time = datetime.datetime.strptime(case[5], "%Y-%m-%d %H:%M")
                
                # Add 4 sequential events to each case to total ~80-100 events
                for step in range(5):
                    i_id = f"INV-2026-{10000 + invest_counter}"
                    actor, action, notes = investigation_actions[step]
                    
                    # Offset time slightly
                    e_time = (o_time + datetime.timedelta(hours=step * 4)).strftime("%Y-%m-%d %H:%M")
                    
                    investigations_data.append((
                        i_id,
                        c_id,
                        actor,
                        action,
                        notes,
                        e_time
                    ))
                    invest_counter += 1

            if self.use_supabase:
                sb_invests = [
                    {
                        "id": x[0],
                        "case_id": x[1],
                        "actor": x[2],
                        "action": x[3],
                        "notes": x[4],
                        "created_at": x[5]
                    } for x in investigations_data
                ]
                self.client.table("investigations").insert(sb_invests).execute()
            else:
                cursor.executemany("""
                    INSERT INTO investigations (id, case_id, actor, action, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, investigations_data)
                self.conn.commit()

            logger.info(f"DatabaseManager: Successfully seeded {len(investigations_data)} timeline events.")
            logger.info("DatabaseManager: Data seeding complete.")

        except Exception as e:
            logger.error(f"DatabaseManager: Seeding failed: {e}")

    # =========================================================
    # CORE APIs FOR ACCOUNTS, CASES, ALERTS, TIMELINES
    # =========================================================
    
    def get_accounts(self, limit: int = 5000) -> List[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("accounts").select("*").limit(limit).execute()
            return res.data
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM accounts LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_account_by_id(self, act_id: str) -> Optional[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("accounts").select("*").eq("id", act_id).execute()
            return res.data[0] if res.data else None
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM accounts WHERE id = ?", (act_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_cases(self) -> List[Dict[str, Any]]:
        if self.use_supabase:
            # We want to join case with account name/details
            res = self.client.table("cases").select("*, accounts(customer_name, customer_segment, balance_volume, product_category)").execute()
            # Flatten join
            flattened = []
            for row in res.data:
                flat = dict(row)
                if "accounts" in row and row["accounts"]:
                    flat["customer_name"] = row["accounts"]["customer_name"]
                    flat["customer_segment"] = row["accounts"]["customer_segment"]
                    flat["balance_volume"] = row["accounts"]["balance_volume"]
                    flat["product_category"] = row["accounts"]["product_category"]
                flattened.append(flat)
            return flattened
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT cases.*, accounts.customer_name, accounts.customer_segment, accounts.balance_volume, accounts.product_category
                FROM cases
                JOIN accounts ON cases.account_id = accounts.id
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("cases").select("*, accounts(*)").eq("id", case_id).execute()
            if not res.data:
                return None
            case_data = dict(res.data[0])
            if "accounts" in case_data:
                case_data["account"] = case_data["accounts"]
            return case_data
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT cases.*, accounts.customer_name, accounts.customer_segment, accounts.balance_volume, accounts.product_category
                FROM cases
                JOIN accounts ON cases.account_id = accounts.id
                WHERE cases.id = ?
            """, (case_id,))
            row = cursor.fetchone()
            if not row:
                return None
            case_data = dict(row)
            
            # Fetch full account data
            act = self.get_account_by_id(case_data["account_id"])
            if act:
                case_data["account"] = act
            return case_data

    def get_alerts_by_account(self, act_id: str) -> List[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("alerts").select("*").eq("account_id", act_id).execute()
            return res.data
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE account_id = ?", (act_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_investigations_by_case(self, case_id: str) -> List[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("investigations").select("*").eq("case_id", case_id).order("created_at", ascending=True).execute()
            return res.data
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM investigations WHERE case_id = ? ORDER BY created_at ASC", (case_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_notes_by_case(self, case_id: str) -> List[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("analyst_notes").select("*").eq("case_id", case_id).order("created_at", ascending=True).execute()
            return res.data
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM analyst_notes WHERE case_id = ? ORDER BY created_at ASC", (case_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_report_by_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        if self.use_supabase:
            res = self.client.table("reports").select("*").eq("case_id", case_id).order("created_at", descending=True).limit(1).execute()
            return res.data[0] if res.data else None
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM reports WHERE case_id = ? ORDER BY created_at DESC LIMIT 1", (case_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================
    # MUTATORS & CASE WORKFLOW ACTIONS
    # =========================================================

    def add_investigation_log(self, case_id: str, actor: str, action: str, notes: str) -> str:
        i_id = f"INV-2026-{random.randint(100000, 999999)}"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if self.use_supabase:
            self.client.table("investigations").insert({
                "id": i_id,
                "case_id": case_id,
                "actor": actor,
                "action": action,
                "notes": notes,
                "created_at": created_at
            }).execute()
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO investigations (id, case_id, actor, action, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (i_id, case_id, actor, action, notes, created_at))
            self.conn.commit()
            
        return i_id

    def add_analyst_note(self, case_id: str, analyst: str, note: str) -> str:
        n_id = f"NTE-2026-{random.randint(100000, 999999)}"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if self.use_supabase:
            self.client.table("analyst_notes").insert({
                "id": n_id,
                "case_id": case_id,
                "analyst": analyst,
                "note": note,
                "created_at": created_at
            }).execute()
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO analyst_notes (id, case_id, analyst, note, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (n_id, case_id, analyst, note, created_at))
            self.conn.commit()
            
        # Also log as an investigation timeline event
        self.add_investigation_log(case_id, analyst, "Added Inquest Note", note)
        return n_id

    def save_report(self, case_id: str, analyst: str, report_content: str) -> str:
        r_id = f"REP-2026-{random.randint(100000, 999999)}"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if self.use_supabase:
            self.client.table("reports").insert({
                "id": r_id,
                "case_id": case_id,
                "analyst": analyst,
                "report_content": report_content,
                "created_at": created_at
            }).execute()
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO reports (id, case_id, analyst, report_content, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (r_id, case_id, analyst, report_content, created_at))
            self.conn.commit()
            
        # Log as an investigation event
        self.add_investigation_log(case_id, analyst, "Compiled Compliance Briefing Report", "A Deloitte-style audit document has been compiled and saved.")
        return r_id

    def update_case_workflow(self, case_id: str, status: str, escalation: str, analyst: str, log_msg: str) -> bool:
        if self.use_supabase:
            update_fields = {
                "status": status,
                "escalation_level": escalation,
                "assigned_analyst": analyst
            }
            if status == "Closed - Resolved":
                update_fields["closed_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                
            self.client.table("cases").update(update_fields).eq("id", case_id).execute()
        else:
            cursor = self.conn.cursor()
            if status == "Closed - Resolved":
                closed_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                cursor.execute("""
                    UPDATE cases
                    SET status = ?, escalation_level = ?, assigned_analyst = ?, closed_time = ?
                    WHERE id = ?
                """, (status, escalation, analyst, closed_time, case_id))
            else:
                cursor.execute("""
                    UPDATE cases
                    SET status = ?, escalation_level = ?, assigned_analyst = ?
                    WHERE id = ?
                """, (status, escalation, analyst, case_id))
            self.conn.commit()
            
        # Log timeline action
        self.add_investigation_log(case_id, analyst, f"Updated Case State: {status}", log_msg)
        return True

# Initialize database registry globally
db_manager = DatabaseManager()
db_manager.seed_data_if_empty()
