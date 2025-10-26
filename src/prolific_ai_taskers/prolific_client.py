"""
Prolific API client module.
"""

import json
import os
import secrets
import string
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


class ProlificClient:
    """Client for Prolific AI Task Builder API."""

    BASE_URL = "https://api.prolific.com/api/v1"

    def __init__(self):
        """Initialize client with credentials from environment variables."""
        self.api_token = os.environ.get("PROLIFIC_API_TOKEN")
        self.workspace_id = os.environ.get("PROLIFIC_WORKSPACE_ID")
        self.project_id = os.environ.get("PROLIFIC_PROJECT_ID")

        if not all([self.api_token, self.workspace_id, self.project_id]):
            raise ValueError("Missing Prolific credentials in .env file")

        self.headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json",
        }

        # Validate credentials
        resp = requests.get(f"{self.BASE_URL}/users/me/", headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        self.researcher_id = data["id"]
        self.researcher_name = data["name"]
        print(f"✅ Authenticated as {self.researcher_name}")

    def create_dataset(self, name: str) -> str:
        """Create a dataset and return its ID."""
        payload = {"name": name, "workspace_id": self.workspace_id}
        resp = requests.post(
            f"{self.BASE_URL}/data-collection/datasets",
            headers=self.headers,
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        dataset_id = resp.json()["id"]
        print(f"✅ Created dataset: {dataset_id}")
        return dataset_id

    def upload_dataset_file(self, dataset_id: str, file_path: Path):
        """Upload a CSV file to dataset."""
        # Get presigned URL
        resp = requests.get(
            f"{self.BASE_URL}/data-collection/datasets/{dataset_id}/upload-url/{file_path.name}",
            headers=self.headers,
        )
        resp.raise_for_status()
        upload_url = resp.json()["upload_url"]

        # Upload to S3
        with open(file_path, "rb") as f:
            put_resp = requests.put(upload_url, data=f)
        put_resp.raise_for_status()
        print(f"✅ Uploaded {file_path.name}")

    def wait_for_dataset_ready(self, dataset_id: str, timeout_sec: int = 900) -> bool:
        """Wait for dataset to be ready."""
        t0 = time.time()
        while True:
            resp = requests.get(
                f"{self.BASE_URL}/data-collection/datasets/{dataset_id}/status",
                headers=self.headers,
            )
            resp.raise_for_status()
            status = resp.json().get("status")
            print(f"Dataset status: {status}")

            if status == "READY":
                print("✅ Dataset ready")
                return True
            elif status == "ERROR":
                print("❌ Dataset processing failed")
                return False

            if time.time() - t0 > timeout_sec:
                print("⏰ Timeout")
                return False
            time.sleep(5)

    def create_batch(self, name: str, dataset_id: str, task_details: Dict) -> str:
        """Create a batch and return its ID."""
        payload = {
            "name": name,
            "workspace_id": self.workspace_id,
            "dataset_id": dataset_id,
            "task_details": task_details,
        }
        resp = requests.post(
            f"{self.BASE_URL}/data-collection/batches",
            headers=self.headers,
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        print(f"✅ Created batch: {batch_id}")
        return batch_id

    def add_batch_instructions(self, batch_id: str, instructions: List[Dict]):
        """
        Add instructions to batch.

        Automatically adds 'answer_limit': 1 if not present (required by Prolific API).
        """
        # Add answer_limit if not present (Prolific API requirement)
        for instruction in instructions:
            if 'answer_limit' not in instruction:
                instruction['answer_limit'] = 1

        payload = {"instructions": instructions}
        resp = requests.post(
            f"{self.BASE_URL}/data-collection/batches/{batch_id}/instructions",
            headers=self.headers,
            data=json.dumps(payload),
        )
        if not resp.ok:
            print(f"Error response: {resp.text}")
        resp.raise_for_status()
        print(f"✅ Added instructions")

    def initialize_batch(self, batch_id: str, dataset_id: str, tasks_per_group: int):
        """Initialize batch (starts processing)."""
        payload = {"dataset_id": dataset_id, "tasks_per_group": tasks_per_group}
        resp = requests.post(
            f"{self.BASE_URL}/data-collection/batches/{batch_id}/setup",
            headers=self.headers,
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        print(f"✅ Initialized batch")

    def wait_for_batch_ready(self, batch_id: str, timeout_sec: int = 900) -> bool:
        """Wait for batch to be ready."""
        t0 = time.time()
        last_status = None

        while True:
            resp = requests.get(
                f"{self.BASE_URL}/data-collection/batches/{batch_id}/status",
                headers=self.headers,
            )
            resp.raise_for_status()
            status = resp.json().get("status")

            if status != last_status:
                print(f"Batch status: {status}")
                last_status = status

            if status == "READY":
                print("✅ Batch ready")
                return True
            elif status == "ERROR":
                print("❌ Batch setup failed")
                return False

            if time.time() - t0 > timeout_sec:
                print("⏰ Timeout")
                return False
            time.sleep(5)

    def create_study(self, batch_id: str, task_name: str, internal_name: str,
                     description: str, estimated_completion_time: int, max_time: int,
                     reward: int, device_compatibility: List[str],
                     filters: List[Dict] = None) -> str:
        """Create a study and return its ID."""
        # Generate completion code
        alnum = string.ascii_uppercase + string.digits
        completion_code = "".join(secrets.choice(alnum) for _ in range(8))

        payload = {
            "name": task_name,
            "internal_name": internal_name,
            "description": description,
            "data_collection_method": "DC_TOOL",
            "data_collection_id": batch_id,
            "project": self.project_id,
            "estimated_completion_time": estimated_completion_time,
            "max_time": max_time,
            "total_available_places": 1,
            "reward": reward,
            "device_compatibility": device_compatibility,
            "completion_option": "code",
            "completion_codes": [{
                "code": completion_code,
                "code_type": "COMPLETED",
                "actions": [{"action": "AUTOMATICALLY_APPROVE"}],
            }],
            "filters": filters or [],
        }

        resp = requests.post(f"{self.BASE_URL}/studies/", headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        study_id = resp.json()["id"]
        print(f"✅ Created study: {study_id}")
        return study_id

    def update_study_participants(self, study_id: str, participants_per_task: int):
        """Update study with correct participant count."""
        resp = requests.get(f"{self.BASE_URL}/studies/{study_id}/", headers=self.headers)
        resp.raise_for_status()
        study = resp.json()

        access_details = [
            {
                "external_url": ad["external_url"],
                "total_allocation": participants_per_task,
                "allocated": ad["allocated"],
            }
            for ad in study.get("access_details", [])
        ]

        payload = {
            "internal_name": study["internal_name"],
            "name": study["name"],
            "description": study["description"],
            "reward": int(study["reward"]),
            "total_available_places": len(access_details) * participants_per_task,
            "study_labels": study["study_labels"],
            "data_collection_metadata": {
                "annotators_per_task": participants_per_task,
                "total_task_groups": len(access_details),
            },
            "audience": study.get("audience", "standard_sample"),
            "device_compatibility": study["device_compatibility"],
            "peripheral_requirements": study.get("peripheral_requirements", []),
            "estimated_completion_time": study["estimated_completion_time"],
            "maximum_allowed_time": study["maximum_allowed_time"],
            "content_warnings": study.get("content_warnings", []),
            "pii": study.get("pii", {"enabled": False}),
            "is_custom_screening": study.get("is_custom_screening", False),
            "filters": study["filters"],
            "study_type": study["study_type"],
            "completion_codes": study["completion_codes"],
            "data_collection_method": study["data_collection_method"],
            "data_collection_id": study["data_collection_id"],
            "access_details": access_details,
            "access_details_collection_id": study["access_details_collection_id"],
            "submissions_config": study["submissions_config"],
        }

        resp = requests.patch(
            f"{self.BASE_URL}/studies/{study_id}/",
            headers=self.headers,
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        print(f"✅ Updated study participants")

    def publish_study(self, study_id: str):
        """Publish study."""
        resp = requests.post(
            f"{self.BASE_URL}/studies/{study_id}/transition/",
            headers=self.headers,
            data=json.dumps({"action": "PUBLISH"}),
        )
        resp.raise_for_status()
        print(f"✅ Published study")

    def wait_for_study_completion(self, study_id: str, timeout_sec: int = 21600) -> bool:
        """Wait for study to complete (default 6 hours)."""
        t0 = time.time()

        while True:
            resp = requests.get(f"{self.BASE_URL}/studies/{study_id}/", headers=self.headers)
            resp.raise_for_status()
            study = resp.json()
            status = study.get("status")
            places_taken = study.get("places_taken", 0)
            total_places = study.get("total_available_places", 0)

            print(f"Study status: {status} ({places_taken}/{total_places} places filled)")

            if status in ("COMPLETED", "FINISHED", "AWAITING_REVIEW", "CLOSED"):
                print("✅ Study completed!")
                return True

            if time.time() - t0 > timeout_sec:
                print("⏰ Timeout")
                return False
            time.sleep(60)

    def fetch_batch_responses(self, batch_id: str) -> pd.DataFrame:
        """Fetch responses from batch."""
        resp = requests.get(
            f"{self.BASE_URL}/data-collection/batches/{batch_id}/report",
            headers=self.headers,
        )
        resp.raise_for_status()
        report_url = resp.json()["url"]

        resp = requests.get(report_url)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))
        print(f"✅ Fetched {len(df)} responses")
        return df

    def fetch_study_demographics(self, study_id: str) -> pd.DataFrame:
        """Fetch participant demographics."""
        resp = requests.get(f"{self.BASE_URL}/studies/{study_id}/export/", headers=self.headers)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))
        df = df[~df["Completed at"].isna()]
        print(f"✅ Fetched demographics for {len(df)} participants")
        return df
