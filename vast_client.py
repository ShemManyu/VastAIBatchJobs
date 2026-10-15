from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import requests

from job_utils import Deadline, JobError, number, object_value, positive_int, redact


BASE_URL = "https://console.vast.ai"
DEFAULT_TEMPLATE_HASH = "661d064bbda1f2a133816b6d55da07c3"
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
PAGE_SIZE = 100
MIN_GPU_RAM_MB = 24_000


class ApiError(JobError):
    def __init__(
        self, message: str, *, status: int | None = None, retryable: bool = False,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.ambiguous = ambiguous


@dataclass(frozen=True)
class Template:
    hash_id: str
    name: str
    image: str
    cuda_version: float
    filters: dict[str, Any]
    recommended_disk: float


@dataclass(frozen=True)
class Offer:
    id: int
    gpu_name: str
    gpu_ram_mb: float
    hourly_price: float
    direct_ports: int
    download_cost_per_gb: float
    upload_cost_per_gb: float
    gpu_count: int = 1

    @property
    def a100(self) -> bool:
        return bool(re.match(r"^A100(?:\b|_)", self.gpu_name, re.IGNORECASE))

    def summary(self) -> dict[str, Any]:
        return {
            "offer_id": self.id,
            "gpu_name": self.gpu_name,
            "gpu_ram_mb": self.gpu_ram_mb,
            "gpu_count": self.gpu_count,
            "total_gpu_ram_mb": self.gpu_ram_mb * self.gpu_count,
            "hourly_price_usd": self.hourly_price,
            "download_cost_per_gb_usd": self.download_cost_per_gb,
            "upload_cost_per_gb_usd": self.upload_cost_per_gb,
        }


def merge_filters(
    template_filters: dict[str, Any], required: dict[str, Any],
) -> dict[str, Any]:
    result = {key: dict(object_value(value, f"Template filter {key}"))
              for key, value in template_filters.items()}
    for key, conditions in required.items():
        current = result.setdefault(key, {})
        for operator, value in conditions.items():
            if operator not in current:
                current[operator] = value
            elif operator in ("gte", "gt"):
                current[operator] = max(
                    number(current[operator], f"Template filter {key}"),
                    number(value, f"Required filter {key}"),
                )
            elif operator in ("lte", "lt"):
                current[operator] = min(
                    number(current[operator], f"Template filter {key}"),
                    number(value, f"Required filter {key}"),
                )
            elif current[operator] != value:
                raise JobError(f"Template has conflicting {key} constraints.")
    return result


def matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, conditions in filters.items():
        value = row.get(field)
        if field == "verified":
            value = row.get("verified")
            if "verification" in row:
                value = row["verification"] == "verified" and value in (None, True, 1)
        if value is None:
            raise JobError(f"VastAI offer is missing the required field {field}.")
        for operator, bound in object_value(conditions, f"Filter {field}").items():
            if operator == "eq":
                matches = value == bound
            elif operator == "neq":
                matches = value != bound
            elif operator in ("in", "notin"):
                if not isinstance(bound, list):
                    raise JobError(f"Template filter {field} has an invalid list.")
                matches = value in bound if operator == "in" else value not in bound
            elif operator in ("gte", "gt", "lte", "lt"):
                left = number(value, f"Offer {field}")
                right = number(bound, f"Filter {field}")
                matches = {
                    "gte": left >= right, "gt": left > right,
                    "lte": left <= right, "lt": left < right,
                }[operator]
            else:
                raise JobError(f"Unsupported template filter operator: {operator}.")
            if not matches:
                return False
    return True


def choose_offer(offers: list[Offer], policy: str, max_price: float) -> Offer:
    eligible = [offer for offer in offers
                if offer.gpu_ram_mb >= MIN_GPU_RAM_MB
                and 0 <= offer.hourly_price <= max_price]
    if policy == "a100-only":
        eligible = [offer for offer in eligible if offer.a100]
    elif policy != "prefer-a100":
        raise JobError(f"Unsupported GPU policy: {policy}.")
    if not eligible:
        raise JobError(
            f"No compatible {policy} offer with >=24 GB VRAM is available "
            f"within ${max_price:.2f}/hour including the requested storage."
        )
    return min(eligible, key=lambda offer: (not offer.a100, offer.hourly_price, offer.id))


class VastClient:
    def __init__(self, api_key: str, *, session: requests.Session | None = None) -> None:
        if not api_key.strip():
            raise JobError("VAST_AI_API_KEY is missing.")
        self.secret = api_key
        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })

    def close(self) -> None:
        self.session.close()

    def _request(
        self,
        method: str,
        path: str,
        deadline: Deadline,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        absent_ok: bool = False,
        create: bool = False,
    ) -> dict[str, Any] | None:
        attempts = 1 if create else 3
        for attempt in range(attempts):
            timeout = deadline.timeout()
            try:
                response = self.session.request(
                    method, BASE_URL + path, json=body, params=params,
                    timeout=(timeout / 2, timeout / 2), allow_redirects=False,
                )
            except requests.RequestException as error:
                failure = ApiError(
                    f"VastAI {method} {path} could not connect ({type(error).__name__}).",
                    retryable=True, ambiguous=create,
                )
            else:
                if response.status_code == 404 and absent_ok:
                    return None
                try:
                    payload = object_value(response.json(), "VastAI response")
                except (ValueError, JobError) as error:
                    failure = ApiError(
                        f"VastAI {method} {path} returned invalid JSON "
                        f"(HTTP {response.status_code}).",
                        status=response.status_code,
                        retryable=response.status_code == 429 or response.status_code >= 500,
                        ambiguous=create,
                    )
                    failure.__cause__ = error
                else:
                    if 200 <= response.status_code < 300 and payload.get("success") is not False:
                        return payload
                    detail = str(payload.get("msg") or payload.get("error") or "Request rejected")
                    failure = ApiError(
                        redact(
                            f"VastAI {method} {path}: HTTP {response.status_code}, {detail[:800]}",
                            (self.secret,),
                        ),
                        status=response.status_code,
                        retryable=response.status_code == 429 or response.status_code >= 500,
                        ambiguous=create and response.status_code >= 500,
                    )
            if not failure.retryable or attempt == attempts - 1:
                raise failure
            deadline.sleep(2 ** attempt)
        raise JobError("VastAI request exhausted its attempts.")

    def template(self, hash_id: str, deadline: Deadline) -> Template:
        if not re.fullmatch(r"[a-fA-F0-9]{16,64}", hash_id):
            raise JobError("Template hash must be a hexadecimal VastAI template hash.")
        data = self._request(
            "GET", "/api/v0/template/", deadline,
            params={"select_filters": json.dumps({"hash_id": {"eq": hash_id}})},
        )
        if data is None or not isinstance(data.get("templates"), list):
            raise JobError("VastAI template lookup returned no template list.")
        rows = [object_value(row, "Template") for row in data["templates"]]
        rows = [row for row in rows if row.get("hash_id") == hash_id]
        if len(rows) != 1:
            raise JobError(f"VastAI template {hash_id} is missing or inaccessible.")
        row = rows[0]
        image, tag = row.get("image"), row.get("tag")
        if not isinstance(image, str) or not isinstance(tag, str):
            raise JobError("Template is missing its image or pinned tag.")
        image = f"{image}:{tag}"
        cuda = re.search(r"cuda(\d+\.\d+)", image, re.IGNORECASE)
        if "devel" not in image.lower() or cuda is None:
            raise JobError("Select a CUDA development template with a pinned cudaX.Y-devel image.")
        if hash_id == DEFAULT_TEMPLATE_HASH and image != DEFAULT_IMAGE:
            raise JobError("The default template image changed; review it or supply a new template hash.")
        raw_filters = row.get("extra_filters")
        try:
            filters = object_value(
                json.loads(raw_filters) if isinstance(raw_filters, str)
                else {} if raw_filters is None else raw_filters,
                "Template extra_filters",
            )
        except ValueError as error:
            raise JobError("Template extra_filters contains invalid JSON.") from error
        return Template(
            hash_id, str(row.get("name") or hash_id), image, float(cuda.group(1)), filters,
            number(row.get("recommended_disk_space"), "Template recommended disk"),
        )

    def offers(
        self, template: Template, disk_gb: int, max_price: float,
        rental_seconds: float, deadline: Deadline, *, offer_id: int | None = None,
        gpu_count: int = 1,
    ) -> list[Offer]:
        maximum_compute = 1200 if template.cuda_version >= 12.8 else 900
        filters = merge_filters(template.filters, {
            "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
            "num_gpus": {"eq": positive_int(gpu_count, "GPU count")}, "gpu_arch": {"eq": "nvidia"},
            "gpu_ram": {"gte": MIN_GPU_RAM_MB}, "disk_space": {"gte": disk_gb},
            "cpu_arch": {"eq": "amd64"}, "cpu_ram": {"gte": 16_000},
            "compute_cap": {"gte": 750, "lte": maximum_compute},
            "cuda_max_good": {"gte": template.cuda_version},
            "duration": {"gte": rental_seconds}, "reliability": {"gte": 0.98},
            "dph_total": {"gte": 0, "lte": max_price},
        })
        if offer_id is not None:
            # The bundles endpoint filters the rentable "id" as "ask_contract_id".
            filters["ask_contract_id"] = {"eq": positive_int(offer_id, "Offer ID")}
        result: list[Offer] = []
        cursor = 0
        while True:
            query = dict(filters)
            if cursor:
                query["ask_contract_id"] = {"gt": cursor}
            query.update({
                "allocated_storage": disk_gb, "type": "ondemand",
                "limit": PAGE_SIZE, "order": [["ask_contract_id", "asc"]],
            })
            data = self._request("POST", "/api/v0/bundles/", deadline, body=query)
            if data is None or not isinstance(data.get("offers"), list):
                raise JobError("VastAI offer search returned no offer list.")
            rows = data["offers"]
            for raw in rows:
                row = object_value(raw, "Offer")
                current = positive_int(row.get("id"), "Offer ID")
                if current <= cursor:
                    raise JobError("VastAI offer pagination did not advance.")
                cursor = current
                if not matches_filters(row, filters):
                    continue
                name = row.get("gpu_name")
                if not isinstance(name, str) or not name.strip():
                    raise JobError("VastAI offer is missing its GPU name.")
                direct_ports = number(row.get("direct_port_count"), "Direct ports")
                if direct_ports < 0 or not direct_ports.is_integer():
                    raise JobError("VastAI direct port count must be a non-negative integer.")
                result.append(Offer(
                    current, name, number(row["gpu_ram"], "GPU RAM"),
                    number(row["dph_total"], "Hourly price"), int(direct_ports),
                    number(row["inet_down_cost"], "Download bandwidth price"),
                    number(row["inet_up_cost"], "Upload bandwidth price"),
                    positive_int(row["num_gpus"], "Offer GPU count"),
                ))
            truncated = data.get("truncated")
            if truncated is not None and not isinstance(truncated, bool):
                raise JobError("VastAI returned an invalid truncation indicator.")
            if truncated is not True and len(rows) < PAGE_SIZE:
                break
            if not rows:
                raise JobError("VastAI returned a truncated empty offer page.")
            if offer_id is not None:
                raise JobError("VastAI could not uniquely revalidate the selected offer.")
        return result

    def create(
        self, offer: Offer, template: Template, disk_gb: int, label: str,
        deadline: Deadline,
    ) -> int:
        data = self._request("PUT", f"/api/v0/asks/{offer.id}/", deadline, create=True, body={
            "client_id": "me", "template_hash_id": template.hash_id, "image": template.image,
            "disk": disk_gb, "label": label, "vm": False,
            "runtype": "ssh_direct" if offer.direct_ports > 0 else "ssh_proxy",
            "target_state": "running", "cancel_unavail": True,
            "onstart": "", "env": {"TZ": "UTC"},
        })
        if data is None or data.get("success") is not True:
            raise ApiError("VastAI did not confirm instance creation.", ambiguous=True)
        try:
            return positive_int(data.get("new_contract"), "Created instance ID")
        except JobError as error:
            raise ApiError("VastAI created an instance without returning its ID.", ambiguous=True) from error

    def attach_key(self, instance_id: int, public_key: str, deadline: Deadline) -> None:
        data = self._request(
            "POST", f"/api/v0/instances/{positive_int(instance_id, 'Instance ID')}/ssh/",
            deadline, body={"ssh_key": public_key},
        )
        if data is None or data.get("success") is not True:
            raise JobError("VastAI did not confirm the instance-specific SSH key attachment.")

    def instance(self, instance_id: int, deadline: Deadline) -> dict[str, Any] | None:
        data = self._request(
            "GET", f"/api/v0/instances/{positive_int(instance_id, 'Instance ID')}/",
            deadline, params={"owner": "me"}, absent_ok=True,
        )
        if data is None or ("instances" in data and data["instances"] is None):
            return None
        if "instances" not in data:
            raise JobError("VastAI instance lookup returned no instance field.")
        row = object_value(data["instances"], "Instance")
        if row.get("id") != instance_id:
            raise JobError("VastAI returned the wrong instance ID.")
        return row

    def labeled_instances(self, label: str, deadline: Deadline) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        params: dict[str, Any] = {
            "limit": 25, "select_filters": json.dumps({"label": {"eq": label}}),
            "select_cols": json.dumps(["id", "label", "actual_status"]),
            "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
        }
        cursors: set[str] = set()
        while True:
            data = self._request("GET", "/api/v1/instances/", deadline, params=params)
            if data is None or not isinstance(data.get("instances"), list):
                raise JobError("VastAI reconciliation returned no instance list.")
            for raw in data["instances"]:
                row = object_value(raw, "Instance")
                positive_int(row.get("id"), "Instance ID")
                if row.get("label") != label:
                    raise JobError("VastAI reconciliation returned an unrelated instance.")
                result.append(row)
            token = data.get("next_token")
            if token is None:
                return result
            if not isinstance(token, str) or not token or token in cursors:
                raise JobError("VastAI instance pagination returned an invalid cursor.")
            cursors.add(token)
            params["after_token"] = token

    def destroy(self, instance_id: int, deadline: Deadline) -> None:
        data = self._request(
            "DELETE", f"/api/v0/instances/{positive_int(instance_id, 'Instance ID')}/",
            deadline, absent_ok=True,
        )
        if data is not None and data.get("success") is not True:
            raise JobError("VastAI did not confirm the destruction request.")
