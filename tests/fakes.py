from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from job_utils import Deadline, JobError, LLAMA_COMMIT
from run_job import Endpoint, RunState, SSHTransport
from vast_client import ApiError, DEFAULT_IMAGE, DEFAULT_TEMPLATE_HASH, Offer, Template, VastClient


VAST_TOKEN = "VAST_TEST_MUST_NOT_LEAK"
HF_TOKEN = "HF_TEST_MUST_NOT_LEAK"
ANSWER = 'A precise response with "quotes", a newline,\nand caf\u00e9.'
MODEL = {
    "repo": "unit/model", "filename": "model-Q4_K_M.gguf",
    "revision": "a" * 40, "size_bytes": 16, "sha256": None,
}


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def deadline(self, seconds: float) -> Deadline:
        return Deadline(seconds, clock=self, sleeper=self.sleep)


def raw_offer(offer_id: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "id": offer_id, "ask_contract_id": offer_id,
        "gpu_name": "A100 SXM4", "gpu_ram": 40960, "gpu_arch": "nvidia",
        "num_gpus": 1, "disk_space": 100, "cpu_arch": "amd64", "cpu_ram": 64000,
        "has_avx": 1, "cuda_max_good": 12.4, "compute_cap": 800,
        "duration": 100_000, "reliability": 0.999,
        "verification": "verified", "rentable": True, "rented": False,
        "dph_total": 1.0, "direct_port_count": 2,
        "inet_down_cost": 0.01, "inet_up_cost": 0.02,
    }
    row.update(overrides)
    return row


class Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, *responses: object) -> None:
        self.headers: dict[str, str] = {}
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Response:
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, Response):
            raise AssertionError("Invalid fixture response")
        return response

    def close(self) -> None:
        pass


class FakeClient(VastClient):
    def __init__(self, api_key: str = VAST_TOKEN, *, fail: str = "") -> None:
        self.fail = fail
        self.exists = False
        self.label = ""
        self.events: list[str] = []
        self.created = 0
        self.destroyed = 0
        self.closed = False
        self.clock: Clock | None = None
        self.offer = Offer(10, "A100 SXM4", 40960, 1, 2, 0.01, 0.02)

    def template(self, hash_id: str, deadline: Deadline) -> Template:
        return Template(hash_id, "Fixture CUDA template", DEFAULT_IMAGE, 12.4, {}, 32)

    def offers(self, template: Template, disk_gb: int, max_price: float,
               rental_seconds: float, deadline: Deadline, *,
               offer_id: int | None = None) -> list[Offer]:
        self.events.append("revalidate" if offer_id else "offers")
        return [] if self.fail == "no-offers" else [self.offer]

    def create(self, offer: Offer, template: Template, disk_gb: int,
               label: str, deadline: Deadline) -> int:
        self.events.append("create")
        self.created += 1
        self.label = label
        if self.fail == "rejected":
            raise ApiError("Fixture rental rejected.", status=400)
        self.exists = self.fail != "ambiguous-empty"
        if self.fail in ("ambiguous", "ambiguous-empty"):
            raise ApiError("Fixture response was lost.", ambiguous=True)
        if self.clock is not None:
            self.clock.sleep(20)
        return 101

    def attach_key(self, instance_id: int, public_key: str, deadline: Deadline) -> None:
        self.events.append(f"attach:{instance_id}")
        if not public_key.startswith("ssh-ed25519 "):
            raise AssertionError("Not an instance public key")
        if self.fail == "attach":
            raise JobError("Fixture key attachment failure.")

    def instance(self, instance_id: int, deadline: Deadline) -> dict[str, Any] | None:
        self.events.append(f"instance:{instance_id}")
        if not self.exists:
            return None
        return {
            "id": 101, "label": self.label,
            "actual_status": "offline" if self.fail == "offline" else "running",
            "ssh_host": "unit.invalid", "ssh_port": 2222,
        }

    def labeled_instances(self, label: str, deadline: Deadline) -> list[dict[str, Any]]:
        self.events.append("reconcile")
        return [{"id": 101, "label": self.label}] if self.exists and label == self.label else []

    def destroy(self, instance_id: int, deadline: Deadline) -> None:
        self.events.append(f"destroy:{instance_id}")
        self.destroyed += 1
        if self.fail == "cleanup":
            raise ApiError("Fixture deletion refused.", status=403)
        if instance_id != 101:
            raise AssertionError("Deleted unrelated instance")
        self.exists = False

    def close(self) -> None:
        self.closed = True


class FakeTransport(SSHTransport):
    def __init__(self, state: RunState, endpoint: Endpoint, client: FakeClient,
                 *, fail: str = "", finish_reason: str = "stop") -> None:
        super().__init__(state, endpoint)
        self.client = client
        self.fail = fail
        self.finish_reason = finish_reason
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.files = {"remote.log": f"Fixture log {HF_TOKEN} Bearer {VAST_TOKEN}\n".encode()}

    def probe(self, deadline: Deadline) -> None:
        self.client.events.append("ssh-ready")

    def upload(self, deadline: Deadline) -> None:
        self.client.events.append("upload")
        if self.fail == "upload":
            raise JobError("Fixture upload failed.")

    def invoke(self, stage: str, request: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
        self.requests.append((stage, request))
        self.client.events.append(stage)
        if self.fail == stage or (self.fail in ("download", "cancel") and stage == "infer"):
            raise JobError(f"Fixture {self.fail} failure at https://unit.invalid/api?token={HF_TOKEN}")
        result = {
            "status": "ready" if stage == "bootstrap" else "completed",
            "llama_commit": LLAMA_COMMIT, "gpu_name": "NVIDIA A100-SXM4-40GB",
            "gpu_vram_mib": 40960,
        }
        if stage == "infer":
            completion = {
                "choices": [{"message": {"role": "assistant", "content": ANSWER},
                             "finish_reason": self.finish_reason}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30},
            }
            if self.fail == "completion":
                completion["choices"][0]["message"]["role"] = "user"
            self.files["response.txt"] = ANSWER.encode()
            self.files["completion.json"] = (json.dumps(completion) + "\n").encode()
            result.update({
                "response_sha256": hashlib.sha256(self.files["response.txt"]).hexdigest(),
                "completion_sha256": hashlib.sha256(self.files["completion.json"]).hexdigest(),
            })
        return result

    def fetch(self, filename: str, destination: Path, deadline: Deadline) -> None:
        self.client.events.append(f"fetch:{filename}")
        if self.fail == "transfer" and filename == "response.txt":
            destination.write_bytes(b"partial")
            raise JobError("Fixture transfer failed.")
        content = self.files[filename]
        if self.fail == "checksum" and filename == "response.txt":
            content += b"corrupt"
        destination.write_bytes(content)


class FakeAgentTransport(FakeTransport):
    def upload_agent(self, inputs: Path | None, deadline: Deadline) -> None:
        self.client.events.append("upload-agent")
        if self.fail == "upload-agent":
            raise JobError("Fixture agent upload failed.")
        self.input_archive = inputs

    def invoke(self, stage: str, request: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
        if stage != "agent":
            return super().invoke(stage, request, deadline)
        self.requests.append((stage, request))
        self.client.events.append(stage)
        if self.fail == "agent":
            raise JobError("Fixture agent transport failed.")
        from agent_utils import extract_archive, pack_directory

        workspace = self.state.directory / "fixture-workspace"
        workspace.mkdir()
        if request.get("inputs") is not None:
            extract_archive(
                self.input_archive, workspace, request["inputs"]["manifest"],
                deadline, maximum=64 * 1024 ** 2,
            )
        (workspace / "scripts").mkdir()
        (workspace / "scripts/task.py").write_text("print(42)\n")
        (workspace / "result.json").write_text('{"value":42}\n')
        archive = self.state.directory / "fixture-artifacts.tar"
        info = pack_directory(workspace, archive, deadline, maximum=64 * 1024 ** 2)
        transcript = {
            "version": 1, "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
            "steps": [
                {"step": 1, "action": {"action": "write_file", "path": "scripts/task.py", "content": "print(42)\n"}},
                {"step": 2, "action": {"action": "run", "argv": ["python3", "scripts/task.py"]},
                 "observation": {"exit_code": 0, "stdout": "42\n"}},
            ],
        }
        failed = self.fail == "agent-failed"
        if not failed:
            transcript["steps"].append({"step": 3, "action": {"action": "finish", "summary": ANSWER}})
        self.files["agent-transcript.json"] = (json.dumps(transcript) + "\n").encode()
        self.files["artifacts.tar"] = archive.read_bytes()
        report = {
            "status": "failed" if failed else "completed", "llama_commit": LLAMA_COMMIT,
            "gpu_name": "NVIDIA A100-SXM4-40GB", "gpu_vram_mib": 40960,
            "steps": len(transcript["steps"]), "max_steps": request["agent"]["max_steps"],
            "environment_names": sorted(request["task_env"]), "archive": info,
            "final_response": None if failed else ANSWER,
            "transcript_sha256": hashlib.sha256(self.files["agent-transcript.json"]).hexdigest(),
        }
        if failed:
            report["error"] = "Fixture agent exhausted its task deadline."
        else:
            self.files["response.txt"] = ANSWER.encode()
            report["response_sha256"] = hashlib.sha256(self.files["response.txt"]).hexdigest()
        self.files["agent-report.json"] = (json.dumps(report) + "\n").encode()
        return {**report, "agent_report_sha256": hashlib.sha256(self.files["agent-report.json"]).hexdigest()}

    def fetch(self, filename: str, destination: Path, deadline: Deadline) -> None:
        if filename not in self.files:
            raise JobError(f"Fixture remote artifact is unavailable: {filename}.")
        if self.fail == "agent-archive" and filename == "artifacts.tar":
            self.client.events.append(f"fetch:{filename}")
            destination.write_bytes(b"corrupt archive")
            return
        super().fetch(filename, destination, deadline)
