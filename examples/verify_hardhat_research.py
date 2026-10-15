from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


POOL = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0x6b175474e89094c44da98b954eedeac495271d0f",
    "0x514910771af9ca656af840dff83e8264ecf986ca",
}
PINS = {
    "hardhat": "2.29.1",
    "@nomicfoundation/hardhat-ethers": "3.1.3",
    "ethers": "6.17.0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def request_json(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "User-Agent": "VastAI-Hardhat-verifier"},
    )
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=15) as response:
        require(response.status == 200, "API HTTP request did not succeed.")
        content = response.read(2 * 1024 ** 2 + 1)
    require(len(content) <= 2 * 1024 ** 2, "API response exceeds the verifier limit.")
    result = json.loads(content)
    require(isinstance(result, dict), "API did not return a JSON object.")
    return result


def artifact(root: Path, name: object) -> Path:
    require(isinstance(name, str) and bool(name) and "\\" not in name, "Invalid artifact path.")
    path = PurePosixPath(name)
    require(
        not path.is_absolute() and path.as_posix() == name
        and all(part not in ("", ".", "..") for part in name.split("/")),
        "Artifact paths must be relative without traversal.",
    )
    candidate = root / name
    require(
        candidate.resolve() == candidate and candidate.is_file() and not candidate.is_symlink(),
        "Required artifact is missing or linked: " + name,
    )
    return candidate


def verify_report_section(summary: str, address: str, contract_name: str, source: str) -> int:
    headings = list(re.finditer(r"(?m)^##[ \t]+[^\n]+", summary))
    selected = [index for index, heading in enumerate(headings) if address.lower() in heading[0].lower()]
    require(len(selected) == 1, f"Report needs one ## {contract_name} ({address}) section.")
    index = selected[0]
    require(contract_name.lower() in headings[index][0].lower(), "Report section needs the verified contract name.")
    end = headings[index + 1].start() if index + 1 < len(headings) else len(summary)
    section = summary[headings[index].end():end]
    masked = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
        lambda match: " " * len(match[0]), source, flags=re.DOTALL,
    )
    declarations: dict[str, tuple[str, set[str], bool]] = {}
    for match in re.finditer(r"\bfunction\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*([^;{}]*?)\{", masked):
        name, tail = match[1], match[2]
        modifiers = set(re.findall(
            r"[A-Za-z_]\w*", re.sub(r"\([^)]*\)", "", tail),
        )) - {"public", "external", "private", "internal", "view", "pure", "constant", "payable", "returns"}
        declarations[name] = (
            source[match.start():match.end() - 1].strip(), modifiers,
            not re.search(r"\b(?:private|internal)\b", tail) and name != contract_name,
        )
    require(bool(declarations), "Verified source contains no named function definitions.")
    header = ["function", "modifiers", "caller conditions", "source declaration"]
    active = False
    tables = 0
    observed: set[str] = set()
    for line in section.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            active = False
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
        if [cell.lower() for cell in cells] == header:
            active = True
            tables += 1
            continue
        if not active:
            require(not cells or cells[0].lower() != "function", "Use the exact four-column source table header.")
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        require(len(cells) == 4, "Source table needs four columns; escape literal pipes as \\|.")
        name = cells[0].strip("`")
        require(name in declarations, f"{contract_name}: reported function {name} does not exist in verified source.")
        require(name not in observed, f"{contract_name}: duplicate source table row for {name}.")
        declaration, modifiers, _ = declarations[name]
        quote = re.fullmatch(r"`([^`]+)`", cells[3])
        require(quote is not None, f"{contract_name}.{name}: quote the actual source declaration in backticks.")
        require(
            re.sub(r"\s+", "", quote[1].rstrip().removesuffix("{")) == re.sub(r"\s+", "", declaration),
            f"{contract_name}.{name}: declaration differs from verified source; quote {declaration!r}.",
        )
        claimed = set(re.split(r"[\s,]+", cells[1].replace("`", "").strip()))
        if claimed == {"none"}:
            claimed = set()
        require(
            claimed == modifiers,
            f"{contract_name}.{name}: named modifiers must be {', '.join(sorted(modifiers)) or 'none'}, "
            "not an assumed owner/auth restriction.",
        )
        require(bool(cells[2]), f"{contract_name}.{name}: explain the actual caller conditions.")
        observed.add(name)
    required = {name for name, (_, _, public) in declarations.items() if public}
    require(tables == 1 and required <= observed,
            f"{contract_name}: source table must cover every named public/external function; "
            f"missing {', '.join(sorted(required - observed)) or 'a single correctly headed table'}.")
    return len(observed)


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    for name in (
        "hardhat.config.cjs", "scripts/read-balances.cjs", "scripts/fetch-sources.cjs",
        "package.json", "package-lock.json", "report.md",
    ):
        require(artifact(root, name).stat().st_size > 0, "Required artifact is empty: " + name)
    package = json.loads(artifact(root, "package.json").read_text())
    dependencies = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    require(all(dependencies.get(name) == version for name, version in PINS.items()),
            "Use the three exact dependency pins in the task.")
    lock = json.loads(artifact(root, "package-lock.json").read_text())
    packages = lock.get("packages")
    require(isinstance(packages, dict) and all(
        isinstance(packages.get("node_modules/" + name), dict)
        and packages["node_modules/" + name].get("version") == version
        for name, version in PINS.items()
    ), "Installed dependency lockfile does not match the three task pins.")
    balances = json.loads(artifact(root, "outputs/balances.json").read_text())
    sources = json.loads(artifact(root, "outputs/contracts.json").read_text())
    require(balances.get("source_chain_id") == 1 and balances.get("fork_chain_id") == 31337,
            "Balances must come from a mainnet-sourced Hardhat fork.")
    block = balances.get("block_number")
    require(isinstance(block, int) and not isinstance(block, bool) and block > 0, "Invalid fork block.")
    require(sources.get("chain_id") == 1 and sources.get("block_number") == block,
            "Bytecode/source block does not match the fork.")
    selected = balances.get("contracts")
    require(isinstance(selected, list) and len(selected) == 2, "Sample exactly two contracts.")
    require(all(isinstance(row, dict) and isinstance(row.get("address"), str) for row in selected),
            "Invalid contract sample rows.")
    addresses = {row["address"].lower() for row in selected}
    require(len(addresses) == 2 and addresses <= POOL, "Sample two different contracts from the task pool.")
    entries = sources.get("contracts")
    require(isinstance(entries, list) and len(entries) == 2, "Both contracts need verified source records.")
    require(all(isinstance(row, dict) and isinstance(row.get("address"), str) for row in entries),
            "Invalid source records.")
    source_rows = {row["address"].lower(): row for row in entries}
    require(set(source_rows) == addresses, "Source records do not match the sampled contracts.")
    endpoint = "https://eth-mainnet.g.alchemy.com/v2/" + os.environ["ALCHEMY_API_KEY"]

    def rpc(method: str, params: list[object]) -> Any:
        response = request_json(endpoint, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        require("error" not in response and "result" in response, "Alchemy " + method + " did not succeed.")
        return response["result"]

    require(rpc("eth_chainId", []) == "0x1", "Alchemy must reference Ethereum mainnet.")
    summary = artifact(root, "report.md").read_text()
    verified = []
    for row in selected:
        address = row["address"]
        source = source_rows[address.lower()]
        require(
            isinstance(row.get("balance_wei"), str)
            and re.fullmatch(r"0|[1-9][0-9]*", row["balance_wei"]) is not None,
            "balance_wei must be a decimal wei string, not RPC hex or a Number; use BigInt(value).toString().",
        )
        require(
            isinstance(row.get("balance_eth"), str)
            and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", row["balance_eth"]) is not None,
            "balance_eth must be a formatted decimal string; use ethers.formatEther(balance), not Number arithmetic.",
        )
        code = rpc("eth_getCode", [address, hex(block)])
        balance = rpc("eth_getBalance", [address, hex(block)])
        require(isinstance(code, str) and re.fullmatch(r"0x(?:[0-9a-fA-F]{2})+", code) is not None,
                "Alchemy returned empty or malformed deployed code.")
        require(isinstance(balance, str) and re.fullmatch(r"0x[0-9a-fA-F]+", balance) is not None,
                "Alchemy returned a malformed balance.")
        code_bytes = bytes.fromhex(code[2:])
        require(row.get("fork_code_bytes") == len(code_bytes),
                "fork_code_bytes must count decoded bytes, not hex characters: (code.length - 2) / 2.")
        require(row.get("balance_wei") == str(int(balance, 16)), "Fork balance differs from mainnet at its block.")
        require(Decimal(row["balance_eth"]) * 10 ** 18 == Decimal(row["balance_wei"]), "Incorrect ETH formatting.")
        require(isinstance(source.get("runtime_bytecode"), str) and source["runtime_bytecode"].lower() == code.lower(),
                "Saved bytecode differs from Alchemy at the recorded block.")
        code_hash = hashlib.sha256(code_bytes).hexdigest()
        require(source.get("runtime_bytecode_sha256") == code_hash, "Incorrect runtime bytecode hash.")
        saved_source_bytes = artifact(root, source.get("source_path")).read_bytes()
        saved_source = saved_source_bytes.decode("utf-8")
        source_hash = hashlib.sha256(saved_source_bytes).hexdigest()
        require(source.get("source_sha256") == source_hash, "Incorrect saved-source hash.")
        saved_response = json.loads(artifact(root, source.get("etherscan_response_path")).read_text())
        query = urllib.parse.urlencode({
            "chainid": 1, "module": "contract", "action": "getsourcecode",
            "address": address, "apikey": os.environ["ETHERSCAN_API_KEY"],
        })
        fresh = request_json("https://api.etherscan.io/v2/api?" + query)
        require(fresh.get("status") == "1",
                "Fresh Etherscan V2 did not provide verified source. Keep /v2/api and include chainid=1.")
        require(
            isinstance(saved_response, dict) and saved_response.get("status") == "1"
            and isinstance(saved_response.get("result"), list) and bool(saved_response["result"])
            and isinstance(saved_response["result"][0], dict),
            "Saved etherscan.json must contain the complete API response with status and result. "
            "Save JSON.stringify(etherscanJson), not etherscanJson.result[0]; rerun the source script.",
        )
        current = fresh["result"][0]
        stored = saved_response["result"][0]
        require(current["SourceCode"] == stored["SourceCode"] and bool(current["SourceCode"].strip()),
                "Saved Etherscan source differs from the independent response.")
        require(current["ContractName"] == stored["ContractName"] == source.get("contract_name"),
                "Verified contract name does not match.")
        original = current["SourceCode"].strip()
        if original.startswith("{{"):
            original = original[1:-1]
        contents = (
            [item["content"] for item in json.loads(original)["sources"].values()]
            if original.startswith("{") else [original]
        )
        require(bool(contents) and all(
            content.replace("\r\n", "\n").strip() in saved_source.replace("\r\n", "\n")
            for content in contents
        ), "Saved Solidity must preserve the actual verified source and license notices, not placeholders.")
        functions = set(re.findall(r"\bfunction\s+([A-Za-z_]\w*)\s*\(", saved_source))
        require(address.lower() in summary.lower() and source["source_path"] in summary,
                "Report must cite each selected address and its saved Solidity path.")
        require(any(function in summary for function in functions), "Report cites no observed contract function.")
        # Agent-writable source additions must not change the declaration checks.
        report_functions = verify_report_section(summary, address, current["ContractName"], "\n".join(contents))
        verified.append({
            "address": address, "contract_name": current["ContractName"], "balance_wei": row["balance_wei"],
            "runtime_code_bytes": len(code_bytes), "runtime_bytecode_sha256": code_hash,
            "source_sha256": source_hash, "report_function_count": report_functions,
        })
    return {"status": "verified", "block_number": block, "contracts": verified}


def main() -> int:
    try:
        print(json.dumps(verify(Path.cwd()), indent=2))
        return 0
    except urllib.error.URLError as error:
        print("Verification failed: API transport error (" + type(error).__name__ + ").", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, IndexError, TypeError, InvalidOperation) as error:
        print("Verification failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
