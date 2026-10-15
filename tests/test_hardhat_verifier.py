import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

from examples.verify_hardhat_research import PINS, verify


class HardhatVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.addresses = (
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        )
        self.code = "0x606000"
        self.source = "// SPDX-License-Identifier: MIT\ncontract Example { function transfer() public {} }\n"
        self.original = self.source
        self.seed()
        self.environment = patch.dict(os.environ, {
            "ALCHEMY_API_KEY": "FAKE_ALCHEMY_TEST_KEY", "ETHERSCAN_API_KEY": "FAKE_ETHERSCAN_TEST_KEY",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.network = patch("examples.verify_hardhat_research.request_json", side_effect=self.response)
        self.request = self.network.start()
        self.addCleanup(self.network.stop)

    def save(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def seed(self):
        (self.root / "scripts").mkdir(exist_ok=True)
        for name in ("hardhat.config.cjs", "scripts/read-balances.cjs", "scripts/fetch-sources.cjs"):
            (self.root / name).write_text("// test fixture\n")
        self.save("package.json", {"devDependencies": PINS})
        self.save("package-lock.json", {"lockfileVersion": 3, "packages": {
            "node_modules/" + name: {"version": version} for name, version in PINS.items()
        }})
        self.balances = {
            "source_chain_id": 1, "fork_chain_id": 31337, "block_number": 123,
            "contracts": [
                {"address": address, "balance_wei": "100", "balance_eth": "0.0000000000000001",
                 "fork_code_bytes": 3} for address in self.addresses
            ],
        }
        self.save("outputs/balances.json", self.balances)
        self.contracts = {"chain_id": 1, "block_number": 123, "contracts": []}
        report = []
        for address in self.addresses:
            source_path = f"sources/{address}/contract.sol"
            response_path = f"sources/{address}/etherscan.json"
            self.save(response_path, {"status": "1", "result": [{"SourceCode": self.original, "ContractName": "Example"}]})
            (self.root / source_path).write_text(self.source)
            self.contracts["contracts"].append({
                "address": address, "runtime_bytecode": self.code,
                "runtime_bytecode_sha256": hashlib.sha256(bytes.fromhex(self.code[2:])).hexdigest(),
                "contract_name": "Example", "source_path": source_path,
                "source_sha256": hashlib.sha256(self.source.encode()).hexdigest(),
                "etherscan_response_path": response_path,
            })
            report.append(f"{address}: transfer() in {source_path}")
        self.save("outputs/contracts.json", self.contracts)
        (self.root / "report.md").write_text("\n".join(report))

    def response(self, url, body=None):
        if body is not None:
            self.assertEqual(url, "https://eth-mainnet.g.alchemy.com/v2/FAKE_ALCHEMY_TEST_KEY")
            self.assertEqual(body["jsonrpc"], "2.0")
            if body["method"] == "eth_chainId":
                return {"result": "0x1"}
            self.assertIn(body["params"][0], self.addresses)
            self.assertEqual(body["params"][1], hex(123))
            self.assertIn(body["method"], ("eth_getCode", "eth_getBalance"))
            return {"result": self.code if body["method"] == "eth_getCode" else "0x64"}
        parsed = urllib.parse.urlsplit(url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "api.etherscan.io")
        self.assertEqual(parsed.path, "/v2/api")
        query = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(query["chainid"], ["1"])
        self.assertEqual(query["module"], ["contract"])
        self.assertEqual(query["action"], ["getsourcecode"])
        self.assertEqual(query["apikey"], ["FAKE_ETHERSCAN_TEST_KEY"])
        self.assertIn(query["address"][0], self.addresses)
        return {"status": "1", "result": [{"SourceCode": self.original, "ContractName": "Example"}]}

    def test_positive_control_checks_exact_bytes_balances_sources_and_both_api_contracts(self):
        result = verify(self.root)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["contracts"]), 2)
        self.assertEqual({row["runtime_code_bytes"] for row in result["contracts"]}, {3})
        self.assertEqual(self.request.call_count, 7)

    def test_hex_character_length_and_incorrect_balance_are_rejected(self):
        for field, incorrect in (("fork_code_bytes", len(self.code)), ("balance_wei", "101")):
            with self.subTest(field=field):
                self.seed()
                self.balances["contracts"][0][field] = incorrect
                self.save("outputs/balances.json", self.balances)
                with self.assertRaises(ValueError):
                    verify(self.root)

    def test_rpc_hex_balance_reports_the_required_decimal_conversion(self):
        self.balances["contracts"][0]["balance_wei"] = "0x64"
        self.save("outputs/balances.json", self.balances)
        with self.assertRaisesRegex(ValueError, "decimal wei string.*BigInt"):
            verify(self.root)
        self.assertEqual(self.request.call_count, 1)

    def test_even_exact_numeric_eth_values_require_formatted_strings(self):
        first = self.balances["contracts"][0]
        first["balance_wei"] = "0"
        first["balance_eth"] = 0
        self.save("outputs/balances.json", self.balances)

        def zero_balance(url, body=None):
            if body and body.get("method") == "eth_getBalance" and body["params"][0] == first["address"]:
                return {"result": "0x0"}
            return self.response(url, body)

        self.request.side_effect = zero_balance
        with self.assertRaises(ValueError):
            verify(self.root)
        first["balance_eth"] = "0.0"
        self.save("outputs/balances.json", self.balances)
        self.assertEqual(verify(self.root)["status"], "verified")

    def test_placeholder_solidity_is_rejected_even_when_its_declared_hash_is_correct(self):
        path = self.root / self.contracts["contracts"][0]["source_path"]
        path.write_text("contract Placeholder { function transfer() public {} }\n")
        self.contracts["contracts"][0]["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save("outputs/contracts.json", self.contracts)
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_double_braced_verified_sources_preserve_all_contents_and_licenses(self):
        extra = "// SPDX-License-Identifier: MIT\nlibrary Helpers {}\n"
        self.original = "{" + json.dumps({"sources": {
            "Example.sol": {"content": self.source}, "lib/Helpers.sol": {"content": extra},
        }}) + "}"
        self.source += "// Source: lib/Helpers.sol\n" + extra
        self.seed()
        self.assertEqual(verify(self.root)["status"], "verified")
        path = self.root / self.contracts["contracts"][0]["source_path"]
        path.write_text(self.source.replace(extra, ""))
        self.contracts["contracts"][0]["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save("outputs/contracts.json", self.contracts)
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_missing_raw_verified_response_and_api_failure_are_not_source_success(self):
        path = self.root / self.contracts["contracts"][0]["etherscan_response_path"]
        path.unlink()
        with self.assertRaises(ValueError):
            verify(self.root)
        self.seed()

        def failed(url, body=None):
            return self.response(url, body) if body is not None else {"status": "0", "result": "API error"}

        self.request.side_effect = failed
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_exact_declarations_do_not_hide_a_different_installed_dependency_lock(self):
        path = self.root / "package-lock.json"
        lock = json.loads(path.read_text())
        lock["packages"]["node_modules/hardhat"]["version"] = "2.0.0"
        path.write_text(json.dumps(lock))
        with self.assertRaises(ValueError):
            verify(self.root)
        self.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
