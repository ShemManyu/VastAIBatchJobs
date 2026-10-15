import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

from agent_utils import MAX_VERIFICATION_BYTES
from examples.verify_hardhat_research import PINS, verify, verify_report_section


class SelfTransferConditionTests(unittest.TestCase):
    address = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

    def report(self, contract, name, modifiers, condition, declaration):
        return (
            f"## {contract} ({self.address})\n\n"
            "| Function | Modifiers | Caller conditions | Source declaration |\n"
            "| --- | --- | --- | --- |\n"
            f"| {name} | {modifiers} | {condition} | `{declaration}` |\n"
        )

    def test_actual_self_funded_calls_require_balance_but_never_recipient_allowance(self):
        for contract, name, recipient in (("Dai", "transfer", "dst"), ("Dai", "push", "usr"),
                                          ("WETH9", "transfer", "dst")):
            declaration = f"function {name}(address {recipient}, uint wad) public"
            source = declaration + f" {{ transferFrom(msg.sender, {recipient}, wad); }}"
            condition = "balanceOf[msg.sender] >= wad"
            with self.subTest(contract=contract, name=name):
                self.assertEqual(verify_report_section(
                    self.report(contract, name, "none", condition, declaration),
                    self.address, contract, source,
                ), 1)
                for invalid in (
                    "none", f"allowance[msg.sender][{recipient}] >= wad",
                    condition + f" and allowance[msg.sender][{recipient}] >= wad",
                ):
                    with self.assertRaisesRegex(ValueError, "self-funded transfer skips allowance"):
                        verify_report_section(
                            self.report(contract, name, "none", invalid, declaration),
                            self.address, contract, source,
                        )

    def test_other_contracts_delegated_calls_and_extra_modifiers_are_not_assumed_self_funded(self):
        for contract, modifiers, call in (
            ("Example", "none", "transferFrom(msg.sender, dst, wad)"),
            ("Dai", "none", "transferFrom(src, dst, wad)"),
            ("Dai", "auth", "transferFrom(msg.sender, dst, wad)"),
        ):
            declaration = "function transfer(address src, address dst, uint wad) public"
            if modifiers != "none":
                declaration += " " + modifiers
            source = declaration + " { return " + call + "; }"
            self.assertEqual(verify_report_section(
                self.report(contract, "transfer", modifiers, "Source-defined conditions", declaration),
                self.address, contract, source,
            ), 1)

    def test_published_verifier_still_fits_the_real_snapshot_and_argv_limit(self):
        path = Path(__file__).resolve().parents[1] / "examples/verify_hardhat_research.py"
        self.assertLessEqual(path.stat().st_size, MAX_VERIFICATION_BYTES)


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
        self.functions = [("transfer", "none", "function transfer() public")]
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
            report.append(
                f"## Example ({address})\nSource: {source_path}\n\n"
                "| Function | Modifiers | Caller conditions | Source declaration |\n"
                "| --- | --- | --- | --- |\n"
                + "\n".join(
                    f"| {name} | {modifiers} | Source-defined caller conditions | `{declaration}` |"
                    for name, modifiers, declaration in self.functions
                )
            )
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
        self.assertEqual({row["report_function_count"] for row in result["contracts"]}, {1})
        self.assertEqual(self.request.call_count, 7)

    def test_flat_source_record_gets_actionable_saved_response_feedback(self):
        path = self.root / self.contracts["contracts"][0]["etherscan_response_path"]
        payload = json.loads(path.read_text())
        path.write_text(json.dumps(payload["result"][0]))
        with self.assertRaisesRegex(ValueError, "complete API response.*JSON.stringify.*rerun"):
            verify(self.root)

    def test_live_source_api_failure_is_not_reported_as_a_saved_file_error(self):
        original = self.response

        def unavailable(url, body=None):
            return original(url, body) if body is not None else {"status": "0", "result": "NOTOK"}

        with patch("examples.verify_hardhat_research.request_json", side_effect=unavailable):
            with self.assertRaisesRegex(ValueError, "Fresh Etherscan V2"):
                verify(self.root)

    def test_invented_owner_function_is_rejected_despite_correct_chain_and_source_data(self):
        report = self.root / "report.md"
        report.write_text(report.read_text().replace(
            "| transfer | none | Source-defined caller conditions | `function transfer() public` |",
            "| transferOwnership | onlyOwner | Only the owner | `function transferOwnership() public onlyOwner` |",
            1,
        ))
        with self.assertRaisesRegex(ValueError, "transferOwnership.*does not exist"):
            verify(self.root)

    def test_burn_cannot_claim_auth_when_its_real_declaration_has_no_modifier(self):
        self.source = (
            "// SPDX-License-Identifier: MIT\ncontract Example { "
            "function transfer() public {} function burn(address usr, uint wad) external {} }\n"
        )
        self.original = self.source
        self.functions.append(("burn", "none", "function burn(address usr, uint wad) external"))
        self.seed()
        self.assertEqual(verify(self.root)["status"], "verified")
        report = self.root / "report.md"
        report.write_text(report.read_text().replace("| burn | none |", "| burn | auth |", 1))
        with self.assertRaisesRegex(ValueError, "burn: named modifiers must be none"):
            verify(self.root)

    def test_quote_must_match_the_real_declaration_not_a_plausible_signature(self):
        report = self.root / "report.md"
        report.write_text(report.read_text().replace(
            "`function transfer() public`", "`function transfer(address to, uint value) public`", 1,
        ))
        with self.assertRaisesRegex(ValueError, "declaration differs from verified source"):
            verify(self.root)

    def test_omitting_a_public_function_is_not_complete_source_evidence(self):
        self.source = (
            "// SPDX-License-Identifier: MIT\ncontract Example { "
            "function transfer() public {} function approve(address guy, uint wad) external {} }\n"
        )
        self.original = self.source
        self.seed()
        with self.assertRaisesRegex(ValueError, "missing approve"):
            verify(self.root)

    def test_appended_unverified_declaration_cannot_change_reported_modifiers(self):
        self.source += "contract Fake { function transfer() public auth {} }\n"
        self.functions = [("transfer", "auth", "function transfer() public auth")]
        self.seed()
        with self.assertRaisesRegex(ValueError, "declaration differs from verified source"):
            verify(self.root)

    def test_flattened_override_uses_the_derived_declaration_and_named_modifier(self):
        self.source = (
            "// SPDX-License-Identifier: MIT\n"
            "contract Base { function transfer() public {} }\n"
            "contract Example is Base { modifier validRecipient { _; } "
            "function transfer() public validRecipient {} }\n"
        )
        self.original = self.source
        self.functions = [("transfer", "validRecipient", "function transfer() public validRecipient")]
        self.seed()
        self.assertEqual(verify(self.root)["status"], "verified")
        report = self.root / "report.md"
        report.write_text(report.read_text().replace(
            "| transfer | validRecipient |", "| transfer | none |", 1,
        ))
        with self.assertRaisesRegex(ValueError, "named modifiers must be validRecipient"):
            verify(self.root)

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
