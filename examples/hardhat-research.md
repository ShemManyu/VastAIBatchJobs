# Execute a Hardhat contract research task

Perform the actions in the remote workspace, do not just describe scripts.
Use Ethereum mainnet as the read-only source and an in-process Hardhat fork as
the execution sandbox. Never send transactions, deploy contracts, mine blocks,
request signing keys, or use a live network for transaction execution.

## Available credentials and tools

- `ALCHEMY_API_KEY` and `ETHERSCAN_API_KEY` are available to command processes.
  Read them from environment variables. Never print or save their values.
- The Alchemy Ethereum mainnet JSON-RPC URL is
  `https://eth-mainnet.g.alchemy.com/v2/` followed by the environment key.
- Etherscan verified-source requests use `https://api.etherscan.io/v2/api`
  with `chainid=1`, `module=contract`, `action=getsourcecode`, `address`,
  and the key from the environment as the `apikey` parameter.
- Alchemy `eth_getCode` returns deployed bytecode. Obtain verified Solidity
  source from Etherscan, not by pretending that bytecode is Solidity.
- Node 22, npm, and Python are installed. Use compact CommonJS scripts.
- Install these compatible pins locally: `hardhat@2.29.1`,
  `@nomicfoundation/hardhat-ethers@3.1.3`, and `ethers@6.17.0`.
  Use npm's `--no-audit --no-fund` options to keep output small.
- Do not install an upgrades plugin, toolbox, or OpenZeppelin package. This
  read-only task needs only the three specified packages. If an import fails,
  inspect the config/script and remove an unnecessary import instead of
  guessing other package names or versions.

## Required actions

1. Create `package.json` and install the pinned dependencies. Create
   `hardhat.config.cjs`, load `@nomicfoundation/hardhat-ethers`, set
   `defaultNetwork` to `hardhat`, and configure `networks.hardhat.forking.url`
   from the Alchemy environment key. Set `networks.hardhat.accounts` to `[]`.
   No Solidity compilation is necessary.
   The config must use `require("@nomicfoundation/hardhat-ethers")`; do not
   substitute a bare `require("ethers")` for loading the Hardhat plugin.
2. Create `scripts/read-balances.cjs`. Randomly sample two different addresses
   from this pool of known non-proxy, verified mainnet contracts:
   - WETH9: `0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2`
   - Dai: `0x6B175474E89094C44Da98b954EedeAC495271d0F`
   - Chainlink token: `0x514910771AF9Ca656af840dff83E8264EcF986CA`
   This pool avoids confusing random EOAs with smart contracts. Verify nonempty
   deployed code for both selected addresses. Use Hardhat's ethers provider to
   read their native ETH balances on the fork and its block number. Record
   balances as decimal wei strings and formatted ETH strings.
3. Execute the script with
   `npx hardhat run scripts/read-balances.cjs --no-compile`.
   Check its actual exit code and output. The script must write
   `outputs/balances.json` and print its results. Use this structure:
   ```json
   {
     "source_chain_id": 1,
     "fork_chain_id": 31337,
     "block_number": 0,
     "contracts": [
       {"address": "selected address", "balance_wei": "decimal integer",
        "balance_eth": "formatted ETH", "fork_code_bytes": 1}
     ]
   }
   ```
   Replace all placeholders with observed values; there must be two contracts.
   A zero ETH balance is valid, but empty contract code is not.
4. Create and execute `scripts/fetch-sources.cjs`. Read the selected addresses
   and fork block from `outputs/balances.json`. For each address:
   - Call Alchemy `eth_getCode` with the same block encoded as a hex quantity.
     Save the full bytecode and SHA-256 of its decoded bytes.
   - Call Etherscan's verified-source endpoint. Check HTTP success, API status
     `1`, and nonempty `SourceCode`. Do not treat an error or missing source as
     success. Pause between requests if needed for the free API rate limit.
   - Save the original response to `sources/<address>/etherscan.json`.
     Save the verified source as `sources/<address>/contract.sol`, preserving
     its license notices. If the response is wrapped standard-input JSON,
     unwrap it and combine its source contents with original filename headers;
     do not use unvalidated API filenames as filesystem paths.
   - Write `outputs/contracts.json` with:
     ```json
     {
       "chain_id": 1,
       "block_number": 0,
       "contracts": [
         {"address": "selected address", "runtime_bytecode": "0x...",
          "runtime_bytecode_sha256": "sha256",
          "contract_name": "verified name",
          "source_path": "sources/address/contract.sol",
          "source_sha256": "sha256 of saved UTF-8 source",
          "etherscan_response_path": "sources/address/etherscan.json"}
       ]
     }
     ```
     All values must come from the actual results, not invented examples.
   Print only concise metadata: address, contract name, source path, and hashes.
   Do not dump full bytecode, ABI, or the raw Etherscan response into tool output.
5. Read the saved Solidity source and balance output. Inspect the contract
   output with a short script that omits `runtime_bytecode`, rather than loading
   large hex strings into the conversation. Write `report.md`
   summarizing each contract's purpose, key functions, state transitions,
   balance/allowance handling, and administrative/access-control behavior.
   Cite the saved source file and actual function names for each explanation.
   Distinguish observed source behavior from assumptions. Include the sampled
   addresses, block, native ETH balances, and the commands that ran successfully.
6. Before finishing, verify both scripts ran successfully, both contracts have
   Alchemy bytecode and verified Solidity files, and the report exists.
   Return a concise final summary pointing to the scripts, JSON outputs,
   downloaded source evidence, and `report.md`.

Keep scripts compact enough to write in individual actions. Correct observed
errors instead of claiming success. Do not copy `node_modules` or caches into
the report; the workflow returns useful files and excludes dependencies.
