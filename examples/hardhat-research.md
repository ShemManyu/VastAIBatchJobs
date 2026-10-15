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
  Declare exact versions in `package.json`, not caret/tilde ranges; use
  `--save-exact` if npm creates dependency declarations.
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
   Begin the config with this exact plugin-loading statement:
   ```js
   require("@nomicfoundation/hardhat-ethers");
   ```
   If `ethers` or `ethers.provider` is undefined, inspect the config and add the
   missing plugin load. Rewriting the balance script without fixing the config
   does not load the plugin. Never replace the required Hardhat fork with direct
   Alchemy balance calls or merely label mainnet RPC results as chain 31337.
   Set `paths.cache` to `.cache/hardhat` so fork caches are excluded from
   returned artifacts. Create output directories before writing results.
2. Create `scripts/read-balances.cjs`. Randomly sample two different addresses
   from this pool of known non-proxy, verified mainnet contracts:
   - WETH9: `0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2`
   - Dai: `0x6B175474E89094C44Da98b954EedeAC495271d0F`
   - Chainlink token: `0x514910771AF9Ca656af840dff83E8264EcF986CA`
   This pool avoids confusing random EOAs with smart contracts. Verify nonempty
   deployed code for both selected addresses. Use Hardhat's ethers provider to
   read their native ETH balances on the fork and its block number. Record
   balances as decimal wei strings and formatted ETH strings.
   Import `const { ethers } = require("hardhat")` in this script. Use
   `balance.toString()` for wei and `ethers.formatEther(balance)` for ETH;
   never use `Number`, `parseInt`, or floating-point division for ETH amounts.
   Record deployed code length in **bytes**, using `(code.length - 2) / 2`
   or `Buffer.from(code.slice(2), "hex").length`, not `code.length`.
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
   Keep the same sampled addresses while correcting an error. Do not resample
   as a response to a format/API/report failure.
4. Create and execute `scripts/fetch-sources.cjs` with
   `node scripts/fetch-sources.cjs`. Read the selected addresses
   and fork block from `outputs/balances.json`. For each address:
   - Call Alchemy `eth_getCode` with the same block encoded as a hex quantity.
     Save the full bytecode and SHA-256 of its decoded bytes.
     Use Node's built-in `fetch` to POST JSON-RPC to the Alchemy URL above;
     check HTTP success and reject a JSON-RPC `error`.
   - Call Etherscan's verified-source endpoint. Check HTTP success, API status
     `1`, and nonempty `SourceCode`. Do not treat an error or missing source as
     success. Pause between requests if needed for the free API rate limit.
     Construct the URL like this, including **both `/v2/api` and `chainid=1`**:
     ```js
     const url = new URL("https://api.etherscan.io/v2/api");
     url.search = new URLSearchParams({
       chainid: "1", module: "contract", action: "getsourcecode",
       address, apikey: process.env.ETHERSCAN_API_KEY
     }).toString();
     ```
     If the API says "Missing chainid", add `chainid=1` without changing
     `/v2/api`. Never switch to the deprecated `/api` endpoint.
     Do not skip a failed contract, create placeholder Solidity, or substitute
     remembered contract behavior for downloaded source.
   - Save the **complete original API response** to
     `sources/<address>/etherscan.json`, including top-level `status`, `message`
     and the `result` array. Use `JSON.stringify(etherscanJson, null, 2)`,
     not `JSON.stringify(etherscanJson.result[0])`. Extract source fields from
     `etherscanJson.result[0]`, but preserve the complete envelope as evidence.
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
   Use Node's global `fetch`; do not import or install `node-fetch`.
5. Read the saved Solidity source and balance output. Inspect the contract
   output with a short script that omits `runtime_bytecode`. Never call
   `read_file` on `outputs/contracts.json`, which contains large hex strings.
   Write `report.md`
   summarizing each contract's purpose, key functions, state transitions,
   balance/allowance handling, and administrative/access-control behavior.
   Cite the saved source file and actual function names for each explanation.
   Distinguish observed source behavior from assumptions. Include the sampled
   addresses, block, native ETH balances, and the commands that ran successfully.
   Create a small extraction/rendering helper to derive function names,
   declarations and named modifiers from the actual saved source. Keep your
   source-grounded caller-condition notes in structured data and render the
   Markdown tables from the extracted values, rather than retyping signatures
   or guessing layout. Inspect bodies separately; a declaration alone does
   not establish caller conditions or state transitions.
   Derive allowed named modifiers from actual `modifier NAME` declarations,
   including inherited ones. Match only those names in function headers and
   render `none` for an empty set; tokens after the parameter list can also
   contain visibility and `returns`, not just named modifiers.
   Include a short access-control table listing
   sensitive functions, their actual modifiers, and their caller conditions.
   Inspect each individual declaration/body, not ERC-20 stereotypes:
   Include a short exact declaration or condition from the saved source beside
   each access-control claim. Keep each contract's functions in its own section:
   Dai's `wards`, `rely`, `deny`, and `auth` do not exist in LINK or WETH9.
   Start each contract section with `## CONTRACT_NAME (ADDRESS)`, using its
   actual verified name and selected address. Include exactly one table with
   this header:
   `| Function | Modifiers | Caller conditions | Source declaration |`
   Cover every named public/external function defined in the source, including
   inherited functions. Use the final derived override where one exists.
   Function cells contain plain function names. Modifiers cells contain only
   the actual named Solidity modifiers, comma-separated, or `none`; visibility,
   `view`, `constant`, and `returns` are not named modifiers.
   Quote each exact declaration before its opening `{` in backticks in the last
   column, collapsing source whitespace onto one line. Explain actual caller
   conditions in the third column. Escape literal Markdown table pipes as `\|`.
   For Dai `transfer`/`push` and WETH9 `transfer`, use the canonical caller
   condition `balanceOf[msg.sender] >= wad`. These pass `msg.sender` as source
   to `transferFrom`, so recipient allowance is never a condition. The verifier
   checks that source call before enforcing this formula for the known pool.
   Do not invent functions or fill the table from ERC-20/Ownable stereotypes.
   Read the downloaded source first; the finish verifier compares the table
   directly with freshly fetched verified declarations and modifiers.
   - For Dai, compare `mint` and `burn` authorization separately, inspect the
     holder/allowance branches, and describe `permit`'s actual approval values.
   - For LINK, inspect the exact `validRecipient` expression and
     `transferAndCall`/`isContract` behavior. Do not confuse `address(this)`
     with all contract addresses. Distinguish raw supply units from token
     quantities using `decimals`.
   - For WETH9, inspect how `totalSupply`, deposits/withdrawals, and delegated
     transfers derive balances rather than assuming a stored supply counter.
6. Before finishing, verify both scripts ran successfully, both contracts have
   Alchemy bytecode and verified Solidity files, and the report exists.
   Return a concise final summary pointing to the scripts, JSON outputs,
   downloaded source evidence, and `report.md`.

Keep scripts compact enough to write in individual actions. Correct observed
errors instead of claiming success. Do not copy `node_modules` or caches into
the report; the workflow returns useful files and excludes dependencies.

Both scripts must signal failure with a nonzero process exit code. Use
`main().catch(error => { console.error(error.message); process.exitCode = 1; });`,
not `main().catch(console.error)`, which prints an error while exiting zero.
Inspect stderr as well as the exit code; an error message is not success.

The example command supplies an independent finish verifier. It rechecks the
recorded balances/bytecode against Alchemy and verified Solidity against
Etherscan. A failed finish returns the actual check error so you can correct
your scripts and rerun them. The verifier cannot be replaced from the workspace.
Read the actual saved source before writing the report; use `next_offset` to
read remaining pages when a source file is longer than one tool response.
