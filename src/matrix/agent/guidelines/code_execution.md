## Code Execution Guidelines
When calling `code.run_python`, follow these rules:

**When to use code execution:**
- Data processing: calculate returns, aggregate holdings, compute ratios
- Numerical analysis: statistical calculations, percentage changes, comparisons
- Format conversion: transform tool results into tables or structured formats
- Multi-step calculations that are error-prone to do mentally

**When NOT to use code execution:**
- Simple arithmetic (e.g., "35000 / 100 = 350") — do it inline
- When a direct tool call already provides the answer
- For web searches or data retrieval — use the appropriate tools instead

**Code writing rules:**
- Use `print()` to output results — the sandbox captures stdout
- Available libraries: Python standard library only (json, csv, math, statistics, datetime, etc.)
- Do NOT use `os.system`, `subprocess`, `shutil.rmtree`, or other system calls
- Do NOT attempt file operations outside the sandbox
- Keep code concise and focused on the specific calculation
- If you need data from a previous tool result, embed it directly in the code as a variable

**Example:**
```python
# Calculate portfolio return
holdings = {"股票": 150000, "债券": 80000, "现金": 20000}
total = sum(holdings.values())
for asset, value in holdings.items():
    pct = value / total * 100
    print(f"{asset}: {value}元 ({pct:.1f}%)")
print(f"总计: {total}元")
```

If execution fails (exit_code != 0), read the stderr, fix the code, and retry. Common issues: syntax errors, missing imports, typos.