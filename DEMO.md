# 60-second demo

```bash
./scripts/run_demo.sh
```

## What you should see

1. **Alice (HR, confidential)** asks about the remote-work stipend → grounded answer citing `POL-HR-001` with `$75`.
2. **Bob (Engineering, internal)** asks the same question → **refusal** (HR confidential doc filtered by ACL).
3. **Eve (Contractor, public)** asks how to report ethics concerns → answer citing `POL-PUB-003`.
4. **Alice** asks about an aquarium cleaning schedule → **refusal** (no corpus evidence; will not fabricate).

Then offline eval prints `score=100%` and pytest goes green.

## Manual commands

```bash
agentthree-rag principals
agentthree-rag ask -p alice_hr "What is the monthly home-office stipend for fully remote employees?"
agentthree-rag ask -p bob_eng "What is the monthly home-office stipend for fully remote employees?"
agentthree-rag eval
```
