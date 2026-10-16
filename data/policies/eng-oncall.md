---
doc_id: POL-ENG-002
title: Engineering On-Call Handbook
classification: internal
allowed_departments: [engineering, sre]
allowed_roles: [employee]
source: Engineering Wiki
effective_date: 2025-03-01
---

# Engineering On-Call Handbook

## Rotation
Primary on-call rotates weekly among senior engineers. Secondary on-call provides escalation coverage within 15 minutes.

## Severity
- Sev-1: customer-facing outage — page immediately, bridge within 5 minutes.
- Sev-2: degraded service — acknowledge within 15 minutes.
- Sev-3: non-urgent defect — ticket during business hours.

## Runbooks
Always follow the linked runbook before applying hotfixes. Document every Sev-1 in the postmortem template within 48 hours.
