---
doc_id: MB-POL-007
title: Customer Data Privacy and Responsible Data Use Policy
owner: Data Protection Office
version: 3.0
effective_date: 2026-01-15
classification: Internal
synthetic: true
---

> SYNTHETIC DOCUMENT. Meridian Bank is a fictional institution. This policy was written for a software portfolio demonstration and does not describe any real bank's practices. Do not use it for real compliance decisions.

# Customer Data Privacy and Responsible Data Use Policy

## 1. Purpose and Scope

This policy governs how Meridian Bank collects, classifies, uses, shares, retains and deletes customer personal data, including its use in credit models and AI tools. It applies to all employees, contractors and vendors who handle customer data.

## 2. Data Classification

| Class | Examples | Handling |
|---|---|---|
| Public | Published rates, product brochures | No restrictions |
| Internal | Credit policies, procedures | Employees only |
| Confidential | Income documents, bureau reports, account balances | Need-to-know access, encrypted at rest |
| Restricted | Government ID numbers, full card numbers (PAN), biometric data, health information | Need-to-know, encrypted, masked in all non-production environments |

## 3. Lawful Use and Minimization

Customer data may only be used for the purposes disclosed in the privacy notice. Credit analysts may access bureau reports only for applications or accounts they are assigned to. Downloading customer data to personal devices is prohibited.

## 4. Use of Data in Credit Decisions and Models

### 4.1 Prohibited Characteristics

Credit decisions, scorecards and pricing must never use race, color, religion, national origin, sex, gender identity, sexual orientation, marital status, disability or health information, whether directly or through a proxy variable. Age may only be used to confirm legal capacity and where explicitly permitted by law.

### 4.2 Model Fairness Review

Every new or materially changed credit model must pass a fairness review by Model Risk Management before deployment, including disparate impact testing, and must be re-tested annually.

## 5. Generative AI and External Tools

- Restricted or Confidential data must never be entered into external AI tools or public chatbots.
- Internal AI assistants approved by the Data Protection Office may process Internal data. They may process Confidential data only when personal identifiers are masked before the data reaches the model.
- Outputs of AI assistants used in credit decisions must cite the source document and be reviewed by a human analyst. AI output alone must never be the basis of a decline.

## 6. Masking Standards

In logs, analytics and non-production systems, government ID numbers must be masked to the last 4 digits, card numbers must show at most the first 6 and last 4 digits, and email addresses and phone numbers must be pseudonymized.

## 7. Retention

| Record | Retention period |
|---|---|
| Approved credit application and account records | 7 years after account closure |
| Declined applications | 25 months after the decision |
| Call recordings in collections | 2 years |
| KYC records | As defined in MB-POL-006 (at least 5 years after relationship end) |

After the retention period, records are deleted or irreversibly anonymized within 90 days.

## 8. Data Subject Requests

Requests from customers to access, correct or delete their data must be answered within 30 calendar days. Deletion requests cannot override legal retention obligations.

## 9. Incidents

Any suspected personal data breach must be reported to the Data Protection Office within 24 hours of discovery. The Data Protection Office decides on regulator and customer notification.
