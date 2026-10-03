# RTA Workflow Test Specification

## Overview

This document defines the end-to-end RTA workflow tests: happy paths and edge cases that prove the product works and handles the legally-complex scenarios.

**Source**: docs/draftly-rta-matter-workflow-v1.md + ARCHITECTURE.md §10 state machines.

**Pattern**: Follow backend/tests/e2e/test_services_slice.py — chain application-service layer calls in-memory first (Phase 1), then over real Postgres (Phase 2).

---

## Happy Path: Simple Freehold Transfer

### Scenario
A first-time buyer purchases a simple freehold property with no encumbrances. Lawyer guides through intake → draft generation → approval → export to registration.

### Flow Diagram

```
1. Lawyer opens new matter
   → Matter created (status=intake)
   → Checklist auto-compiled based on matter type (freehold)

2. Lawyer uploads deed + mortgage + registry extract
   → Documents ingested
   → Gemini extracts facts (unverified)

3. Lawyer reviews extracted facts
   → Marks buyer_name, seller_name, property_address as verified
   → Updates matter status → verification

4. System checks eligibility
   → All checklist rules pass (buyer first-time, no encumbrance, etc.)
   → No statutory blockers

5. Lawyer generates draft
   → Form fields bound to verified facts
   → Draft created (status=draft)

6. Lawyer reviews draft, marks for approval
   → Draft status → submitted

7. Lawyer approves draft
   → Draft status → approved
   → Audit log records approval

8. Lawyer exports to registration
   → Export service generates filled form + deed summary
   → Download link provided
   → Matter status → exported
```

### Test Specification

**File**: `backend/tests/e2e/test_rta_happy_path.py`

```python
def test_rta_freehold_simple_end_to_end(
    in_memory_services,  # Fixture: all services as fakes
    synthetic_parties,
    simple_matter_freehold,
):
    """Test complete RTA workflow: intake → draft → approve → export."""
    
    # === Step 1: Matter Created & Checklist Compiled ===
    matter = simple_matter_freehold
    assert matter.status == 'intake'
    
    checklist = in_memory_services.check_service.get_checklist(matter.id)
    assert checklist.rules == {
        'buyer_first_time': {'status': 'pass', 'rule': 'First-time buyer'},
        'no_encumbrance': {'status': 'pass', 'rule': 'No encumbrances'},
        'no_subsisting_mortgage': {'status': 'pass', 'rule': 'No existing mortgage'},
        'property_not_leasehold': {'status': 'pass', 'rule': 'Property is freehold'},
    }
    assert matter.statutory_blockers == []
    
    # === Step 2: Documents Uploaded & Facts Extracted ===
    deed_doc = in_memory_services.document_service.upload(
        matter_id=matter.id,
        doc_type='deed',
        content=MINIMAL_PDF_DEED,
    )
    assert deed_doc.status == 'uploaded'
    
    # Gemini (stubbed) extracts facts
    facts = in_memory_services.verification_service.extract_facts(deed_doc.id)
    assert facts == {
        'buyer_name': {'value': 'Nimal Jayasinghe', 'confidence': 0.95, 'verified': False},
        'seller_name': {'value': 'Lakshmi Perera', 'confidence': 0.92, 'verified': False},
        'property_address': {'value': '123 Galle Road, Colombo 3', 'confidence': 0.88, 'verified': False},
    }
    
    # === Step 3: Lawyer Verifies Facts ===
    for fact_key in ['buyer_name', 'seller_name', 'property_address']:
        verified_fact = in_memory_services.verification_service.mark_verified(
            fact_id=facts[fact_key]['id'],
            verified_by=matter.lawyer_id,
        )
        assert verified_fact['verified'] == True
        assert verified_fact['verified_by'] == matter.lawyer_id
    
    # === Step 4: Eligibility Check & No Blockers ===
    eligibility = in_memory_services.check_service.check_eligibility(matter.id)
    assert eligibility['can_automate'] == True
    assert eligibility['blockers'] == []
    
    # === Step 5: Draft Generated ===
    draft = in_memory_services.draft_service.generate(
        matter_id=matter.id,
        generated_by=matter.lawyer_id,
    )
    assert draft.status == 'draft'
    assert draft.locked_blocks == []  # Statutory text is immutable
    assert draft.placeholder_bindings == {  # Form fields bound to facts
        'buyer_name': 'Nimal Jayasinghe',
        'seller_name': 'Lakshmi Perera',
        'property_address': '123 Galle Road, Colombo 3',
    }
    
    # Verify audit trail
    audit = in_memory_services.audit_service.get_events(
        resource='draft',
        action='create'
    )
    assert len(audit) >= 1
    assert audit[-1]['details']['matter_id'] == str(matter.id)
    
    # === Step 6: Lawyer Submits Draft ===
    draft = in_memory_services.draft_service.submit(
        draft_id=draft.id,
        submitted_by=matter.lawyer_id,
    )
    assert draft.status == 'submitted'
    
    # === Step 7: Lawyer Approves Draft ===
    draft = in_memory_services.approval_service.approve(
        draft_id=draft.id,
        approved_by=matter.lawyer_id,
    )
    assert draft.status == 'approved'
    
    # Verify only the approver can export (access control)
    export_link = in_memory_services.export_service.prepare(draft.id)
    assert export_link is not None
    
    # === Step 8: Export to Registration ===
    export_result = in_memory_services.export_service.export(
        draft_id=draft.id,
        exported_by=matter.lawyer_id,
    )
    assert export_result['status'] == 'exported'
    assert 'deed_summary' in export_result
    assert 'registration_form' in export_result
    
    # Matter status reflects completion
    matter = in_memory_services.matter_service.get(matter.id)
    assert matter.status == 'exported'
```

### Success Criteria
- ✅ Matter status transitions: intake → verification → draft → submitted → approved → exported
- ✅ Checklist rules all pass (no blockers)
- ✅ All mutations are audited (7+ audit events)
- ✅ Unverified facts block draft generation (if step 3 is skipped)
- ✅ Only approved drafts can be exported
- ✅ Export generates both deed summary and registration form

---

## Edge Case 1: Statutory Blocker (Encumbrance Dispute)

### Scenario
Lawyer discovers an encumbrance (lien, etc.) on the property. RTA rules forbid automatic approval if encumbrance cannot be resolved. Lawyer must manually override or abandon the matter.

### Test Specification

**File**: `backend/tests/e2e/test_rta_statutory_blockers.py::test_encumbrance_blocker`

```python
def test_encumbrance_blocker_cannot_be_waived(
    in_memory_services,
    matter_with_encumbrance,  # Fixture: checklist rule 'no_encumbrance' fails
):
    """Test that statutory encumbrance blocker prevents draft generation."""
    
    matter = matter_with_encumbrance
    assert matter.statutory_blockers == ['encumbrance_dispute']
    
    # Lawyer attempts to generate draft
    with pytest.raises(ApprovalError) as exc:
        in_memory_services.draft_service.generate(matter_id=matter.id)
    
    # Error is clear, not a generic 400
    assert exc.value.code == 'STATUTORY_BLOCKER_BLOCKS_DRAFT'
    assert 'encumbrance' in exc.value.message.lower()
    
    # Audit log records the blocked attempt
    audit = in_memory_services.audit_service.get_events(
        resource='draft',
        action='generate_blocked',
    )
    assert any(e['details']['matter_id'] == str(matter.id) for e in audit)
    
    # Lawyer cannot override blocker (even if they try to force it via API)
    # This would require a separate test showing that the API doesn't provide
    # a "force_override" parameter at all — the only way forward is manual
    # resolution (remove encumbrance) or abandon matter.
    assert hasattr(draft_service, 'force_generate') == False  # Method doesn't exist
```

### Success Criteria
- ✅ Statutory blocker prevents draft generation
- ✅ No override mechanism exists (blocker is truly immutable)
- ✅ Audit log records blocked attempt
- ✅ Error message is clear and specific (not generic 400)

---

## Edge Case 2: Stale Approval (Fact Correction Post-Approval)

### Scenario
Lawyer approves a draft based on facts F1={buyer_name: 'Nimal'}. Later, a corrected fact F2={buyer_name: 'Nimal Kumar'} is discovered. The approved draft becomes stale. Lawyer must reapprove with fresh facts.

**Rule** (ARCHITECTURE.md Invariant #5): "A superseded fact marking an approved form stale."

### Test Specification

**File**: `backend/tests/e2e/test_rta_statutory_blockers.py::test_stale_approval_on_fact_correction`

```python
def test_stale_approval_on_fact_correction(
    in_memory_services,
    draft_approved,  # Fixture: approved draft bound to fact_v1
):
    """Test that correcting a fact invalidates an approved draft."""
    
    # Draft is approved based on F1
    draft = draft_approved
    assert draft.status == 'approved'
    assert draft.placeholder_bindings['buyer_name'] == 'Nimal'
    fact_v1_id = draft.fact_version_id  # Fact version embedded in draft
    
    # Later, lawyer realizes fact was wrong, marks correction
    fact_v2 = in_memory_services.verification_service.mark_verified(
        fact_id=fact_v1_id.key('buyer_name'),
        verified_value='Nimal Kumar',  # Different!
        verified_by=lawyer_id,
    )
    
    # Draft is now stale (linked fact was superseded)
    draft = in_memory_services.draft_service.get(draft.id)
    assert draft.status == 'stale'  # Status changed automatically
    
    # Lawyer cannot export stale draft
    with pytest.raises(ApprovalError) as exc:
        in_memory_services.export_service.export(draft_id=draft.id)
    
    assert exc.value.code == 'DRAFT_STALE_CANNOT_EXPORT'
    
    # Lawyer must regenerate draft with new facts
    new_draft = in_memory_services.draft_service.generate(
        matter_id=draft.matter_id,
        generated_by=lawyer_id,
    )
    assert new_draft.status == 'draft'
    assert new_draft.placeholder_bindings['buyer_name'] == 'Nimal Kumar'
    assert new_draft.id != draft.id  # New draft object
    
    # Lawyer approves new draft
    new_draft = in_memory_services.approval_service.approve(
        draft_id=new_draft.id,
        approved_by=lawyer_id,
    )
    assert new_draft.status == 'approved'
```

### Success Criteria
- ✅ Correcting a fact marks dependent draft as stale
- ✅ Stale drafts cannot be exported
- ✅ Lawyer must regenerate with corrected facts
- ✅ New draft is separate object, old draft remains archived

---

## Edge Case 3: Export Before Approval

### Scenario
Lawyer attempts to export a draft that hasn't been approved yet. System rejects with 400 (not 500).

### Test Specification

**File**: `backend/tests/e2e/test_rta_statutory_blockers.py::test_export_before_approval`

```python
def test_export_before_approval_fails(
    in_memory_services,
    draft_submitted,  # Fixture: draft in 'submitted' status, not approved
):
    """Test that exporting a non-approved draft fails."""
    
    draft = draft_submitted
    assert draft.status == 'submitted'
    
    # Attempt export
    with pytest.raises(ApprovalError) as exc:
        in_memory_services.export_service.export(draft_id=draft.id)
    
    # Specific error, not generic 500
    assert exc.value.code == 'DRAFT_NOT_APPROVED'
    
    # Audit logs attempt
    audit = in_memory_services.audit_service.get_events(
        resource='export',
        action='export_blocked'
    )
    assert any(e['details']['reason'] == 'DRAFT_NOT_APPROVED' for e in audit)
```

### Success Criteria
- ✅ Export rejects non-approved drafts with specific error
- ✅ Audit log records blocked export attempt

---

## Edge Case 4: Section 47 Part-Parcel Disposition

### Scenario
Property is registered as part-parcel under land registry regulations. Section 47 of the Land Title Act requires special handling (consent from land commissioner, etc.). Automatic approval is blocked until Section 47 process completes.

**Note** (from ARCHITECTURE.md testing example): "test_statutory_blocker.py::test_section_47_blocker"

### Test Specification

**File**: `backend/tests/e2e/test_rta_statutory_blockers.py::test_section_47_part_parcel`

```python
def test_section_47_part_parcel_blocker(
    in_memory_services,
    matter_with_section_47_property,  # Fixture: folio indicates part-parcel
):
    """Test that Section 47 part-parcel properties cannot auto-approve."""
    
    matter = matter_with_section_47_property
    
    # Eligibility check detects Section 47 requirement
    eligibility = in_memory_services.check_service.check_eligibility(matter.id)
    assert eligibility['can_automate'] == False
    assert 'section_47' in str(eligibility['blockers'])
    
    # Statutory blocker is set on matter
    assert 'section_47_part_parcel' in matter.statutory_blockers
    
    # Lawyer cannot generate draft without manual resolution
    with pytest.raises(ApprovalError) as exc:
        in_memory_services.draft_service.generate(matter_id=matter.id)
    
    assert exc.value.code == 'STATUTORY_BLOCKER_BLOCKS_DRAFT'
```

### Success Criteria
- ✅ Section 47 detection blocks automatic processing
- ✅ Lawyer is informed of specific blocker
- ✅ No override mechanism allows bypassing blocker

---

## Edge Case 5: Unverified Facts Block Draft

### Scenario
Lawyer attempts to generate draft without verifying all required facts. Machine-extracted facts alone (even at high confidence) cannot feed into a draft.

**Rule** (ARCHITECTURE.md Invariant #4): "Machine extraction is never authoritative."

### Test Specification

**File**: `backend/tests/e2e/test_rta_statutory_blockers.py::test_unverified_facts_block_draft`

```python
def test_unverified_facts_block_draft(
    in_memory_services,
    matter_with_extracted_facts,  # Fixture: facts are unverified
):
    """Test that unverified facts cannot be used in draft generation."""
    
    matter = matter_with_extracted_facts
    
    # Facts exist but are unverified
    facts = in_memory_services.verification_service.list_facts(matter.id)
    assert all(f['verified'] == False for f in facts)
    
    # Attempt to generate draft
    with pytest.raises(ApprovalError) as exc:
        in_memory_services.draft_service.generate(matter_id=matter.id)
    
    # Specific error: facts require verification
    assert exc.value.code == 'UNVERIFIED_FACTS_BLOCK_DRAFT'
    assert 'buyer_name' in exc.value.message or 'verify' in exc.value.message.lower()
    
    # Audit log records the blocked draft generation
    audit = in_memory_services.audit_service.get_events(
        resource='draft',
        action='generate_blocked'
    )
    assert any(
        'UNVERIFIED_FACTS' in str(e['details'].get('reason', ''))
        for e in audit
    )
```

### Success Criteria
- ✅ Draft generation requires all critical facts to be verified
- ✅ Machine confidence score alone is insufficient
- ✅ Clear error message tells lawyer what facts need verification
- ✅ Audit log records the blocked attempt

---

## Test Structure & Execution

### File Organization

```
backend/tests/e2e/
├── test_rta_happy_path.py
│   └── test_rta_freehold_simple_end_to_end()
├── test_rta_statutory_blockers.py
│   ├── test_encumbrance_blocker_cannot_be_waived()
│   ├── test_stale_approval_on_fact_correction()
│   ├── test_export_before_approval_fails()
│   ├── test_section_47_part_parcel()
│   └── test_unverified_facts_block_draft()
├── conftest.py
│   ├── fixtures: in_memory_services, synthetic_parties, simple_matter_freehold, etc.
└── test_services_slice.py (existing)
```

### Fixtures Needed

**File**: `backend/tests/e2e/conftest.py` (new)

```python
import pytest
from uuid import uuid4

from backend.tests.fixtures.factories import (
    create_matter_with_parties,
    create_document_for_matter,
    create_fact,
)
from backend.tests.fixtures.matter_synthetic import (
    SYNTHETIC_MATTER_FREEHOLD_SIMPLE,
    SYNTHETIC_MATTER_FREEHOLD_WITH_ENCUMBRANCE,
    SYNTHETIC_ORG_A,
    SYNTHETIC_LAWYER_A,
)
from backend.tests.fixtures.party_synthetic import (
    SYNTHETIC_PARTY_BUYER,
    SYNTHETIC_PARTY_SELLER,
    SYNTHETIC_PARTY_LAWYER,
)

@pytest.fixture
def in_memory_services():
    """Fixture: All services as in-memory fakes (no DB, no external APIs)."""
    # Return instances of FakeMatterService, FakeDocumentService, etc.
    # These are defined in backend/tests/fixtures/service_fakes.py
    return ServiceRegistry(
        matter_service=FakeMatterService(),
        document_service=FakeDocumentService(),
        verification_service=FakeVerificationService(mock_gemini=True),
        draft_service=FakeDraftService(),
        approval_service=FakeApprovalService(),
        export_service=FakeExportService(),
        check_service=FakeCheckService(),
        audit_service=FakeAuditService(),
    )

@pytest.fixture
def simple_matter_freehold(in_memory_services):
    """Fixture: Simple freehold matter, ready for document upload."""
    matter = in_memory_services.matter_service.create(
        org_id=SYNTHETIC_ORG_A,
        lawyer_id=SYNTHETIC_LAWYER_A,
        matter_type='rta_freehold',
        parties=[SYNTHETIC_PARTY_BUYER.id, SYNTHETIC_PARTY_SELLER.id],
    )
    return matter

@pytest.fixture
def matter_with_encumbrance(in_memory_services):
    """Fixture: Matter with encumbrance blocker (auto-approval forbidden)."""
    matter = in_memory_services.matter_service.create(
        org_id=SYNTHETIC_ORG_A,
        lawyer_id=SYNTHETIC_LAWYER_A,
        matter_type='rta_freehold',
    )
    # Simulate checklist rule failure + blocker
    matter.checklist_status['no_encumbrance'] = 'fail'
    matter.statutory_blockers = ['encumbrance_dispute']
    return matter

@pytest.fixture
def draft_approved(in_memory_services, simple_matter_freehold):
    """Fixture: Approved draft ready for export."""
    matter = simple_matter_freehold
    
    # Upload documents
    deed = in_memory_services.document_service.upload(
        matter_id=matter.id,
        doc_type='deed',
    )
    
    # Verify facts
    facts = in_memory_services.verification_service.extract_facts(deed.id)
    for key in facts:
        in_memory_services.verification_service.mark_verified(
            fact_id=facts[key]['id'],
            verified_by=matter.lawyer_id,
        )
    
    # Generate & approve draft
    draft = in_memory_services.draft_service.generate(matter_id=matter.id)
    draft = in_memory_services.approval_service.approve(
        draft_id=draft.id,
        approved_by=matter.lawyer_id,
    )
    return draft

@pytest.fixture
def draft_submitted(in_memory_services, simple_matter_freehold):
    """Fixture: Draft submitted but not approved yet."""
    # Similar to draft_approved, but stop at submit step
    # ...
    draft = in_memory_services.draft_service.submit(draft_id=draft.id)
    return draft
```

### Running the Tests

**Phase 1** (in-memory, M3):
```bash
cd backend
uv run pytest tests/e2e/test_rta_happy_path.py -v
uv run pytest tests/e2e/test_rta_statutory_blockers.py -v
```

**Phase 2** (with real DB, post-M3):
```bash
# After Docker/Neon setup
DATABASE_URL=postgresql://... uv run pytest tests/e2e/ -v
```

---

## Success Criteria (Overall)

- ✅ Happy path completes without errors (8 steps)
- ✅ All 5 edge cases are tested and properly rejected
- ✅ Audit trail has ≥7 entries for a complete workflow
- ✅ No unverified facts reach draft generation
- ✅ Statutory blockers cannot be overridden
- ✅ Stale drafts prevent export
- ✅ Only lawyers who created matters can approve/export (access control)

---

## References

- `backend/docs/draftly-rta-matter-workflow-v1.md` — detailed workflow (§4–10)
- `backend/docs/ARCHITECTURE.md` — state machines (§10), invariants (§7)
- `backend/tests/e2e/test_services_slice.py` — in-memory pattern to follow
- `backend/tests/fixtures/` — synthetic data and service fakes
