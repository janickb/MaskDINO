# Specification: [Feature Name]

## 1. Context & Goals
* **Problem:** Briefly describe the issue this feature solves.
* **Objective:** What does success look like in 1-2 sentences?
* **User Story:** As a [user], I want to [action] so that [benefit].

## 2. Technical Architecture & Data Flow
*High-level overview of how the feature plugs into the existing codebase.*

### Components impacted:
- `src/components/NewFeature.tsx` (To be created)
- `src/server/router.ts` (To be modified)

### Data Models / Schema:
```typescript
// Define exact types or DB schemas to prevent the AI from guessing field names
interface FeaturePayload {
  id: string;
  createdAt: string;
  status: 'pending' | 'success' | 'failed';
}
```

## 3. Core Requirements & Logic
*Be incredibly literal here. Use concrete if/then logic.*

- [ ] **Input Validation:** Reject payloads if `id` is missing with a `400 Bad Request`.
- [ ] **State Machine:** Moving status from `pending` to `success` must trigger an email alert.
- [ ] **Edge Cases:** If the network drops during the transaction, retry exactly 3 times before failing.

## 4. API Endpoints / Interface Definitions
*Define the exact contract the AI must build.*

- **Path:** `POST /api/v1/feature`
- **Request Body:** `FeaturePayload`
- **Success Response:** `200 OK` -> `{ success: true }`
- **Error Responses:** `401 Unauthorized`, `422 Unprocessable Entity`

## 5. Definition of Done (DoD) & Testing
*AI will stop coding once these criteria are met.*

- [ ] Unit tests cover all 3 status changes (`pending`, `success`, `failed`).
- [ ] Test coverage for the new router file is >90%.
- [ ] No `any` types used (Strict TypeScript compliance).
- [ ] The feature passes `npm run build` without warnings.

## 6. Implementation Checklist (The TODO)
*The step-by-step roadmap for the AI.*

- [ ] **Phase 1: Database & Types**
  - [ ] Create the database migration for the new table.
  - [ ] Export the updated Prisma types.
- [ ] **Phase 2: Backend API**
  - [ ] Implement the `POST /api/v1/feature` endpoint.
  - [ ] Add input validation middleware.
- [ ] **Phase 3: Frontend UI**
  - [ ] Build the interactive form component.
  - [ ] Connect the form to the backend API using React Query.
