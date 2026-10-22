# VEIL Vision Audit Taxonomy & Vision-Language (VLM) Pipeline (`docs/VEIL_VISION_PIPELINE.md`)

This document details the **Vision-Language (VLM) Pipeline Track (`VISION_VLM`)**, the **VEIL Vision Privacy Hazard Auditing Taxonomy**, the **10,000–15,000 Multi-Turn Split Strategy**, the **Vision Tokenizer & Patch Density Profiler**, and the **25 Curated HuggingFace Vision Dataset Blends**.

---

## 1. VEIL Vision Privacy Hazard Auditing Taxonomy

Small multimodal models (`LFM2.5-VL-450M`, `LFM2.5-VL-1.6B`, `Qwen3.5-VL-0.8B`, `Qwen3.5-VL-2B`, and `Gemma4-E2B-Vision`) deployed for creator safety and visual privacy auditing must detect subtle background leaks while maintaining a **false-positive rate below `<2.0%`** on clean studio sets.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VEIL VISION AUDIT TAXONOMY                      │
├───────────────────────────────┬────────────────────────────────────────┤
│ 1. Residential & Geolocation  │ Envelopes, packages, street/apt signs  │
│ 2. Medical, Legal & Financial │ Rx bottles, boarding passes, ID cards  │
│ 3. Specular Crew & Gear Leaks │ Mirror reflections, whiteboards, rigs  │
│ 4. Identifying Biometric Ink  │ Distinctive tattoos, identifiable scars│
│ 5. Clean Baseline (Negatives) │ Zero-hazard studio sets, safe props    │
└───────────────────────────────┴────────────────────────────────────────┘
```

### Task 1: Physical Doxxing & Residential Geolocation Detection (`25% Split`)
- **First-Class Mail & Envelopes**: Envelopes on side tables, counters, or desks showing sender return addresses (IRS, state tax board, utility companies) or recipient names and street addresses.
- **Shipping Packages & Labels**: Amazon/FedEx/UPS shipping boxes showing printed thermal labels, tracking barcodes, QR codes, and destination addresses.
- **Signage & Architectural Markers**: Visible apartment door numbers (`"Apt 4B"`), complex nameplates, rearview mirror parking permits, or street intersection signs visible through open windows.
- **Keychains & Fobs**: Brass/plastic key tags labeled with unit numbers or property management names (`"1408 KEY"`).
- **Default Remediation Action**: `"suggested_action": "REDACT"` (or `"BLUR"`).

### Task 2: Medical, Travel & Financial PII Extraction (`20% Split`)
- **Prescription Medication Bottles**: Amber pill vials, blister packs, and topical tubes showing pharmacy labels with patient legal names, medication names (SSRIs, HRT, specialty treatments), dosage, and Rx reference numbers.
- **Travel & Airline Stubs**: Physical or mobile boarding passes and baggage claim tags showing passenger name, flight number, 6-character PNR/record locator, and date.
- **Hotel Key Sleeves**: Branded keycard jackets showing guest surname, room number, and checkout dates (`"The Ritz-Carlton LA — Room 1408 — Ms. Vance"`).
- **Financial Instruments**: Debit/credit cards, bank statements, checks, and wire transfer receipts showing routing numbers or partial PANs.
- **Default Remediation Action**: `"suggested_action": "REDACT"`.

### Task 3: Specular Reflection & Off-Camera Crew/Production Exposure (`25% Split`)
- **Mirror Reflections**: Full-length mirrors, bathroom vanities, or polished metal fixtures reflecting camera operators, lighting technicians, boom mics, or cinema camera rigs.
- **Screens & Glass Panels**: Blacked-out TV monitors, tablet screens, or glass doors reflecting off-camera room contents.
- **Confidential Set Whiteboards**: Whiteboards or pinned call sheets listing confidential shoot titles (`"Project Velvet — Call Time 0800"`), director legal names, NDAs, or schedules.
- **Unconsenting Third Parties**: Passersby, hotel housekeeping staff, or background individuals visible through windows or open doorways.
- **Default Remediation Action**: `"suggested_action": "BLUR"` or `"REDACT"`.

### Task 4: Identifying Biometric Markings — Tattoos & Scars (`10% Split`)
- **Distinctive Tattoos**: Custom sleeve art, chest/collarbone script, wrist/ankle coordinates, dates, names, or recognizable geometric/illustrative ink on exposed skin.
- **Identifiable Scars & Birthmarks**: Prominent surgical scars or unique skin markings that allow cross-account re-identification.
- **Required Remediation Routing**: Explicitly outputs `"suggested_action": "BLUR"` or `"MOSAIC"` with tight `[0, 1000]` bounding boxes `[x1, y1, x2, y2]`.
- **Required Hard Negatives**: Temporary metallic jewelry, body chains, lace garment shadow cast lines, window blind shadow stripes, and washable body paint.

### Task 5: Clean Invariant Baseline — True Negatives (`20% Split` · `<2% FPR`)
- **Zero-Hazard Studio Sets**: Professionally staged sets containing blank prop envelopes, unbranded amber glass apothecary bottles, clean mirrors reflecting only empty walls or softboxes, and generic hotel decor with zero location markers.
- **Required Remediation Routing**: Emits `"hazard_detected": false`, `"hazards": []`, and `"suggested_action": "SAFE_PASS"`.

---

## 2. Dataset Composition & Diversity Protocol (10,000–15,000 Multi-Turn Pairs)

| Category | Allocation % | Target Pair Count | Primary Remediation |
| :--- | :--- | :--- | :--- |
| **1. Physical Mail & Shipping Labels** | **25%** | `3,000 – 3,500` | `REDACT` |
| **2. Medical, Travel & Financial PII** | **20%** | `2,500 – 3,000` | `REDACT` |
| **3. Specular Reflections & Crew** | **25%** | `3,000 – 3,500` | `BLUR` / `REDACT` |
| **4. Biometric Ink & Scars** | **10%** | `1,200 – 1,500` | `BLUR` / `MOSAIC` |
| **5. Clean Invariant Negatives** | **20%** | `2,500 – 3,000` | `SAFE_PASS` (`<2% FPR`) |
| **Total Corpus** | **100%** | **`12,200 – 14,500`** | Multi-Turn Grounded |

### Camera Angle & Distance Diversity
- **40% Close-Up / Macro Shots**: High-resolution crops of tables, nightstands, bathroom counters, and skin surfaces where text/ink occupies `5%–25%` of the frame.
- **40% Medium Room Shots**: Standard portrait/boudoir framing (waist-up or full-body) where hazards sit in the background or periphery occupying `1%–5%` of the frame.
- **20% Wide Environmental Shots**: Full room views with small, distant, or partially occluded hazards (`<1%` of the frame) requiring high-resolution NaViT tiling.

### Lighting, Bokeh & Occlusion Augmentation Rules
1. **Motion Blur & Out-of-Focus Bokeh**: Shallow depth-of-field (`f/1.4–f/2.8`) background pill bottles and envelopes that remain partially legible.
2. **Low-Light & Color Casts**: Warm tungsten bedroom lamps (`2700K`), pink/blue RGB neon strips, or harsh bathroom vanity sidelighting.
3. **Partial Occlusion (`30%–50%`)**: Envelopes tucked under laptops, pill bottles behind makeup brushes, or tattoos partially covered by sheer fabric.

---

## 3. Vision Tokenizer & Patch Density Profiler

Under dynamic resolution encoders (`Qwen3.5-NaViT` and `LFM2.5-SigLIP`), an image of resolution $W \times H$ clamped to $[\text{min\_pixels}, \text{max\_pixels}]$ with a $14\times 14$ patch size and $2\times 2$ spatial merger consumes:

$$\text{VisualPatchTokens} = \text{round}\left(\frac{\text{clamp}(W \times H, \text{min\_pixels}, \text{max\_pixels})}{28 \times 28}\right)$$

The **Image-to-Text Token Ratio** is defined as:

$$\text{Ratio} = \frac{\text{VisualPatchTokens}}{\text{TextTokens}}$$

- **`< 3.5x` (Compact Edge Tile)**: Fast connector supervision (`256–512` patches).
- **`3.5x – 5.0x` (Sweet-Spot Sub-2B Ratio)**: Optimal balance between visual grounding and text cross-entropy loss (`512–800` patches).
- **`5.0x – 6.2x` (High-Density NaViT Grid)**: Suitable for dense OCR and wide-angle VEIL scans (`800–960` patches).
- **`> 6.2x` (Excessive NaViT Patch Dilution)**: Visual tokens dominate attention and dilute text supervision; use the **Cap Excessive NaViT Patches** button in `VisionDatasetStudioView` to down-tile outliers automatically.

---

## 4. Complete Catalog of 25 Curated Vision-Language Dataset Blends

| # | Blend ID | Task Category | Title & Primary HuggingFace Repositories |
| :--- | :--- | :--- | :--- |
| 1 | `veil-blend-master-15k` | `VEIL_PRIVACY_AUDIT` | **VEIL Master Audit Blend (12.5k Split)** — `Docmatix-Sub2B-Clean` (25%), `ocr-vqa-200k` (20%), `ShareGPT4V-Region` (25%), `Molmo-PixMo-BBox` (10%), `RealWorldQA` (20%) |
| 2 | `veil-blend-task1-geolocation` | `VEIL_PRIVACY_AUDIT` | **VEIL Task 1: Physical Doxxing, Envelopes & Street Signage (25%)** — `Docmatix` (40%), `cord-v2` (30%), `InternVL-SceneText-StreetView` (30%) |
| 3 | `veil-blend-task2-medical-financial` | `VEIL_PRIVACY_AUDIT` | **VEIL Task 2: Prescription Vials, Boarding Passes & Key Sleeves (20%)** — `ocr-vqa-200k` (45%), `DocVQA-FinePrint-PII` (35%), `cord-v2` (20%) |
| 4 | `veil-blend-task3-specular-crew` | `VEIL_PRIVACY_AUDIT` | **VEIL Task 3: Specular Reflections, Mirror Crew Leaks & Whiteboards (25%)** — `ShareGPT4V-Region` (40%), `Molmo-PixMo-Points` (35%), `ChartQA-Whiteboard` (25%) |
| 5 | `veil-blend-task4-biometric-ink` | `VEIL_PRIVACY_AUDIT` | **VEIL Task 4: Biometric Markings — Tattoos & Scars (`BLUR`/`MOSAIC` · 10%)** — `Molmo-PixMo-BBox` (45%), `ShareGPT4V-Region` (30%), `RealWorldQA` (25%) |
| 6 | `veil-blend-task5-clean-negatives` | `VEIL_PRIVACY_AUDIT` | **VEIL Task 5: Clean Invariant Baseline (`<2%` FPR · 20%)** — `RealWorldQA` (40%), `ShareGPT4V-Clean-Interiors` (35%), `the_cauldron-negatives` (25%) |
| 7 | `vlm-blend-ocr-doc` | `OCR_DOCUMENT` | **High-Density OCR, Invoices & Document JSON Extraction** |
| 8 | `vlm-blend-handwritten-forms` | `OCR_DOCUMENT` | **Handwritten Clinical Intake, Cursive Notes & Archival Paleography OCR** |
| 9 | `vlm-blend-multilingual-packaging` | `OCR_DOCUMENT` | **Multilingual Retail Packaging, Nutrition Facts & Curved Barcode OCR** |
| 10 | `vlm-blend-latex-molecular` | `OCR_DOCUMENT` | **Academic PDF Equations, LaTeX Matrices & Chemical Markush Structures** |
| 11 | `vlm-blend-contract-redaction` | `OCR_DOCUMENT` | **Legal Contracts, KYC Identity Documents & Automated Redaction Grounding** |
| 12 | `vlm-blend-ui-grounding` | `UI_GROUNDING` | **UI Screen Grounding, Bounding Boxes & Autonomous GUI Action** |
| 13 | `vlm-blend-mobile-android` | `UI_GROUNDING` | **Mobile Android/iOS Navigation, Swipe Trajectories & Icon Grounding** |
| 14 | `vlm-blend-qa-visual-regression` | `UI_GROUNDING` | **Automated Web QA, CSS Layout Overflow & Z-Index Collision Detection** |
| 15 | `vlm-blend-cad-floorplan` | `UI_GROUNDING` | **Architectural Floorplans, P&ID Engineering Blueprints & CAD Symbol Grounding** |
| 16 | `vlm-blend-chart-science` | `CHART_DIAGRAM` | **Scientific Charts, Financial Plots & Architecture Diagrams** |
| 17 | `vlm-blend-financial-candlestick` | `CHART_DIAGRAM` | **SEC 10-K Earnings Charts, OHLCV Candlesticks & Waterfall Plot Analytics** |
| 18 | `vlm-blend-radiology-pathology` | `CHART_DIAGRAM` | **Clinical Radiology (CXR / CT) & Histopathology Slide Grounded VQA** |
| 19 | `vlm-blend-satellite-geospatial` | `CHART_DIAGRAM` | **Satellite Remote Sensing, SAR Maritime Tracking & Aerial Land-Use VQA** |
| 20 | `vlm-blend-visual-cot` | `VISUAL_COT_MATH` | **Visual Chain-of-Thought (`<think>`) & Multimodal Geometry/Math** |
| 21 | `vlm-blend-physics-circuits` | `VISUAL_COT_MATH` | **STEM Free-Body Mechanics, Ray Optics & Digital Logic Gate Schematics** |
| 22 | `vlm-blend-algorithmic-visual-iq` | `VISUAL_COT_MATH` | **Abstract Raven Matrices, 3D Mental Rotation & Spatial Logic Puzzles** |
| 23 | `vlm-blend-edge-spatial` | `SPATIAL_EDGE_VQA` | **Edge Robotics, Spatial Depth & Industrial Defect VQA (450M / 0.8B)** |
| 24 | `vlm-blend-autonomous-driving` | `SPATIAL_EDGE_VQA` | **Autonomous Driving Dashcam, Traffic Sign OCR & Ego-Lane Hazard Grounding** |
| 25 | `vlm-blend-hallucination-guard` | `SPATIAL_EDGE_VQA` | **Hallucination Suppression, Object Absence Verification & Adversarial POPE/RLHF-V** |
