# Application and service development

Version 0.1.0 · 20 October 2026. This is a development plan. No customer demand, paid pilot, partnership, price or domain performance is established by this document.

## Current product

Interval is a working single-user research application. It compares conditional estimators, trains a small neural estimator, runs a small quantum circuit simulation, calibrates detector settings on separate observations and exports evidence. A locally adapted open-weight language model selects among admitted procedures; native code executes them. Its instruction evaluation covers synthetic requests for those procedures.

The included numerical evidence uses synthetic observations. The medical and infrastructure profiles configure the shared application; they are not validated domain products or separate completed implementations. The current local import accepts positive 2D intensity arrays. DICOM ingestion, volumetric processing, geospatial registration and sensor-specific calibration require additional adapters and validation.

The initial service offering is a reproducible evaluation and integration engagement. A customer should pay for a useful, supported workflow and its evidence, rather than an assumed quantum improvement. The application must retain competitive classical methods even when they outperform the learned alternatives. See the [research protocol](research-protocol.md), [model card](model-card.md) and [deployment guide](deployment.md).

## Three application tracks

| Track | Candidate first buyer and user | Initial service hypothesis | Required evidence before a domain claim |
| --- | --- | --- | --- |
| Tissue research | Imaging laboratory or CRO research lead; imaging analyst as user | Reproducible image-pair analysis and repeatability assessment | Independent specimens or patients, agreed image preparation, specialist annotations and comparison with the existing research workflow |
| Infrastructure monitoring | Asset operator, maintenance contractor or geospatial integrator; inspection analyst as user | Passive change screening to help review maintenance or post-event inspection evidence | Licensed acquisitions, geometry and registration controls, independent inspection records and a held-out site/date evaluation |
| Client workspace | Research or engineering group with an existing numerical task; technical lead as buyer | Local installation, domain adapter, bounded procedure selection and portable evidence | Customer-owned acceptance fixtures, reproducible model restoration and evaluation on independent data from the intended task |

These are candidate buyer roles, not qualified leads. Discovery must establish who owns the problem, what decision follows an output, how frequently the workflow runs, what failure costs and what budget owner can approve a pilot.

## Tissue research pilot

Choose one research question, imaging modality and unit of analysis. The first study can assess repeatability or image change measurement; identification of a therapeutic protein, selective delivery to cells and treatment decisions are separate research programmes.

A suitable partner supplies image preparation expertise, permission to process the observations and an independent assessment protocol. [RIDER Lung CT](https://wiki.cancerimagingarchive.net/display/Public/RIDER%2BLung%2BCT) offers same-day repeat acquisitions for a possible repeatability study. Such pairs can test apparent change under repeated measurement; they cannot establish treatment-response accuracy. No-change cases are a false-alarm control and need a suitable calibration rule rather than an F1-only selection on empty positive labels.

Before receiving the held-out outcomes, agree the following pilot contract:

- One primary endpoint: for example, false alarms on repeat acquisitions, measurement agreement, or change-mask quality where independent labels exist. Select the endpoint for the actual research task.
- The current laboratory method and at least one simple numerical baseline, with identical observation access and documented preparation.
- Patient/specimen and, where available, site-level separation. Fix exclusions, annotation handling and uncertainty reporting before evaluation.
- A clinically or experimentally interpretable acceptance margin chosen with the specialist, a compute/latency budget and a maximum analyst review burden.
- A blinded assessment and an export that the partner can reproduce from its own permitted data.

Proceed if the agreed endpoint or workflow benefit is met without unacceptable failure modes. Stop or narrow the offering if the simple baseline is sufficient, annotations cannot support the claim, or preprocessing dominates the result. A successful research pilot does not qualify diagnostic or therapeutic use.

## Infrastructure monitoring pilot

Select one asset category and one maintenance or post-event review decision. Start with retrospective paired acquisitions and independent inspection evidence. Satellite tasking, imagery purchases and continuous operations are outside the present reference application.

[Capella's open collection](https://www.capellaspace.com/legal/data-licensing) provides a possible research source under its stated attribution terms. The presence of open imagery does not establish that suitable repeat acquisitions, matching labels or the paper's exact study chips are available. No relationship with Capella or IonQ is asserted.

Freeze a site/date split, the comparison workflow, a tolerable false-alert burden and the minimum useful detection rate before examining test labels. Report review workload, missed annotated changes, performance by acquisition conditions and the cost of registration and analyst review. Use a silent retrospective evaluation before allowing results to influence operational inspections. Continue only if the accepted improvement survives held-out sites or dates and can be delivered within the customer's workflow budget.

This track concerns passive infrastructure protection, maintenance and inspection. The application does not implement weapon targeting, terminal guidance or autonomous engagement.

## Editable client workspace

The editable profile supplies project identity and a common execution interface. Local observation import trains the numerical estimator; instruction adaptation teaches the language model to select existing procedures. These are different training operations.

A customer can run the documented installation, import correctly prepared arrays, reproduce a procedure and adapt instruction terminology. New domains, new tools and new output types require implementation and tests. The software does not automatically discover a valid study design from arbitrary files.

Proposed engineering acceptance is: every agreed valid fixture completes with an inspectable result; every agreed invalid fixture receives an intelligible rejection; a restored inference bundle reproduces the original predictions within a declared numerical tolerance; no procedure executes a tool outside its admitted contract. Measure these on the customer's target hardware. Assess instruction selection on independently written requests, including ambiguity and unsupported tasks, rather than training-template variants alone.

## Revenue and delivery

New Interval source and synthetic materials are released under Apache 2.0; upstream assets and external datasets retain their own terms. The open reference can support paid work through:

- Installation and packaging for the customer's approved compute environment.
- Data adapters, preparation pipelines and reproducible independent evaluations.
- Local instruction adaptation, terminology evaluation and supported model updates.
- Workflow integration, analyst training, maintenance and agreed support obligations.

Managed hosting is a later option. Authentication, tenant separation, retention, incident handling and service objectives require a separate implementation and acceptance exercise. The public synthetic demonstration is not a multi-tenant institutional service.

Quote only after a written scope identifies integration effort, data rights, compute, ongoing support and the responsible buyer. Track delivery time, compute consumption, support burden and the measured customer outcome. No price, revenue projection or market-size estimate is supplied here.

## Milestones and decisions

| Gate | Deliverable | Go criterion | Stop or revise when |
| --- | --- | --- | --- |
| Reference release | Executable source, documented limits, measured evidence and review record | Installation and reproduction pass on the declared environment | Results cannot be reproduced or claims exceed evidence |
| Problem qualification | One use case, workflow owner, buyer and existing baseline | Partner supplies a testable problem and a route to permitted data | Interest has no owner, usable data or decision consequence |
| Pilot agreement | Dataset manifest, frozen split, endpoints, margins and resource budget | Both sides accept the evaluation and commercial scope before test results | Acceptance depends on selecting favourable test cases |
| Independent pilot | Raw outcomes, uncertainty, failures and review time | Agreed scientific and workflow criteria are met | Benefit disappears against a viable baseline or under independent data |
| Supported deployment | Restore/recovery checks, support scope and intended-use boundary | Customer acceptance on its approved environment | Support, data or infrastructure costs exceed the agreed service scope |

Data partnerships should specify acquisition metadata, permitted processing, annotation ownership, retention, redistribution of derived artifacts and permission to publish aggregate results. Sharing source code does not imply permission to publish customer observations or fitted intensity distributions.

Publish complete methods and limitations with the evidence. Link the repository, model card, application and research article to one another. Community posts should disclose authorship, explain a concrete result and follow the venue's posting rules. This plan records no outreach, email or social publication.
