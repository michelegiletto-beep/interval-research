# Installation and deployment

## Supported delivery

Interval 0.1.0 is delivered as editable source and a Docker reference application. It is a single-user research service, not a qualified institutional multiuser system. It has no authentication layer. Bind locally to 127.0.0.1 for own-data work. Use an institution's established authenticated gateway and an independently reviewed deployment design before broader access.

The local CLI uses `python -m interval.cli serve --port 8765`. All model inference happens locally after the pinned base-weight download. No OpenAI key is used. A base model without the supplied adapter can still run, but its measured procedure-selection accuracy is substantially lower; inspect the installed model card and receipt.

## Docker demonstration

The public Hugging Face Space runs the same backend and interface in synthetic-only mode through the Gradio SDK, with a Gradio diagnostics panel alongside the FastAPI interface. The pinned public base weights download during first startup; the adapted weights are included in the source package. Inference does not call an external model service. Public mode rejects own-data uploads, large configurations and multi-run sweeps. It admits at most one job per 20 seconds, has one execution worker and two admitted slots, retains at most 20 completed generated records on disk and eight in memory. Public run IDs are unguessable capability links; prompts are not listed to other visitors. Do not enter confidential information into a public demonstration.

The separate root Dockerfile is an optional self-hosted reference. It downloads the pinned base during image construction. Hugging Face currently gates Docker creation behind a paid plan; this release uses the account's available free Gradio/ZeroGPU path without purchasing hosting. Numerical work and the small language model execute on CPU; no GPU allocation is requested.

Environment variables:

| Name | Purpose |
| --- | --- |
| INTERVAL_PUBLIC_DEMO=1 | Enable synthetic-only public budgets and ephemeral record retention |
| INTERVAL_RUNTIME | Directory for generated local job records |
| INTERVAL_ALLOWED_HOSTS | Explicit additional hostnames, comma-separated |

Default local hosts are localhost, 127.0.0.1 and ::1. Public mode also admits *.hf.space. Cross-origin writes are rejected. The runtime does not send messages or publish results externally.

## Local input preparation

The NPZ schema is listed in the root README. Splits must reflect independent patients/scenes/dates according to the study. The importer checks byte budgets, NPY headers, numeric shapes, finite positive observations, binary masks and identical observation identities. Those checks cannot detect near duplicates, overlapping regions or hidden patient identity overlap.

All-negative calibration masks select the no-change detector when all F1 values tie, preferring fewer false alarms. Such a run supports negative-control inspection, not sensitivity estimation.

## Delivery acceptance

Run the tests, reproduce the included reference, complete an interface-driven procedure and restore a saved inference bundle. Confirm that the intended data and compute remain local. For a new domain, define an external-data acceptance study and preserve the distinction between research outputs and clinical/field decisions.
