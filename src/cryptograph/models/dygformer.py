"""DyGFormer adapter (placeholder).

DyGFormer (Yu et al., NeurIPS 2023) is the Transformer-based temporal graph model in
the model plan. The intended route is an adapter from our TGN event stream
(`data/processed/<version>/tgn_events.pt`) to DyGLib's input format, so the official
DyGLib implementation can be trained and evaluated on our edge-level contagion labels.
Whether that route works, and what the adapter must do, is assessed in the model-plan
feasibility check (docs/model_plan.md). Nothing is implemented here yet.
"""
