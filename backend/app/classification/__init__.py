"""ModelNet10 point-cloud object classification (DS 7400 HW2, extended in HW3).

One class per complete object. ``dataset`` downloads and preprocesses
ModelNet10, ``models`` defines the shallow baseline, Mini-PointNet and the HW3
pooling variants (``hw3`` aggregates and diagnoses the HW3 experiments),
``training`` runs the optimisation loop, ``inference`` serves predictions and
``jobs`` runs GUI-launched training in the background.
"""
