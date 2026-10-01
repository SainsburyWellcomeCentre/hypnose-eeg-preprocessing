"""Pipeline stages run end to end: preprocessing, sleep scoring, QC, batch runs, and review.

Each stage module runs the matching `hypnose_eeg` CLIs in this process and is
itself a CLI (`python -m hypnose_eeg.pipeline.run`, or the `hypnose-eeg-*`
console scripts). `hypnose_eeg.api` is the keyword-argument facade over them.
"""
